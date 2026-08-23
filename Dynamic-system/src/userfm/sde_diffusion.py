import abc

import jax
import jax.numpy as jnp

from userfm import cs, utils


class StructuredCovariance(abc.ABC):
    """Abstract base class for noise covariance matrices defined implicitly.

    The class is organized as a bijector, mapping white noise to structured
    noise, with forward, inverse, and logdet methods just like the interface
    of a normalizing flow.
    (StructuredCovariance is an instance of a rudimentary normalizing flow.)
    """

    @classmethod
    def forward(cls, v):
        """Multiplies the input vector v by Sigma^{1/2}."""
        raise NotImplementedError

    @classmethod
    def inverse(cls, v):
        """Multiplies the input vector v by Sigma^{-1/2}."""
        raise NotImplementedError

    @classmethod
    def logdet(cls, shape):
        """Computes the log determinant logdet(Sigma^{1/2})."""
        raise NotImplementedError

    @classmethod
    def sample(cls, key, shape):
        """Sample the structured noise by compting Sigma^{1/2}z."""
        return cls.forward(jax.random.normal(key, shape))


class Identity(StructuredCovariance):
    """Identity covariance matrix (equivalent to white noise)."""

    @classmethod
    def forward(cls, v):
        return v

    @classmethod
    def inverse(cls, v):
        return v

    @classmethod
    def logdet(cls, shape):
        return jnp.zeros(shape[0])


class FourierCovariance(StructuredCovariance):
    """Base class for covariance matrices which are diagonal in Fourier domain.

    Subclasses must implement spectrum(f) classmethod (of Sigma^{1/2})
    """

    @classmethod
    def spectrum(cls, f):
        """The spectrum (eigenvalues) of the Fourier covariance of Sigma^{1/2}."""
        raise NotImplementedError

    @classmethod
    def forward(cls, v, invert=False):
        """Maps v -> Sigma^{1/2}v.

        Args:
          v: of shape (b,n,c) or (b,h,w,c).
          invert: whether to use inverse transformation

        Returns:
          Sigma^{1/2}v
        """
        assert all(
            k % 2 == 0 for k in v.shape[1:-1]
        ), "requires even lengths for fft for now"
        f = jnp.sqrt(
            sum(jnp.meshgrid(*[jnp.fft.rfftfreq(k) ** 2 for k in v.shape[1:-1]]))
        )

        scaling = cls.spectrum(f)
        assert scaling.shape == f.shape, "cls.spectrum should output same shape"

        if invert:
            scaling = 1 / scaling
        if len(v.shape) == 3:
            scaled_fft_v = jnp.fft.rfft(v, axis=1) * scaling[None, :, None]
            return jnp.fft.irfft(scaled_fft_v, axis=1)
        elif len(v.shape) == 4:
            scaled_fft_v = jnp.fft.rfft2(v, axes=(1, 2)) * scaling[None, :, :, None]
            return jnp.fft.irfft2(scaled_fft_v, axes=(1, 2))
        else:
            raise NotImplementedError

    @classmethod
    def inverse(cls, v):
        """Maps v -> Sigma^{-1/2}v.

        Args:
          v: of shape (b,n,c) or (b,h,w,c).

        Returns:
          Sigma^{-1/2}v
        """
        return cls.forward(v, invert=True)

    @classmethod
    def logdet(cls, shape):
        """Assumes input shape is (b,n,c) or (b,h,w,c) for 2d."""
        f = jnp.sqrt(sum(jnp.meshgrid(*[jnp.fft.fftfreq(k) ** 2 for k in shape[1:-1]])))
        return jnp.log(cls.spectrum(f)).sum() * shape[-1] + jnp.zeros(shape[0])


class WhiteCovariance(FourierCovariance):
    """White Noise Covariance matrix, equivalent to Identity."""

    multiplier: float = 1.0

    @classmethod
    def spectrum(cls, f):
        return jnp.ones_like(f) * cls.multiplier


class BrownianCovariance(FourierCovariance):
    """Brown Noise Covariance matrix: (1/f) spectral noise."""

    multiplier: float = 30.0  # Tuned scaling to use same scale as Identity

    @classmethod
    def spectrum(cls, f):
        scaling = jnp.where(f == 0, jnp.ones_like(f), 1.0 / f)
        scaling = scaling / jnp.max(scaling)
        return jnp.where(f == 0, jnp.ones_like(f), scaling) * cls.multiplier


class PinkCovariance(FourierCovariance):
    """Pink Noise Covariance matrix: 1/sqrt(f) spectral noise."""

    multiplier: float = 1.0  # Tuned scaling to use same scale as Identity

    @classmethod
    def spectrum(cls, f):
        scaling = jnp.where(f == 0, jnp.ones_like(f), 1 / jnp.sqrt(f))
        scaling = scaling / jnp.max(scaling)
        return jnp.where(f == 0, jnp.ones_like(f), scaling) * cls.multiplier


class Diffusion(abc.ABC):
    """Abstract class for diffusion types.

    Subclasses must implement sigma(t) and scale(t)
    """

    def __init__(self, cfg, covariance=Identity):
        self.cfg = cfg
        self.covsqrt = covariance

    @property
    def tmin(self):
        return self.cfg.time_min

    @property
    def tmax(self):
        return self.cfg.time_max

    def sigma(self, t):
        """Noise schedule."""
        raise NotImplementedError

    def scale(self, t):
        """Scale schedule."""
        raise NotImplementedError

    def f(self, t):
        """Internal f func from https://arxiv.org/abs/2011.13456."""
        return jax.grad(lambda s: jnp.log(self.scale(s)))(t)

    def g2(self, t):
        """Internal g^2 func from https://arxiv.org/abs/2011.13456."""
        dsigma2 = jax.grad(lambda s: self.sigma(s) ** 2)(t)
        return dsigma2 - 2 * self.f(t) * self.sigma(t) ** 2

    def dynamics(self, score_fn, x, t):
        """Backwards probability flow ODE dynamics."""
        return self.f(t) * x - 0.5 * self.g2(t) * score_fn(x, t)

    def drift(self, score_fn, x, t):
        """Backwards SDE drift term."""
        return self.f(t) * x - self.g2(t) * score_fn(x, t)

    def diffusion(self, score_fn, x, t):  # pylint: disable=unused-argument
        """Backwards SDE diffusion term (independent of score_fn)."""
        return jnp.sqrt(self.g2(t))

    def noise_score(self, xt, x0, t):
        r"""Actually the score times the cov matrix. `\Sigma\nabla\logp(xt)`."""
        s, sig = self.scale(t), self.sigma(t)
        return -(xt - s * x0) / sig**2

    def noise_input(self, x, t, key):
        """Apply the noise at scale sigma(t) and with covariance to the input."""
        s, sig = self.scale(t), self.sigma(t)
        return s * x + sig * self.noise(key, x.shape)

    def noise(self, key, shape):
        """Sample from the structured noise covariance (without scale sigma(t))."""
        return self.covsqrt.sample(key, shape)


class VarianceExploding(Diffusion):
    """Variance exploding variant of Score-SDE diffusion models."""

    def __hash__(self):
        return hash(id(self))

    def sigma(self, t):
        # Taos: similar to Eqn.(31) in
        # Song et al. 2021, "Score-Based Generative Modeling through Stochastic Differential Equations"
        # The difference is that here we subtract 1 in the sqrt.
        # This appears to fix the discontinuity at 0 of Eqn.(31).
        return self.cfg.sigma_min * jnp.sqrt((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)

    def dsigma(self, t):
        return (
            self.cfg.sigma_min
            * jnp.log(self.cfg.sigma_max / self.cfg.sigma_min)
            * (self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t)
            / jnp.sqrt((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)
        )

    def g2(self, t):
        return (
            2 * self.cfg.sigma_min**2
            * jnp.log(self.cfg.sigma_max / self.cfg.sigma_min)
            * (self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t)
        )

    def scale(self, t):
        return jnp.ones_like(t)


def int_b(t):
    """Integral b(t) for Variance preserving noise schedule."""
    bm = 0.1
    bd = 20
    return bm * t + (bd - bm) * t**2 / 2


class VariancePreserving(Diffusion):
    tmin = 1e-4

    def sigma(self, t):
        return jnp.sqrt(1 - jnp.exp(-int_b(t)))

    def scale(self, t):
        return jnp.exp(-int_b(t) / 2)


class SubVariancePreserving(Diffusion):
    tmin = 1e-4

    def sigma(self, t):
        return 1 - jnp.exp(-int_b(t))

    def scale(self, t):
        return jnp.exp(-int_b(t) / 2)


def get_sde_diffusion(cfg):
    if isinstance(cfg, cs.SDEVarianceExploding):
        return VarianceExploding(cfg)
    else:
        raise ValueError(f'Unknown sde diffusion: {cfg}')
