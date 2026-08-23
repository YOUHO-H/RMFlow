import jax.numpy as jnp

from userfm import cs


class Lorenz:
    threshold = -.6

    @staticmethod
    def statistic(x):
        fourier_magnitudes = jnp.abs(jnp.fft.rfft(x[..., 0], axis=-1))
        return -fourier_magnitudes[..., 1:].mean(-1)

    @staticmethod
    def constraint(x):
        return  Lorenz.statistic(x) - Lorenz.threshold


class FitzHughNagumo:
    threshold = 2.5

    @staticmethod
    def statistic(x):
        return jnp.max(x[..., :2].mean(-1), -1)

    def constraint(x):
        return FitzHughNagumo.statistic(x) - FitzHughNagumo.threshold


def get_event_constraint(cfg_dataset):
    if isinstance(cfg_dataset, cs.DatasetGaussianMixture):
        return None
    elif isinstance(cfg_dataset, cs.DatasetLorenz):
        return Lorenz
    elif isinstance(cfg_dataset, cs.DatasetFitzHughNagumo):
        return FitzHughNagumo
    elif isinstance(cfg_dataset, cs.DatasetSimpleHarmonicOscillator):
        return None
    elif isinstance(cfg_dataset, cs.DatasetBatchShape):
        return None
    elif isinstance(cfg_dataset, cs.DatasetPendulum):
        return None
    else:
        raise ValueError(f'Unknown dataset: {cfg_dataset}')
