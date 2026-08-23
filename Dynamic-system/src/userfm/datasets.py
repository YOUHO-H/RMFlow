# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for generating datasets based on general ODEs and hamiltonian systems."""

import abc
import functools

import hydra
from omegaconf import OmegaConf
import jax
from jax import device_count
from jax import grad
from jax import jit
from jax import vmap
from jax.experimental import mesh_utils
from jax.experimental.ode import odeint
from jax.experimental.pjit import pjit
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
from tqdm.auto import tqdm

from userfm import cs, utils
from userdiffusion.animations import Animation, PendulumAnimation


class GaussianMixture:
    def __init__(self, cfg, key):
        self.cfg = cfg
        self.key = key

        self.times, self.trajectories = self.generate_data()

    def __len__(self):
        return self.cfg.batch_size * (
            self.cfg.batch_count_train + self.cfg.batch_count_val + self.cfg.batch_count_test
        )

    def __getitem__(self, i):
        return self.trajectories[i]

    def generate_data(self):
        coeffs = [.5, .5]
        means = [-2., 2.]
        stddevs = [-.5, .5]

        x_shape = (len(self), self.cfg.time_step_count, 1)

        self.key, *key_normals = jax.random.split(self.key, num=len(coeffs) + 1)
        normals = []
        for c, mu, std, key_normal in zip(coeffs, means, stddevs, key_normals):
            normals.append(jax.random.normal(key_normal, x_shape) * std + mu)
        normals = jnp.concat(normals, axis=2).reshape(-1, len(normals))

        self.key, key_select = jax.random.split(self.key)
        keys_select = jax.random.split(key_select, num=x_shape[0] * x_shape[1])
        trajectories = jax.vmap(jax.random.choice)(keys_select, normals).reshape(x_shape)

        times = jnp.zeros(x_shape[0])

        return times, trajectories


class ODEDataset(abc.ABC):
    """An (abstract) dataset that generates trajectory chunks from an ODE.

    For a given dynamical system and initial condition distribution,
    each element ds[i] = ((ic,T),z_target) where ic (state_dim,) are the
    initial conditions, T are the evaluation timepoints,
    and z_target (T,state_dim) is the ground truth trajectory chunk.
    To use, one must specify both the dynamics and the initial condition
    distribution for a subclass.

    Class attributes:
      animator: which animator to use for the given dataset
      burnin_time: amount of time to discard for burnin for the given dataset

    Attributes:
      trajectories: state variables z for each trajectory, of shape (N, L, C)
      T_long: full integration timesteps, of shape (L, )
      T: the integration timesteps for chunk if chunk_len is specified, same as
        T_long if chunk_len not specified, otherwise of shape (chunk_len,)
    """

    animator = Animation  # associated object to produce an animation of traj

    def __init__(self, cfg, key, chunk_len=None):
        self.cfg = cfg
        self.key = key

        self.times, self.trajectories = self.generate_trajectory_data()

    def __len__(self):
        return self.cfg.batch_size * (
            self.cfg.batch_count_train + self.cfg.batch_count_val + self.cfg.batch_count_test
        )

    def __getitem__(self, i):
        return self.trajectories[i]

    def integrate(self, z0s, ts, tol=1e-4):
        # Taos: only used for `def animation`
        # Animations use `rtol=1e-4`, whereas `def generate_trajectory_data` uses `rtol=1e-6`.
        dynamics = jit(self.dynamics)
        return odeint(dynamics, z0s, ts, rtol=tol)

    def generate_trajectory_data(self):
        """Returns ts: (N, traj_len) zs: (N, traj_len, z_dim)."""
        n_gen = 0
        device_batch_size = min(self.cfg.device_batch_size, len(self))
        z_batches = []
        mesh = Mesh(mesh_utils.create_device_mesh((device_count(),)), ("data",))
        # odeint_rtol
        integrate = jit(
            vmap(lambda z0, t: odeint(self.dynamics, z0, t, rtol=self.cfg.odeint_rtol), (0, None), 0)
        )
        jintegrate = pjit(integrate, (P("data", None), None), P("data", None, None))
        # batched_dynamics = jit(vmap(self.dynamics, (0, None)))
        num_devices = len(mesh.devices)
        with mesh:
            for _ in tqdm(range(0, len(self), device_batch_size * num_devices)):
                z0s = self.sample_initial_conditions(device_batch_size * num_devices)
                time_step_indices = jnp.arange(0, self.cfg.time_step_count)
                new_zs = jintegrate(z0s, self.cfg.time_step * time_step_indices)
                new_zs = new_zs[:, time_step_indices >= self.cfg.time_step_count_drop_first]
                z_batches.append(new_zs)
                n_gen += device_batch_size
        zs = jnp.concatenate(z_batches, axis=0)[:len(self)]
        time_step_indices = jnp.arange(0, self.cfg.time_step_count)
        times = self.cfg.time_step * time_step_indices
        times = times[time_step_indices >= self.cfg.time_step_count_drop_first]
        return times, zs

    def chunk_training_data(self, zs, chunk_len):
        """Helper function to separate the generated trajectories into chunks."""
        batch_size, traj_len, *_ = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = self.rng.randint(0, n_chunks, (batch_size,))
        chunked_zs = np.stack(np.split(zs, n_chunks, axis=1))
        chosen_zs = chunked_zs[chunk_idx, np.arange(batch_size)]
        return chosen_zs

    def sample_initial_conditions(self, bs):
        """Initial condition distribution."""
        raise NotImplementedError

    def dynamics(self, z, t):
        """Implements the dynamics dz/dt = F(z,t). z is shape (d,)."""
        raise NotImplementedError

    def animate(self, zt=None):
        """Visualize the dynamical system, or given input trajectories.

            Usage
                  from IPython.display import HTML
                  HTML(dataset.animate())
                  or
                  from matplotlib import rc
                  rc('animation',html='jshmtl')
                  dataset.animate()
        Args:
          zt: array of shape (n,d)

        Returns:
          the animation object
        """
        if zt is None:
            zt = np.asarray(
                self.integrate(self.sample_initial_conditions(10)[0], self.times)
            )
        anim = self.animator(zt)
        return anim.animate()


class Lorenz(ODEDataset):
    """
    ODEDataset generated from the Lorenz equations with dynamics.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    where we have chosen rho=28, sigma=10, beta=8/3
    """

    def dynamics(self, z, t):
        x = self.cfg.rescaling * z
        zdot = jnp.array([
                self.cfg.sigma * (x[1] - x[0]),
                x[0] * (self.cfg.rho - x[2]) - x[1],
                x[0] * x[1] - self.cfg.beta * x[2],
        ], dtype=x.dtype)
        return zdot / self.cfg.rescaling

    def sample_initial_conditions(self, bs):
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, (bs, 3))


class FitzHughNagumo(ODEDataset):
    """FitzHugh dynamics from https://arxiv.org/pdf/1803.06277.pdf."""

    def dynamics(self, z, t):
        # Taos: why divide by 5, and multiply by 5 later?
        z = z / 5.0
        a = jnp.array([self.cfg.a1, self.cfg.a2])
        b = jnp.array([self.cfg.b1, self.cfg.b2])
        c = jnp.array([self.cfg.c1, self.cfg.c2])
        k = self.cfg.k
        coupling = self.cfg.coupling12
        n = z.shape[0] // 2
        assert n == 2, "System should have 4 components"
        xs = z[:n]
        ys = z[n:]
        xdot = xs * (a - xs) * (xs - 1) - ys + k * coupling * (xs[::-1] - xs)
        ydot = b * xs - c * ys
        return jnp.concatenate([xdot, ydot]) * 5.0

    def sample_initial_conditions(self, bs):
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, (bs, 4)) * 0.2


def unpack(z):
    D = jnp.shape(z)[-1]  # pylint: disable=invalid-name,unused-variable
    assert D % 2 == 0, "unpack requires even dimension"
    d = D // 2
    q, p_or_v = z[Ellipsis, :d], z[Ellipsis, d:]
    return q, p_or_v


def pack(q, p_or_v):
    return jnp.concatenate([q, p_or_v], axis=-1)


def symplectic_form(z):
    """Equivalent to multiplying z by the matrix J=[[0,I],[-I,0]]."""
    q, p = unpack(z)
    return pack(p, -q)


def hamiltonian_dynamics(hamiltonian, z):
    """Computes hamiltonian dynamics dz/dt=J∇H.

    Args:
      hamiltonian: function state->scalar
      z: state vector (concatenation of q and momentum p)

    Returns:
      dz/dt
    """
    grad_h = grad(hamiltonian)  # ∇H
    gh = grad_h(z)  # ∇H(z)
    return symplectic_form(gh)  # J∇H(z)


class HamiltonianDataset(ODEDataset):
    """ODEDataset with dynamics given by a Hamiltonian system.

    q denotes the generalized coordinates and p denotes the momentum
    Hamiltonian is used along with an associated mass matrix M,
    which is used to convert the momentum p to velocity v
    after generating the trajectories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convert the momentum into velocity
        qs, ps = unpack(self.trajectories)
        Ms = vmap(vmap(self.mass))(qs)  # pylint: disable=invalid-name
        vs = jnp.linalg.solve(Ms, ps)
        self.trajectories = pack(qs, vs)

    def dynamics(self, z, t):
        return hamiltonian_dynamics(self.hamiltonian, z)

    def hamiltonian(self, z):  # pylint: disable=invalid-name
        """The Hamiltonian function, depending on z=pack(q,p)."""
        raise NotImplementedError

    def mass(self, q):  # pylint: disable=invalid-name
        """Mass matrix used for Kinetic energy T=vTM(q)v/2."""
        raise NotImplementedError

    def animate(self, zt=None):  # type: ignore  # jax-ndarray
        if zt is None:
            zt = np.asarray(
                self.integrate(self.sample_initial_conditions(10)[0], self.times)
            )
        # bs, T, 2nd
        if len(zt.shape) == 3:
            j = np.random.randint(zt.shape[0])
            zt = zt[j]
        xt, _ = unpack(zt)
        anim = self.animator(xt)
        return anim.animate()


class SimpleHarmonicOscillator(HamiltonianDataset):
    """A basic simple harmonic oscillator."""

    def hamiltonian(self, z):
        ke = (z[Ellipsis, 1] ** 2).sum() / 2
        pe = (z[Ellipsis, 0] ** 2).sum() / 2
        return ke + pe

    def mass(self, q):
        return jnp.eye(1)

    def sample_initial_conditions(
        self, bs
    ):  # pytype: disable=signature-mismatch  # jax-ndarray
        self.key, key = jax.random.split(self.key)
        return jax.random.normal(key, (bs, 2))


class NPendulum(HamiltonianDataset):
    """An n-link (chaotic) pendulum.

    The generalized coordinates q are the angles (in radians) with respect to
    the vertical down orientation measured counterclockwise. ps are the
    conjugate momenta p = M(q)dq/dt.
    Mass matrix M(q) and Hamiltonian derived in https://arxiv.org/abs/2010.13581,
    page 20.
    """

    animator = PendulumAnimation

    def __init__(self, *args, n=2, dt=0.5, **kwargs):
        """NPendulum constructor.

        Uses additional arguments over base class.

        Args:
          *args: ODEDataset args
          n: number of pendulum links
          dt: timestep size (not for the integrator, but for the final subsampling)
          **kwargs: ODEDataset kwargs
        """
        self.n = n
        super().__init__(*args, dt=dt, **kwargs)

    def mass(self, q):
        # assume all ls are 1 and ms are 1
        ii = jnp.tile(jnp.arange(self.n), (self.n, 1))
        m = jnp.maximum(ii, ii.T)
        return jnp.cos(q[:, None] - q[None, :]) * (self.n - m + 1)

    def hamiltonian(self, z):
        """Energy H(q,p) = pTM(q)^-1p/2 + Sum(yi)."""
        q, p = unpack(z)
        kinetic = (p * jnp.linalg.solve(self.mass(q), p)).sum() / 2
        # assume all ls are 1 and ms are 1
        potential = -jnp.sum(jnp.cumsum(jnp.cos(q)))  # height of bobs
        return kinetic + potential

    def sample_initial_conditions(
        self, bs
    ):  # pytype: disable=signature-mismatch  # jax-ndarray
        self.key, key = jax.random.split(self.key)
        z0 = jax.random.normal(key, (bs, 2 * self.n))
        z0[:, self.n :] *= 0.2
        z0[:, -1] *= 1.5
        return z0


class BatchShape:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.dataset_cfg_name == 'cs.DatasetGaussianMixture':
            self.dimension = 1
        elif cfg.dataset_cfg_name == 'cs.DatasetLorenz':
            self.dimension = 3
        elif cfg.dataset_cfg_name == 'cs.DatasetFitzHughNagumo':
            self.dimension = 4
        elif cfg.dataset_cfg_name == 'cs.DatasetSimpleHarmonicOscillator':
            self.dimension = 2
        elif cfg.dataset_cfg_name == 'cs.DatasetPendulum':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown dataset cfg name: {cfg.dataset_cfg_name}')

        self.trajectories = jnp.array([self[i] for i in range(len(self))])

    def __len__(self):
        return (
            self.cfg.batch_count_train + self.cfg.batch_count_val + self.cfg.batch_count_test
        )

    def __getitem__(self, i):
        return (self.cfg.batch_size, self.cfg.time_step_count - self.cfg.time_step_count_drop_first, self.dimension)


def get_dataset(cfg, key=None, rng_seed=None):
    if key is None:
        key = jax.random.key(rng_seed)
    if isinstance(cfg, cs.DatasetGaussianMixture):
        return GaussianMixture(cfg, key=key)
    elif isinstance(cfg, cs.DatasetLorenz):
        return Lorenz(cfg, key=key)
    elif isinstance(cfg, cs.DatasetFitzHughNagumo):
        return FitzHughNagumo(cfg, key=key)
    elif isinstance(cfg, cs.DatasetSimpleHarmonicOscillator):
        return SimpleHarmonicOscillator(cfg, key=key)
    elif isinstance(cfg, cs.DatasetBatchShape):
        return BatchShape(cfg)
    elif isinstance(cfg, cs.DatasetPendulum):
        return NPendulum(cfg, key=key)
    else:
        raise ValueError(f'Unknown dataset: {cfg}')


def split_dataset(cfg, dataset):
    batch_counts = jnp.array([cfg.batch_count_train, cfg.batch_count_val, cfg.batch_count_test])
    batch_size = 1 if isinstance(cfg, cs.DatasetBatchShape) else cfg.batch_size
    splits = jnp.split(dataset.trajectories, jnp.cumsum(batch_size * batch_counts))
    assert sum(map(len, splits)) == len(dataset)
    return {n: s for n, s in zip(('train', 'val', 'test'), splits)}


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = cs.get_engine()
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        ds = get_dataset(cfg.dataset, rng_seed=cfg.rng_seed)
        print('end')


if __name__ == '__main__':
    main()
