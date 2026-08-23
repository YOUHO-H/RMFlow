import functools
import logging

import einops
import jax
import jax.numpy as jnp
import torch
import lightning.pytorch as pl
import optax

from userdiffusion import samplers
from userfm import cs, sde_diffusion, utils


log = logging.getLogger(__file__)


class JaxLightning(pl.LightningModule):
    def __init__(self, cfg, key, dataloaders, train_data_std, cond_fn, model, predict_sample_event_conditioned=False, predict_event_constraint=None):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.key = key
        self.dataloaders = dataloaders
        self.train_data_std = train_data_std
        self.x_shape = next(iter(dataloaders['train'])).shape
        self.cond_fn = cond_fn
        self.model = model
        self.predict_sample_event_conditioned = predict_sample_event_conditioned
        self.predict_event_constraint = predict_event_constraint

        self.diffusion = sde_diffusion.get_sde_diffusion(self.cfg.model.sde_diffusion)
        self.ema_ts = self.cfg.model.architecture.epochs / self.cfg.model.architecture.ema_folding_count

        self.loss_and_grad = jax.value_and_grad(self.loss, argnums=3, has_aux=True)

    def __hash__(self):
        return hash(id(self))

    def setup(self, stage):
        if stage == 'fit':
            self.key, key_train = jax.random.split(self.key)
            self.params = self.model_init(key_train, self.x_shape, self.cond_fn, self.model)
            self.params_ema = self.params
        elif stage == 'val':
            pass
        elif stage == 'predict':
            pass
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def model_init(self, key, x_shape, cond_fn, model):
        x = jnp.ones(x_shape)
        t = jnp.ones(x_shape[0])
        cond = cond_fn(x)
        params = model.init(key, x=x, t=t, train=False, cond=cond)
        return params

    def configure_optimizers(self):
        self.optimizer = optax.adam(learning_rate=self.cfg.model.architecture.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def train_dataloader(self):
        return self.dataloaders['train']

    def training_step(self, batch, batch_idx):
        cond = self.cond_fn(batch)
        self.key, key_train = jax.random.split(self.key)
        loss, monitors, self.params, self.params_ema, self.opt_state = self.step(
            key_train, batch, cond,
            self.params, self.params_ema,
            self.opt_state,
        )
        # use same key to ensure identical sampling
        loss_ema, monitors_ema = self.loss(key_train, batch, cond, self.params_ema)
        self.optimizers().step()  # increment global step for logging and checkpointing
        outputs = dict(
            loss=loss,
            loss_ema=loss_ema,
            monitors=monitors,
            monitors_ema=monitors_ema,
        )
        return jax.tree.map(lambda x: torch.tensor(x.item()), outputs)

    def val_dataloader(self):
        # from pytorch_lightning.utilities import CombinedLoader
        return self.dataloaders['val']

    def validation_step(self, batch, batch_idx):
        self.key, key_val = jax.random.split(self.key)
        if self.cfg.ckpt_monitor is cs.CkptMonitor.VAL_RELATIVE_ERROR_EMA:
            cond = self.cond_fn(batch)
            samples = self.sample(key_val, 1., cond, batch.shape, params=self.params_ema)
            return dict(
                loss_val=torch.tensor(einops.reduce(utils.relative_error(batch, samples), 'b t ->', 'mean').item()),
            )
        elif self.trainer.sanity_checking:
            return dict(loss_val=torch.tensor(-1.))
        else:
            return dict(loss_val=self.trainer.callback_metrics[str(self.cfg.ckpt_monitor.value)])

    def predict_dataloader(self):
        return self.dataloaders['predict']

    def predict_step(self, batch, batch_idx):
        self.key, key_pred = jax.random.split(self.key)
        x_shape = batch
        cond = None
        if self.predict_sample_event_conditioned:
            def score(x, t):
                if not hasattr(t, 'shape') or not t.shape:
                    t = jnp.ones((x_shape[0], 1, 1)) * t
                return self.score(x, t, cond, self.params_ema)
            event_scores = samplers.event_scores(
                self.diffusion, score, self.predict_event_constraint.constraint, reg=1e-3
            )
            return samplers.sde_sample(
                self.diffusion, event_scores, key_pred, x_shape, nsteps=self.cfg.model.time_step_count_sampling
            )
        else:
            return self.sample(key_pred, 1., cond, x_shape)

    def sample(self, key, tmax, cond, x_shape, params=None, keep_path=False):
        if params is None:
            params = self.params_ema

        def score(x, t):
            if not hasattr(t, 'shape') or not t.shape:
                t = jnp.ones((x_shape[0], 1, 1)) * t
            return self.score(x, t, cond, params)

        return samplers.sde_sample(self.diffusion, score, key, x_shape, nsteps=self.cfg.model.time_step_count_sampling, traj=keep_path)

    @functools.partial(jax.jit, static_argnames=['self', 'train'])
    def score(self, x, t, cond, params, train=False):
        """Score function with appropriate input and output scaling."""
        # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
        sigma, scale = self.diffusion.sigma(t), self.diffusion.scale(t)
        # Taos: Karras et al. $c_in$ and $s(t)$ of EDM.
        input_scale = 1 / jnp.sqrt(sigma**2 + (scale * self.train_data_std) ** 2)
        cond = cond / self.train_data_std if cond is not None else None
        out = self.model.apply(params, x=x * input_scale, t=t.squeeze((1, 2)), train=train, cond=cond)
        # Taos: Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
        return out / jnp.sqrt(sigma**2 + scale**2 * self.train_data_std**2)

    @functools.partial(jax.jit, static_argnames=['self'])
    def loss(self, key, x_data, cond, params):
        """Score matching MSE loss from Yang's Score-SDE paper."""
        # Use lowvar grid time sampling from https://arxiv.org/pdf/2107.00630.pdf
        # Appendix I
        key, key_time = jax.random.split(key)
        u0 = jax.random.uniform(key_time)
        u = jnp.remainder(u0 + jnp.linspace(0, 1, x_data.shape[0]), 1)
        t = u * (self.diffusion.tmax - self.diffusion.tmin) + self.diffusion.tmin
        t = t[:, None, None]

        key, key_noise = jax.random.split(key)
        xt = self.diffusion.noise_input(x_data, t, key_noise)
        target_score = self.diffusion.noise_score(xt, x_data, t)
        # weighting from Yang Song's https://arxiv.org/abs/2011.13456
        # Taos: this appears to be using the weighting from Eqn.(1) used for discrete noise levels.
        weighting = self.diffusion.sigma(t)**2
        error = self.score(xt, t, cond, params, train=True) - target_score
        flow_loss = jnp.mean((self.diffusion.covsqrt.inverse(error)**2) * weighting)
        return flow_loss, {'flow_loss': flow_loss}

    def compute_nll(self, key, tmax, x_data, params=None):
        if params is None:
            params = self.params_ema
        cond = None

        def score(x, t):
            if not hasattr(t, 'shape') or not t.shape:
                t = jnp.ones((x_data.shape[0], 1, 1)) * t
            return self.score(x, t, cond, params)

        return samplers.compute_nll(self.diffusion, score, key, x_data)

    @functools.partial(jax.jit, static_argnames=['self'])
    def step(self, key, batch, cond, params, params_ema, opt_state):
        (loss, monitors), grads = self.loss_and_grad(key, batch, cond, params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / self.ema_ts
        params_ema = jax.tree.map(ema_update, params, params_ema)
        return loss, monitors, params, params_ema, opt_state
