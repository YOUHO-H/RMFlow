import functools
import logging

import einops
import jax
import jax.numpy as jnp
import torch
import lightning.pytorch as pl
import optax

from userdiffusion import samplers
from userfm import cs, optimal_transport, sde_diffusion, utils


log = logging.getLogger(__file__)


def heun_sample(key, tmax, velocity, x_shape, nsteps=1000, keep_path=False):
    x_noise = jax.random.normal(key, x_shape)
    timesteps = (.5 + jnp.arange(nsteps)) / nsteps
    x0, xs = samplers.heun_integrate(velocity, x_noise, timesteps)
    return xs if keep_path else x0


def heun_sample_diffusion(key, diffusion, tmax, velocity, x_shape, nsteps=1000, keep_path=False):
    x_noise = jax.random.normal(key, x_shape) * diffusion.sigma(tmax)
    timesteps = (.5 + jnp.arange(nsteps)) / nsteps
    x0, xs = samplers.heun_integrate(velocity, x_noise, timesteps)
    return xs if keep_path else x0


def inner_prod(a, b):
    return (a * b).reshape(a.shape[0], -1).sum(-1, keepdims=True)



class JaxLightning(pl.LightningModule):
    def __init__(self, cfg, key, dataloaders, train_data_std, cond_fn, model, proj, predict_sample_using_score=False, predict_sample_event_conditioned=False, predict_event_constraint=None):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.key = key
        self.dataloaders = dataloaders
        self.train_data_std = train_data_std
        self.cond_fn = cond_fn
        self.model = model
        self.proj = proj
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            self.diffusion = sde_diffusion.get_sde_diffusion(self.cfg.model.conditional_flow.sde_diffusion)
        self.predict_sample_using_score = predict_sample_using_score
        self.predict_sample_event_conditioned = predict_sample_event_conditioned
        self.predict_event_constraint = predict_event_constraint

        self.ema_ts = self.cfg.model.architecture.epochs / self.cfg.model.architecture.ema_folding_count

        self.loss_and_grad = jax.value_and_grad(self.loss, argnums=3, has_aux=True)

    def __hash__(self):
        return hash(id(self))

    def setup(self, stage):
        if stage == 'fit':
            self.key, key_train = jax.random.split(self.key)
            self.params = self.model_init(key_train, 
                                          next(iter(self.dataloaders['train'])).shape, 
                                          self.cond_fn, self.model)
            self.params_proj = self.proj_init(key_train,
                                             (self.cfg.dataset.batch_size,
                                              self.cfg.dataset.time_step_count_conditioning,
                                              4),
                                             self.proj)
            self.opt_params = {
                'model': self.params,
                'proj': self.params_proj,
            }
            self.params_ema = self.opt_params
        elif stage == 'val':
            pass
        elif stage == 'predict':
            pass
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def model_init(self, key, x_shape, cond_fn, model):
        x = jnp.ones(x_shape)
        t = jnp.ones(x_shape[0])
        r = jnp.ones(x_shape[0])  # initial time step conditioning
        cond = cond_fn(x)
        params = model.init(key, x=x, t=t, r=r, train=False, cond=cond)
        return params

    def proj_init(self, key, x_shape, proj):
        x = jnp.ones(x_shape)
        params = proj.init(key, x=x)
        return params

    def configure_optimizers(self):
        learning_rate_scheduler = optax.exponential_decay(
            init_value=self.cfg.model.architecture.learning_rate,  # Initial LR
            transition_steps=512,  # Number of steps before decay
            decay_rate=self.cfg.model.architecture.learning_rate_decay,  # Decay factor
            staircase=True  # Whether to use staircase decay
        )
        self.optimizer = optax.adam(learning_rate=learning_rate_scheduler)
        self.opt_state = self.optimizer.init(self.opt_params)

    def train_dataloader(self):
        return self.dataloaders['train']

    def training_step(self, batch, batch_idx):
        cond = self.cond_fn(batch)
        self.key, key_train = jax.random.split(self.key)
        loss, monitors, self.opt_params, self.params_ema, self.opt_state = self.step(
            key_train, batch, cond,
            self.opt_params, self.params_ema,
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
            samples = self.sample(key_val, 1., cond, batch, params=self.params_ema)
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
            return self.sample(key_pred, 1., cond, batch, use_score=self.predict_sample_using_score)

    def sample(self, key, tmax, cond, batch, params=None, keep_path=False, use_score=False):
        x_data = batch
        x_shape = x_data.shape
        if params is None:
            params = self.params_ema
        if self.predict_event_constraint:
            event = self.predict_event_constraint.statistic(x_data)
            event = jnp.broadcast_to(event[:, None, None], (event.shape[0], 3, 1))
        else:
            event = jnp.zeros((x_data.shape[0], 3, 1))
        src = jnp.concatenate([cond, event], axis=-1) if cond is not None else event
        src_cond = self.src_project(src, params['proj'], train=True)
        noise = jax.random.normal(key, x_shape) * 0.1
        noise = src_cond + noise
        mu_pred = noise + self.velocity(noise, jnp.zeros((x_shape[0], 1, 1)), 
                                        jnp.ones((x_shape[0], 1, 1)), 
                                        cond, params['model'])
        noise_eps = jax.random.normal(key, x_shape) * 1e-4
        return mu_pred + noise_eps

        # import pdb; pdb.set_trace()
        # def velocity(x, t):
        #     if not hasattr(t, 'shape') or not t.shape:
        #         t = jnp.ones((x_shape[0], 1, 1)) * t
        #     return self.velocity(x, t, jnp.zeros_like(t), cond, params['model'])

        # if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
        #     if use_score:
        #         def score(x, t):
        #             if not hasattr(t, 'shape') or not t.shape:
        #                 t = jnp.ones((x_shape[0], 1, 1)) * t
        #             return self.score(x, t, cond, params)

        #         return samplers.sde_sample(self.diffusion, score, key, x_shape, nsteps=self.cfg.model.time_step_count_sampling, traj=keep_path)
        #     else:
        #         return heun_sample_diffusion(key, self.diffusion, tmax, velocity, x_shape=x_shape, nsteps=self.cfg.model.time_step_count_sampling, keep_path=keep_path)
        # else:
        #     if use_score:
        #         raise ValueError(
        #             f'Writing the score function in terms of the flow matching vector field is only supported when cfg.model.conditional_flow is {cs.ConditionalSDE.__name__}, not {type(self.cfg.model.conditional_flow)}.'
        #             'Please set use_score=False.'
        #         )
        #     return heun_sample(key, tmax, velocity, x_shape=x_shape, nsteps=self.cfg.model.time_step_count_sampling, keep_path=keep_path)

    @functools.partial(jax.jit, static_argnames=['self'])
    def score(self, x, t, cond, params):
        if not isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            raise ValueError(
                f'Writing the score function in terms of the flow matching vector field is only supported when cfg.model.conditional_flow is {cs.ConditionalSDE.__name__}, not {self.cfg.model.conditional_flow.__class__.__name__}.'
            )
        if not isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
            raise ValueError(
                f'Writing the score function in terms of the flow matching vector field is only implemented for when cfg.model.conditional_flow.sde_diffusion is {cs.SDEVarianceExploding.__name__}, not {self.cfg.model.conditional_flow.sde_diffusion.__class__.__name__}.'
            )
        # sde_sample integrates from 1 to 0, so
        # 1. drop the negative sign
        # 2. pass the reversed time to the flow matching model
        # Lemma 1 of the original Lipman et al. paper on flow matching.
        return 2 / self.diffusion.g2(t) * self.velocity(x, 1 - t, jnp.zeros_like(t), cond, params['model'])

    @functools.partial(jax.jit, static_argnames=['self', 'train'])
    def velocity(self, x, t, r, cond, params, train=False):
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if self.cfg.model.conditional_flow.finzi_karras_weighting:
                # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
                sigma = self.diffusion.sigma(1 - t)
                # Taos: Karras et al. $c_in$ and $s(t)$ of EDM.
                input_scale = 1 / jnp.sqrt(sigma**2 + self.train_data_std**2)
                cond = cond / self.train_data_std if cond is not None else None
                out = self.model.apply(params, x=x * input_scale, t=t.squeeze((1, 2)), r=r.squeeze((1, 2)), train=train, cond=cond)
                # Taos: Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
                return out / jnp.sqrt(sigma**2 + self.train_data_std**2)
            else:
                return self.model.apply(params, x=x, t=t.squeeze((1, 2)), r=r.squeeze((1, 2)), train=train, cond=cond)
        else:
            return self.model.apply(params, x=x, t=t.squeeze((1, 2)), r=r.squeeze((1, 2)), train=train, cond=cond)

    @functools.partial(jax.jit, static_argnames=['self', 'train'])
    def src_project(self, x, params, train=False):
        return self.proj.apply(params, x=x, train=train)

    @functools.partial(jax.jit, static_argnames=['self'])
    def conditional_ot(self, t, x_noise, x_data):
        mean_scale, std = t, 1 - t
        xt = std * x_noise + mean_scale * x_data
        velocity_target = x_data - x_noise
        eps = 1e-6
        return dict(
            xt=xt,
            mean_scale=mean_scale, std=std,
            velocity_target=velocity_target,
            dx_velocity_target=-1 / (std + eps),
            dx_log_pt=-(xt - mean_scale * x_data) / (std + eps)**2,
        )

    @functools.partial(jax.jit, static_argnames=['self'])
    def minimatch_ot_conditional_ot(self, key, t, x_noise, x_data):
        x_noise, x_data = optimal_transport.OTPlanSamplerJax.sample_plan(
            key,
            x_noise, x_data,
            reg=self.cfg.model.conditional_flow.sinkhorn_regularization,
            replace=self.cfg.model.conditional_flow.sample_with_replacement,
        )
        return self.conditional_ot(t, x_noise, x_data)

    @functools.partial(jax.jit, static_argnames=['self'])
    def variance_exploding_conditional(self, t, x_noise, x_data):
        mean_scale, std = jnp.ones_like(t), self.diffusion.sigma(1 - t)
        eps = 1e-6
        # add eps here to make equal to divisor in velocity_target
        xt = x_data + (std + eps) * x_noise
        dt_std = self.diffusion.dsigma(1 - t)
        dx_velocity_target = -dt_std / (std + eps)
        velocity_target = dx_velocity_target * (xt - x_data)
        return dict(
            xt=xt,
            mean_scale=1., std=std,
            velocity_target=velocity_target,
            dx_velocity_target=dx_velocity_target,
            dx_log_pt=-(xt - mean_scale * x_data) / (std + eps)**2,
            dt_std=dt_std,
        )

    @functools.partial(jax.jit, static_argnames=['self'])
    def loss(self, key, x_data, cond, params):
        
        if self.cfg.model.time_samples_uniformly_spaced:
            key, key_time = jax.random.split(key)
            u0 = jax.random.uniform(key_time)
            u = jnp.remainder(u0 + jnp.linspace(0, 1, x_data.shape[0]), 1)
            t = u * (self.diffusion.tmax - self.diffusion.tmin) + self.diffusion.tmin
            t = t[:, None, None]
        else:
            key, key_time = jax.random.split(key)
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))
            r = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))
            t, r = jnp.minimum(t, r), jnp.maximum(t, r)


        key, key_noise = jax.random.split(key)
        x_noise = jax.random.normal(key_noise, x_data.shape)
        if self.predict_event_constraint:
            event = self.predict_event_constraint.statistic(x_data)
            event = jnp.broadcast_to(event[:, None, None], (event.shape[0], 3, 1))
        else:
            event = jnp.zeros((x_data.shape[0], 3, 1))

        src = jnp.concatenate([cond, event], axis=-1) if cond is not None else event
        src_cond = self.src_project(src, params['proj'], train=True)
        x_noise = src_cond + x_noise * 0.1

        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalOT):
            context = self.conditional_ot(t, x_noise, x_data)
            weighting = 1.
        elif isinstance(self.cfg.model.conditional_flow, cs.MinibatchOTConditionalOT):
            key, key_plan = jax.random.split(key)
            context = self.minimatch_ot_conditional_ot(key_plan, t, x_noise, x_data)
            weighting = 1.
        elif isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
                context = self.variance_exploding_conditional(t, x_noise, x_data)
                weighting = 1 / context['dt_std']**2
            else:
                raise ValueError(f'Unknown SDE diffusion: {self.cfg.model.conditional_flow.sde_diffusion}')
        else:
            raise ValueError(f'Unknown conditional flow: {self.cfg.model.conditional_flow}')

        regularization_values = {}

        def u_fn(x, t, r):
            return self.velocity(x, t, r-t, cond, params['model'], train=True)
        # velocity_pred = self.velocity(context['xt'], t, r, cond, params['model'], train=True)
        dt_dt = jnp.ones_like(t)
        dr_dt = jnp.zeros_like(t)
        u, du_dt = jax.jvp(u_fn, 
                           (context['xt'], t, r), 
                           (context['velocity_target'], dt_dt, dr_dt))

        u_tgt = context['velocity_target'] + (r-t) * du_dt
        u_tgt = jax.lax.stop_gradient(u_tgt)

        # flow_loss = ((velocity_pred - context['velocity_target'])**2 * weighting).mean()
        flow_loss = (u - u_tgt)**2 * weighting
        flow_loss = jnp.sum(flow_loss, axis=(1, 2))  # sum over pixels
        
        adp_wt = (flow_loss + 1e-3) ** 0.5
        flow_loss = flow_loss / jax.lax.stop_gradient(adp_wt)
        flow_loss = flow_loss.mean()  # mean over batch

        mu_pred = x_noise + u_fn(x_noise, jnp.zeros_like(t), jnp.ones_like(t))
        elbo_loss = ((mu_pred - x_data)**2).mean()

        flow_loss += elbo_loss

        regularization = sum((reg.coefficient * v for reg, v in regularization_values.items()))

        monitors = {'flow_loss': flow_loss, **{reg.__class__.__name__: v for reg, v in regularization_values.items()}}

        return flow_loss + regularization, monitors

    def compute_nll(self, key, tmax, x_data, params=None, use_score=False):
        if params is None:
            params = self.params_ema

        if use_score:
            if (
                isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE)
                and not isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding)
            ):
                raise ValueError(
                    f'Writing the score function in terms of the flow matching vector field is only implemented for when cfg.model.conditional_flow.sde_diffusion is {cs.SDEVarianceExploding.__name__}, not {self.cfg.model.conditional_flow.sde_diffusion.__class__.__name__}.'
                )

        cond = None
        def probability_flow(x, t):
            if not hasattr(t, 'shape') or not t.shape:
                t = jnp.ones((x_data.shape[0], 1, 1)) * t
            if False:
            # if use_score:
                # for VE path, by the definition of self.score and
                # self.diffusion.dynamics, the probability_flow is the same
                # for use_score and not.
                def score(x, t):
                    if not hasattr(t, 'shape') or not t.shape:
                        t = jnp.ones((x_data.shape[0], 1, 1)) * t
                    return self.score(x, t, cond, params)
                return self.diffusion.dynamics(score, x, t[0, 0, 0])
            else:
                # negative sign due to change of variables t -> 1 - t
                return -self.velocity(x, 1 - t, jnp.zeros_like(t), cond, params)

        key, key_hutchinson = jax.random.split(key)
        noise_hutchinson = jax.random.normal(key_hutchinson, x_data.shape)

        @jax.jit
        def value_and_divergence(y, t):
            x = y[0]
            pred, pred_jvp = jax.jvp(
                lambda xt: probability_flow(xt, t),
                [x], [noise_hutchinson],
            )
            divergence_pred = inner_prod(noise_hutchinson, pred_jvp)
            return pred, divergence_pred

        evaluations_at_t = jax.experimental.ode.odeint(
            func=value_and_divergence,
            y0=[x_data, jnp.zeros(x_data.shape[0])],
            t=jnp.array([0, 1.]),
            rtol=1e-3,
        )
        x_noise, neg_logdet__dx_data__dx_noise = [y[-1] for y in evaluations_at_t]

        dim = (x_noise.shape[1] * x_noise.shape[2])
        if isinstance(self.cfg.model.conditional_flow, cs.ConditionalSDE):
            if isinstance(self.cfg.model.conditional_flow.sde_diffusion, cs.SDEVarianceExploding):
                std_max = self.diffusion.sigma(self.diffusion.cfg.time_max)
                log__p__x_noise = (
                    # liklihood of N(x_noise; 0, (std_max**2)I)
                    - .5 * dim * jnp.log(2 * jnp.pi)
                    - jnp.log(std_max)
                    - .5 * einops.reduce(x_noise**2, 'batch time dim -> batch', 'sum') / std_max**2
                    # integral
                    + neg_logdet__dx_data__dx_noise
                )
            else:
                raise ValueError(f'Unknown SDE diffusion: {self.cfg.model.conditional_flow.sde_diffusion}')
        else:
            log__p__x_noise = (
                # liklihood of N(x_noise; 0, I)
                -.5 * dim * jnp.log(2 * jnp.pi)
                -.5 * einops.reduce(x_noise**2, 'batch time dim -> batch', 'sum')
                # integral
                + neg_logdet__dx_data__dx_noise
            )

        return x_noise, -log__p__x_noise, -log__p__x_noise / dim

    @functools.partial(jax.jit, static_argnames=['self'])
    def step(self, key, batch, cond, params, params_ema, opt_state):
        (loss, monitors), grads = self.loss_and_grad(key, batch, cond, params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / self.ema_ts
        params_ema = jax.tree.map(ema_update, params, params_ema)
        return loss, monitors, params, params_ema, opt_state
