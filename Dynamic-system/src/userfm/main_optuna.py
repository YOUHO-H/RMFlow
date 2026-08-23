import functools
import inspect
import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import tensorflow as tf
import torch
import torch.utils.data.dataloader
import jax
import lightning.pytorch as pl
import optuna
import optuna_integration

from userdiffusion import unet
from userfm import callbacks, cs, datasets, diffusion, flow_matching, utils


log = logging.getLogger(__name__)


def condition_on_initial_time_steps(z, time_step_count):
    if time_step_count > 0:
        return z[:, :time_step_count]
    return None


def apply_trial_params(trial_params, cfg):
    for k, v in trial_params.items():
        if isinstance(v, dict):
            cfg_nested = getattr(cfg, k)
            if isinstance(cfg_nested, list):
                for i, vv in v.items():
                    apply_trial_params(vv, cfg_nested[i])
            else:
                apply_trial_params(v, cfg_nested)
        else:
            setattr(cfg, k, v)


def get_trial_params(cfg, trial):
    return dict(
        model=dict(
            regularizations={
                i: dict(
                    coefficient=trial.suggest_float(f'model.regularizations[{i}].coefficient', 1e-5, 10, log=True),
                )
                for i in range(len(cfg.model.regularizations))
            },
            architecture=dict(
              learning_rate=trial.suggest_float('model.architecture.learning_rate', 5e-5, 1e-3),
              # learning_rate_decay=trial.suggest_float('model.architecture.learning_rate_decay', .985, .999),
            ),
        ),
    )


def objective_trainer(cfg, key, dataloaders, train_data_std, model, trial):
    trial_params = get_trial_params(cfg, trial)

    apply_trial_params(trial_params, cfg)

    cond_fn = functools.partial(condition_on_initial_time_steps, time_step_count=cfg.dataset.time_step_count_conditioning)
    key, key_trainer = jax.random.split(key)
    if isinstance(cfg.model, cs.ModelDiffusion):
        jax_lightning = diffusion.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
    elif isinstance(cfg.model, cs.ModelFlowMatching):
        jax_lightning = flow_matching.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
    else:
        raise ValueError(f'Unknown model: {cfg.model}')

    logger = pl.loggers.TensorBoardLogger(cfg.run_dir, name='tb_logs')
    pl_trainer = pl.Trainer(
        max_epochs=cfg.model.architecture.epochs,
        logger=logger,
        precision=32,
        callbacks=[
            callbacks.LogStats(),
            optuna_integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, cfg.ckpt_monitor.value),
        ],
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        enable_checkpointing=False,
        deterministic=True,
    )

    pl_trainer.fit(jax_lightning)

    return pl_trainer.callback_metrics[cfg.ckpt_monitor.value].item()


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = cs.get_engine(name='optuna_runs')
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        pprint.pp(cfg)
        log.info('Command: %s', ' '.join(sys.argv))
        log.info(f'Outputs will be saved to: {cfg.run_dir}')

        # Hide GPUs from Tensorflow to prevent it from reserving memory,
        # and making it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        log.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        log.info('JAX devices: %r', jax.devices())

        key = jax.random.key(cfg.rng_seed)
        key, key_dataset = jax.random.split(key)
        if isinstance(cfg.dataset, cs.DatasetGaussianMixture) and cfg.use_ckpt_monitor:
            log.warn(
                f'cfg.dataset=GaussianMixture and {cfg.use_ckpt_monitor=}.'
                'The monitored value may not improve with this dataset.'
                'Consider setting cfg.use_ckpt_monitor=false.'
            )
        ds = datasets.get_dataset(cfg.dataset, key=key_dataset)
        splits = datasets.split_dataset(cfg.dataset, ds)
        if splits['train'].shape[1] != 60:
            log.warn(
                'Finzi et al., 2023, trim the trajectories to include only first 60 time steps after the "burn-in" time steps, but these trajectories have %(time_steps)d time steps.'
                'Consider setting dataset.time_step_count equal to dataset.time_step_count_drop_first + 60.',
                dict(time_steps=splits['train'].shape[1])
            )
        dataloaders = {}
        for n, s in splits.items():
            dataloaders[n] = torch.utils.data.dataloader.DataLoader(
                list(tf.data.Dataset.from_tensor_slices(s).batch(cfg.dataset.batch_size).as_numpy_iterator()),
                batch_size=1,
                collate_fn=lambda x: x[0],
            )

        cfg_unet = unet.unet_64_config(
            splits['train'].shape[2],
            base_channels=cfg.model.architecture.base_channel_count,
            attention=cfg.model.architecture.attention,
        )
        model = unet.UNet(cfg_unet)

        train_data_std = splits['train'].std()
        log.info('Training set standard deviation: %(data_std).7f', dict(data_std=train_data_std))

        objective = functools.partial(objective_trainer, cfg, key, dataloaders, train_data_std, model)
        study = optuna.create_study(
            storage='sqlite:///optuna.sqlite',
            direction='minimize',
            study_name=f'metric={cfg.ckpt_monitor},{hydra.core.hydra_config.HydraConfig.get().job.override_dirname}',
            # sampler=...,
            # pruner=optuna.pruners.SuccessiveHalvingPruner(),
            pruner=optuna.pruners.ThresholdPruner(lower=-1),  # set lower to impossible value to prune only on NaN
        )
        study.set_metric_names([cfg.ckpt_monitor.value])
        study.set_user_attr('trial_params', inspect.getsource(get_trial_params))
        study.set_user_attr('run_dir', str(cfg.run_dir))
        study.optimize(objective, n_trials=50)


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = cs.get_engine(name='optuna_runs')
        cs.create_all(engine)
        with cs.orm.Session(engine, expire_on_commit=False) as db:
            cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit and '-c' not in sys.argv:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


if __name__ == '__main__':
    last_override, run_dir = get_run_dir()
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)
    main()
