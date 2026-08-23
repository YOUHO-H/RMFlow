from pathlib import Path
import functools
import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import tensorflow as tf
import torch
import torch.utils.data.dataloader
import jax
import jax.numpy as jnp
import numpy as np
import orbax
import lightning.pytorch as pl

from userdiffusion import unet
from userfm import callbacks, cs, datasets, diffusion, flow_matching, event_constraints, utils


log = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames='integrate')
def log_prediction_metric(qs, times, integrate):
    """
    Log geometric mean of rollout relative error computed over a trajectory.
    """
    trajectory = qs
    trajectory_groud_truth = integrate(trajectory[0], times)
    return jnp.log(
        utils.relative_error(trajectory, trajectory_groud_truth)[1:len(times)//2]
    ).mean()


@functools.partial(jax.jit, static_argnames='integrate')
def pmetric(qs, times, integrate):
    """
    Geometric mean of rollout relative error, also taken over the batch.
    """
    log_metric = jax.vmap(
        functools.partial(log_prediction_metric, times=times, integrate=integrate)
    )(qs)
    std_err = jnp.exp(log_metric.std() / jnp.sqrt(log_metric.shape[0]))
    return jnp.exp(log_metric.mean()), std_err


def condition_on_initial_time_steps(z, time_step_count):
    if time_step_count > 0:
        return z[:, :time_step_count]
    return None


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = cs.get_engine()
    cs.create_all(engine)
    with cs.orm.Session(engine, expire_on_commit=False) as db:
        cfg = cs.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        pprint.pp(cfg)
        log.info('Command: %s', ' '.join(sys.argv))
        log.info(f'Outputs will be saved to: {cfg.run_dir}')

        # Hide GPUs from Tensorflow to prevent it from reserving memory,
        # and making it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        log.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        log.info('JAX devices: %r', jax.devices())

        # TODO: split off all the needed keys immediately
        # That way the splits can be reproduced predictably from the initial seed.
        # For example, we need to create the the n-th key split, but do not know n.
        # With how things are written now, we would need to run the entire code body.
        key = jax.random.key(cfg.rng_seed)
        key, key_dataset = jax.random.split(key)
        if isinstance(cfg.dataset, cs.DatasetGaussianMixture) and cfg.use_ckpt_monitor:
            log.warning(
                f'cfg.dataset=GaussianMixture and {cfg.use_ckpt_monitor=}.'
                ' The monitored value may not improve with this dataset.'
                ' Consider setting cfg.use_ckpt_monitor=false.'
            )
        if isinstance(cfg.model, cs.ModelTrained):
            path_train_data_std = cfg.model.config.run_dir/'train_data_std.npy'
            if isinstance(cfg.dataset, cs.DatasetBatchShape):
                if path_train_data_std.exists():
                    train_data_std = np.load(path_train_data_std)
                else:
                    ds = datasets.get_dataset(cfg.model.config.dataset, key=key_dataset)
                    splits = datasets.split_dataset(cfg.model.config.dataset, ds)
                    train_data_std = splits['train'].std()
                    np.save(path_train_data_std, np.array(train_data_std))
                constraint = event_constraints.get_event_constraint(cfg.model.config.dataset)
                log.info('Training set standard deviation: %(data_std).7f', dict(data_std=train_data_std))
                ds_shapes = datasets.get_dataset(cfg.dataset, key=key_dataset)
                splits_shapes = datasets.split_dataset(cfg.dataset, ds_shapes)
                dataloaders = {}
                for n, s in splits_shapes.items():
                    dataloaders[n] = torch.utils.data.dataloader.DataLoader(
                        np.array(s),
                        batch_size=1,
                        collate_fn=lambda x: x[0],
                    )
                dataloaders['predict'] = dataloaders[cfg.predict_dataset_split]
            else:
                if cfg.dataset != cfg.model.config.dataset:
                    raise ValueError('Evaluating a trained model on a dataset different than the one it was trained on is not supported yet.')
                raise NotImplementedError()
        else:
            ds = datasets.get_dataset(cfg.dataset, key=key_dataset)
            constraint = event_constraints.get_event_constraint(cfg.dataset)
            splits = datasets.split_dataset(cfg.dataset, ds)
            train_data_std = splits['train'].std()
            np.save(cfg.run_dir/'train_data_std.npy', np.array(train_data_std))
            log.info('Training set standard deviation: %(data_std).7f', dict(data_std=train_data_std))
            if splits['train'].shape[1] != 60:
                log.warning(
                    'Finzi et al., 2023, trim the trajectories to include only first 60 time steps after the "burn-in" time steps, but these trajectories have %(time_steps)d time steps.'
                    ' Consider setting dataset.time_step_count equal to dataset.time_step_count_drop_first + 60.',
                    dict(time_steps=splits['train'].shape[1])
                )
            dataloaders = {}
            for n, s in splits.items():
                dataloaders[n] = torch.utils.data.dataloader.DataLoader(
                    list(tf.data.Dataset.from_tensor_slices(s).batch(cfg.dataset.batch_size).as_numpy_iterator()),
                    batch_size=1,
                    collate_fn=lambda x: x[0],
                )
            dataloaders['predict'] = dataloaders[cfg.predict_dataset_split]

        dimension = ds_shapes.dimension if isinstance(cfg.dataset, cs.DatasetBatchShape) else splits['train'].shape[2]
        cfg_unet = unet.unet_64_config(
            dimension,
            base_channels=cfg.model.architecture.base_channel_count,
            attention=cfg.model.architecture.attention,
        )
        
        cfg_proj = unet.projection_config(
            seq_len = cfg.dataset.time_step_count - cfg.dataset.time_step_count_drop_first
        )
        model = unet.UNet(cfg_unet)
        proj = unet.Projection(cfg_proj)


        cond_fn = functools.partial(condition_on_initial_time_steps, time_step_count=cfg.dataset.time_step_count_conditioning)
        key, key_trainer = jax.random.split(key)
        if isinstance(cfg.model, cs.ModelDiffusion):
            jax_lightning = diffusion.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model)
        elif isinstance(cfg.model, cs.ModelFlowMatching):
            jax_lightning = flow_matching.JaxLightning(cfg, key_trainer, dataloaders, train_data_std, cond_fn, model, proj, predict_event_constraint=constraint)
        elif isinstance(cfg.model, cs.ModelTrained):
            if isinstance(cfg.model.config.model, cs.ModelDiffusion):
                jax_lightning = diffusion.JaxLightning(
                    cfg.model.config, key_trainer, dataloaders, train_data_std, cond_fn, model,
                    predict_sample_event_conditioned=cfg.model.sample_event_conditioned,
                    predict_event_constraint=constraint,
                )
            elif isinstance(cfg.model.config.model, cs.ModelFlowMatching):
                jax_lightning = flow_matching.JaxLightning(
                    cfg.model.config, key_trainer, dataloaders, train_data_std, cond_fn, model,
                    predict_sample_using_score=cfg.model.sample_using_score,
                    predict_sample_event_conditioned=cfg.model.sample_event_conditioned,
                    predict_event_constraint=constraint,
                )
            else:
                raise ValueError(f'Unknown model: {cfg.model.config.model}')
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            jax_lightning.params = orbax_checkpointer.restore(cfg.model.config.run_dir/cfg.model.ckpt_dir)
            jax_lightning.params_ema = orbax_checkpointer.restore(cfg.model.config.run_dir/f'{cfg.model.ckpt_dir}_ema')
        else:
            raise ValueError(f'Unknown model: {cfg.model}')

        logger = pl.loggers.TensorBoardLogger(cfg.run_dir, name='', version='tb_logs')

        if cfg.use_ckpt_monitor and cfg.dataset.time_step_count_conditioning > 0 and cfg.ckpt_monitor != cs.CkptMonitor.VAL_RELATIVE_ERROR_EMA:
            log.warning(f'{cfg.dataset.time_step_count_conditioning=} > 0, but {cfg.ckpt_monitor=}, not {cs.CkptMonitor.VAL_RELATIVE_ERROR_EMA}.')
        pl_trainer = pl.Trainer(
            max_epochs=cfg.model.architecture.epochs,
            logger=logger,
            precision=32,
            callbacks=[
                callbacks.ModelCheckpoint(
                    dirpath=cfg.run_dir,
                    filename='{epoch}',
                    save_top_k=1,
                    monitor=cfg.ckpt_monitor if cfg.use_ckpt_monitor else None,
                    save_last='link',
                    save_on_train_epoch_end=False,
                    enable_version_counter=False,
                ),
                callbacks.LogStats(),
            ],
            log_every_n_steps=1,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
            deterministic=True,
        )

        if cfg.fit:
            pl_trainer.fit(jax_lightning)
        if cfg.predict:
            pred = pl_trainer.predict(jax_lightning)
            pred = np.array(jnp.concat(pred))
            np.save(cfg.run_dir/cfg.prediction_filename, pred)
        if not (cfg.fit or cfg.predict):
            raise ValueError(f'Please set either fit=true (currently {cfg.fit=}) or predict=true (currently {cfg.predict=}).')


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = cs.get_engine()
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
