import pytest
from omegaconf import OmegaConf
import jax

from fixtures import init_hydra_cfg, engine
from userfm import cs

from userfm import datasets
from userdiffusion import ode_datasets as datasets_old


@pytest.mark.parametrize('overrides', [
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=Lorenz', 'dataset.batch_size=1', 'dataset.time_step_count=100', 'model=DiffusionVE'],
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=FitzHughNagumo', 'dataset.batch_size=1', 'dataset.time_step_count=667', 'model=DiffusionVE'],
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=SimpleHarmonicOscillator', 'dataset.batch_size=1', 'dataset.time_step_count=60', 'model=DiffusionVE'],
])
def test_datasets_deterministic_with_rng_seed(engine, overrides):
    cfg = init_hydra_cfg('config', overrides)
    with cs.orm.Session(engine) as session:
        cfg = cs.instantiate_and_insert_config(session, OmegaConf.to_container(cfg))
        dss = []
        key = jax.random.key(cfg.rng_seed)
        for _ in range(2):
            dss.append(datasets.get_dataset(cfg.dataset, key=key))
        ds1, ds2 = dss
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            assert (ds1[i][0][0] == ds2[i][0][0]).all()
            assert (ds1[i][0][1] == ds2[i][0][1]).all()
            assert (ds1[i][1] == ds2[i][1]).all()


@pytest.mark.parametrize('overrides', [
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=Lorenz', 'dataset.batch_size=1', 'dataset.time_step_count=100', 'model=DiffusionVE'],
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=FitzHughNagumo', 'dataset.batch_size=1', 'dataset.time_step_count=667', 'model=DiffusionVE'],
    ['+experiment=TrainInitialTimeStepConditioned', 'dataset=SimpleHarmonicOscillator', 'dataset.batch_size=1', 'dataset.time_step_count=60', 'model=DiffusionVE'],
])
def test_datasets_equal(engine, overrides):
    cfg = init_hydra_cfg('config', overrides)
    with cs.orm.Session(engine) as session:
        cfg = cs.instantiate_and_insert_config(session, OmegaConf.to_container(cfg))
        key = jax.random.key(cfg.rng_seed)
        ds = datasets.get_dataset(cfg.dataset, key=key)
        ds_old = datasets_old.get_dataset(cfg.dataset, key=key)
        assert len(ds) == len(ds_old)
        for i in range(len(ds)):
            assert (ds[i][0][0] == ds_old[i][0][0]).all()
            assert (ds[i][0][1] == ds_old[i][0][1]).all()
            assert (ds[i][1] == ds_old[i][1]).all()
