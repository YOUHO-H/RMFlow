import pytest
from omegaconf import OmegaConf

from fixtures import init_hydra_cfg, engine
from userfm import cs

from userfm import main_optuna


def test_apply_trial_params(engine):
    with cs.orm.Session(engine) as session:
        cfg = init_hydra_cfg('config', [
            '+experiment=TrainInitialTimeStepConditioned',
            'dataset=SimpleHarmonicOscillator',
            'model=FlowMatchingOT',
            (
                'model.regularizations=['
                    '{_target_:cs.RegularizationDivergence,coefficient:-1},'
                    '{_target_:cs.RegularizationDerivative,coefficient:-1}'
                ']'
            ),
        ])
        trial_params = dict(
            model=dict(
                regularizations={
                    i: dict(
                        coefficient=i,
                    )
                    for i in range(len(cfg.model.regularizations))
                },
                architecture=dict(
                    learning_rate=1e-4,
                    learning_rate_decay=0.995,
                ),
            ),
        )
        cfg = cs.instantiate_and_insert_config(session, OmegaConf.to_container(cfg))
        main_optuna.apply_trial_params(trial_params, cfg)

        for i in range(len(cfg.model.regularizations)):
            assert cfg.model.regularizations[i].coefficient == i
        assert cfg.model.architecture.learning_rate == 1e-4
        assert cfg.model.architecture.learning_rate_decay == 0.995
