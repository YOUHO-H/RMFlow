import functools
import string
import random
import typing
import dataclasses
from dataclasses import dataclass, field
import enum
from pathlib import Path
import omegaconf
from omegaconf import OmegaConf

import hydra
import sqlalchemy as sa
from sqlalchemy import orm


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
DIR_SRC = DIR_ROOT/'userdiffusion'/'src'


mapper_registry = orm.registry()


MODULE_NAME = Path(__file__).stem


ColumnRequired = functools.partial(sa.Column, nullable=False)


class CfgWithTable:
    __sa_dataclass_metadata_key__ = 'sa'

    def __init_subclass__(cls):
        cls.__hash__ = CfgWithTable.__hash__
        return mapper_registry.mapped(dataclass(cls))

    def __hash__(self):
        return hash(self.id)


class Dataset(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=ColumnRequired(sa.String(20)),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    device_batch_size: int = field(default=10)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))

    batch_count_train: int = field(default=8, metadata=dict(sa=ColumnRequired(sa.Integer)))
    batch_count_val: int = field(default=1, metadata=dict(sa=ColumnRequired(sa.Integer)))
    batch_count_test: int = field(default=1, metadata=dict(sa=ColumnRequired(sa.Integer)))
    batch_size: int = field(default=100, metadata=dict(sa=ColumnRequired(sa.Integer)))

    # Trajectory evaluation points and length
    time_step: float  = field(default=.1, metadata=dict(sa=ColumnRequired(sa.Double)))
    time_step_count: int = field(default=100, metadata=dict(sa=ColumnRequired(sa.Integer)))
    time_step_count_drop_first: int = field(default=30, metadata=dict(sa=ColumnRequired(sa.Integer)))
    time_step_count_conditioning: int = field(default=3, metadata=dict(sa=ColumnRequired(sa.Integer)))

    odeint_rtol: float = field(default=1e-6, metadata=dict(sa=ColumnRequired(sa.Double)))


class DatasetBatchShape(Dataset):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Dataset.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    dataset_cfg_name: str = field(default=omegaconf.MISSING, metadata=dict(sa=ColumnRequired(sa.String(len('cs.DatasetSimpleHarmonicOscillator')))))

    def __post_init__(self):
        if self.dataset_cfg_name == omegaconf.MISSING:
            raise ValueError('Please set a dataset cfg name with dataset.dataset_cfg_name=cs.<dataset_name>')


class DatasetGaussianMixture(Dataset):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Dataset.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))


class DatasetLorenz(Dataset):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Dataset.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    # Lorenz system parameters
    rho: float = field(default=28., metadata=dict(sa=ColumnRequired(sa.Double)))
    sigma: float = field(default=10., metadata=dict(sa=ColumnRequired(sa.Double)))
    beta: float = field(default=8/3, metadata=dict(sa=ColumnRequired(sa.Double)))
    # Scale strange attractor
    rescaling: float = field(default=20., metadata=dict(sa=ColumnRequired(sa.Double)))


class DatasetFitzHughNagumo(Dataset):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Dataset.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    a1: float = field(default=-.025794, metadata=dict(sa=ColumnRequired(sa.Double)))
    a2: float = field(default=-.025794, metadata=dict(sa=ColumnRequired(sa.Double)))
    b1: float = field(default=.0065, metadata=dict(sa=ColumnRequired(sa.Double)))
    b2: float = field(default=.0135, metadata=dict(sa=ColumnRequired(sa.Double)))
    c1: float = field(default=.02, metadata=dict(sa=ColumnRequired(sa.Double)))
    c2: float = field(default=.02, metadata=dict(sa=ColumnRequired(sa.Double)))
    k: float = field(default=.128, metadata=dict(sa=ColumnRequired(sa.Double)))
    coupling12: float = field(default=1., metadata=dict(sa=ColumnRequired(sa.Double)))
    coupling21: float = field(default=1., metadata=dict(sa=ColumnRequired(sa.Double)))


class DatasetSimpleHarmonicOscillator(Dataset):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Dataset.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))


class ModelArchitecture(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.String(20), nullable=False),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))

    epochs: int = field(default=10_000, metadata=dict(sa=ColumnRequired(sa.Integer)))
    learning_rate: float = field(default=1e-4, metadata=dict(sa=ColumnRequired(sa.Double)))
    learning_rate_decay: float = field(default=0.995, metadata=dict(sa=ColumnRequired(sa.Double)))
    ema_folding_count: int = field(default=5, metadata=dict(sa=ColumnRequired(sa.Integer)))


class UNet(ModelArchitecture):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{ModelArchitecture.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    base_channel_count: int = field(default=32, metadata=dict(sa=ColumnRequired(sa.Integer)))
    attention: bool = field(default=False, metadata=dict(sa=ColumnRequired(sa.Boolean)))


class SDEDiffusion(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.String(20), nullable=False),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))

    time_min: float = field(default=1e-3, metadata=dict(sa=ColumnRequired(sa.Double)))
    time_max: float = field(default=1., metadata=dict(sa=ColumnRequired(sa.Double)))


class SDEVarianceExploding(SDEDiffusion):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{SDEDiffusion.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    sigma_min: float = field(default=1e-3, metadata=dict(sa=ColumnRequired(sa.Double)))
    sigma_max: float = field(default=300., metadata=dict(sa=ColumnRequired(sa.Double)))


class Model(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.String(20), nullable=False),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))

    architecture_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(ModelArchitecture.__name__, sa.ForeignKey(f'{ModelArchitecture.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    architecture: ModelArchitecture = field(default_factory=UNet, metadata=dict(sa=orm.relationship(ModelArchitecture.__name__, foreign_keys=[architecture_id.metadata['sa']])))


class ModelDiffusion(Model):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)
    defaults: typing.List[typing.Any] = field(repr=False, default_factory=lambda: [
        dict(sde_diffusion=omegaconf.MISSING),
        '_self_'
    ])

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Model.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    sde_diffusion_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(SDEDiffusion.__name__, sa.ForeignKey(f'{SDEDiffusion.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    sde_diffusion: SDEDiffusion = field(default=omegaconf.MISSING, metadata=dict(sa=orm.relationship(SDEDiffusion.__name__, foreign_keys=[sde_diffusion_id.metadata['sa']])))

    time_step_count_sampling: int = field(default=1000, metadata=dict(sa=ColumnRequired(sa.Integer)))


class ConditionalFlow(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.String(20), nullable=False),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))


class ConditionalOT(ConditionalFlow):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{ConditionalFlow.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))


class OTSolver(str, enum.Enum):
    EXACT = 'exact'  # Earth movers distance
    SINKHORN = 'sinkhorn'
    UNBALANCED_SINKHORN_KNOPP = 'unbalanced'
    PARTIAL = 'partial'


class MinibatchOTConditionalOT(ConditionalFlow):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{ConditionalFlow.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    ot_solver: OTSolver = field(default=OTSolver.EXACT, metadata=dict(sa=ColumnRequired(sa.Enum(OTSolver))))
    sinkhorn_regularization: float = field(default=.05, metadata=dict(sa=ColumnRequired(sa.Double)))
    unbalanced_sinkhorn_knopp_regularization: float = field(default=1., metadata=dict(sa=ColumnRequired(sa.Double)))
    normalize_cost: bool = field(default=False, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    sample_with_replacement: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))


class ConditionalSDE(ConditionalFlow):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)
    defaults: typing.List[typing.Any] = field(repr=False, default_factory=lambda: [
        dict(sde_diffusion=omegaconf.MISSING),
        '_self_'
    ])

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{ConditionalFlow.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    sde_diffusion_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(SDEDiffusion.__name__, sa.ForeignKey(f'{SDEDiffusion.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    sde_diffusion: SDEDiffusion = field(default=omegaconf.MISSING, metadata=dict(sa=orm.relationship(SDEDiffusion.__name__, foreign_keys=[sde_diffusion_id.metadata['sa']])))

    finzi_karras_weighting: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))


class Regularization(CfgWithTable):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    sa_inheritance: str = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.String(20), nullable=False),
        omegaconf_ignore=True,
    ))
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))

    coefficient: float = field(default=0, metadata=dict(sa=ColumnRequired(sa.Double)))


class RegularizationDerivative(Regularization):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Regularization.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))


class RegularizationDivergence(Regularization):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Regularization.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    use_hutchinson_trace_for_divergence_target: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))


class ModelFlowMatching(Model):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)
    defaults: typing.List[typing.Any] = field(repr=False, default_factory=lambda: [
        dict(conditional_flow=omegaconf.MISSING),
        '_self_'
    ])

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Model.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    conditional_flow_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(ConditionalFlow.__name__, sa.ForeignKey(f'{ConditionalFlow.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    conditional_flow: ConditionalFlow = field(default=omegaconf.MISSING, metadata=dict(sa=orm.relationship(ConditionalFlow.__name__, foreign_keys=[conditional_flow_id.metadata['sa']])))

    time_samples_uniformly_spaced: bool = field(default=False, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    time_step_count_sampling: int = field(default=1000, metadata=dict(sa=ColumnRequired(sa.Integer)))

    regularizations: typing.List[typing.Any] = field(default_factory=list, metadata=dict(
        sa=orm.relationship(Regularization.__name__, secondary=lambda: table_m2m_ModelFlowMatching__Regularization)
    ))

    # just override _target_
    # @staticmethod
    # def transform_regularizations(session, regularizations):
    #     instantiated = []
    #     for reg in regularizations:
    #         if reg['_target_'] == 'cs.DerivativeRegularization':
    #             instantiated.append(DerivativeRegularization(**reg))
    #         elif reg['_target_'] == 'cs.DerivativeRegularization':
    #             instantiated.append(DivergenceRegularization(**reg))
    #         else:
    #             raise ValueError(f"Unknown regularization: {reg['target']}")
    #     return instantiated


table_m2m_ModelFlowMatching__Regularization = sa.Table(
    f'{ModelFlowMatching.__name__}__{Regularization.__name__}',
    mapper_registry.metadata,
    sa.Column(ModelFlowMatching.__name__, sa.ForeignKey(f'{ModelFlowMatching.__name__}.id'), primary_key=True),
    sa.Column(Regularization.__name__, sa.ForeignKey(f'{Regularization.__name__}.id'), primary_key=True),
)


class ModelTrained(Model):
    __tablename__ = __qualname__
    __mapper_args__ = dict(
        polymorphic_on='sa_inheritance',
        polymorphic_identity=__tablename__,
    )
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey(f'{Model.__name__}.id'), primary_key=True),
        omegaconf_ignore=True,
    ))

    config_alt_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(sa.ForeignKey('Config.alt_id'), nullable=False),
        omegaconf_ignore=True,
    ))
    config: typing.Optional[typing.Any] = field(default=omegaconf.MISSING, metadata=dict(
        sa=orm.relationship('Config', foreign_keys=[config_alt_id.metadata['sa']])
    ))

    ckpt_dir: str = field(default='last', metadata=dict(sa=ColumnRequired(sa.String(len('epoch_##')))))
    sample_using_score: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    sample_event_conditioned: bool = field(default=False, metadata=dict(sa=ColumnRequired(sa.Boolean)))

    @staticmethod
    def transform_config(session, config_alt_id):
        if config_alt_id == omegaconf.MISSING:
            raise ValueError('Please set a config alt_id with model.config=<config_alt_id>.')
        config = (
            sa.select(Config)
            .where(Config.alt_id == config_alt_id)
        )
        config = session.execute(config)
        config = list(zip(range(2), config))
        assert len(config) == 1
        config = config[0][1][0]
        return config


class CkptMonitor(str, enum.Enum):
    TRAIN_LOSS_EMA = 'train_loss_ema'
    VAL_RELATIVE_ERROR_EMA = 'loss_val'
    FLOW_LOSS_EMA = 'flow_loss_ema'


class Config(CfgWithTable):
    __tablename__ = __qualname__
    __table_args__ = tuple()
    _target_: str = field(default=f'{MODULE_NAME}.{__qualname__}', repr=False)
    defaults: typing.List[typing.Any] = field(repr=False, default_factory=lambda: [
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_'
    ])

    root_dir: str = field(default=str(DIR_ROOT.resolve()))
    src_dir: str = field(default=str(DIR_SRC.resolve()))
    # data_dir: str = field(default=str(DIR_DATA.resolve()))
    out_dir: str = field(default=str((DIR_ROOT/'..'/'..'/'out'/'diffusion-dynamics'/'pmlr-v202-finzi23a').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='prediction.npy')
    device: str = field(default='cuda')

    id: int = field(init=False, metadata=dict(
        sa=sa.Column(sa.Integer, primary_key=True),
        omegaconf_ignore=True,
    ))
    alt_id: str = field(init=False, metadata=dict(
        sa=ColumnRequired(sa.String(8), index=True, unique=True),
        omegaconf_ignore=True
    ))
    rng_seed: int = field(default=42, metadata=dict(sa=ColumnRequired(sa.Integer)))
    fit: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    predict: bool = field(default=False, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    predict_dataset_split: str = field(default='train', metadata=dict(sa=ColumnRequired(sa.String(5))))
    check_val_every_n_epoch: int = field(default=100, metadata=dict(sa=ColumnRequired(sa.Integer)))
    use_ckpt_monitor: bool = field(default=True, metadata=dict(sa=ColumnRequired(sa.Boolean)))
    ckpt_monitor: CkptMonitor = field(default=CkptMonitor.VAL_RELATIVE_ERROR_EMA, metadata=dict(sa=ColumnRequired(sa.Enum(CkptMonitor))))

    model_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(Model.__name__, sa.ForeignKey(f'{Model.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    model: Model = field(default=omegaconf.MISSING, metadata=dict(sa=orm.relationship(Model.__name__, foreign_keys=[model_id.metadata['sa']])))

    dataset_id: int = field(init=False, repr=False, metadata=dict(
        sa=sa.Column(Dataset.__name__, sa.ForeignKey(f'{Dataset.__name__}.id'), nullable=False),
        omegaconf_ignore=True,
    ))
    dataset: Dataset = field(default=omegaconf.MISSING, metadata=dict(sa=orm.relationship(Dataset.__name__)))

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


@sa.event.listens_for(Config, 'before_insert')
def generate_random_string_id(mapper, connection, target):
    while True:
        target.alt_id = generate_random_string()
        if connection.execute(
            sa.select(Config.alt_id).where(Config.alt_id == target.alt_id)
        ).first() is None:
            break


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(group=Config.dataset.key, name=DatasetBatchShape.__name__, node=DatasetBatchShape)
cs.store(group=Config.dataset.key, name=DatasetGaussianMixture.__name__, node=DatasetGaussianMixture)
cs.store(group=Config.dataset.key, name=DatasetLorenz.__name__, node=DatasetLorenz)
cs.store(group=Config.dataset.key, name=DatasetFitzHughNagumo.__name__, node=DatasetFitzHughNagumo)
cs.store(group=Config.dataset.key, name=DatasetSimpleHarmonicOscillator.__name__, node=DatasetSimpleHarmonicOscillator)
cs.store(group=Config.model.key, name=ModelDiffusion.__name__, node=ModelDiffusion)
cs.store(group=f'{Config.model.key}/{ModelDiffusion.sde_diffusion.key}', name=SDEVarianceExploding.__name__, node=SDEVarianceExploding)
cs.store(group=Config.model.key, name=ModelFlowMatching.__name__, node=ModelFlowMatching)
cs.store(group=Config.model.key, name=ModelTrained.__name__, node=ModelTrained)
cs.store(group=f'{Config.model.key}/{ModelFlowMatching.conditional_flow.key}', name=ConditionalOT.__name__, node=ConditionalOT)
cs.store(group=f'{Config.model.key}/{ModelFlowMatching.conditional_flow.key}', name=MinibatchOTConditionalOT.__name__, node=MinibatchOTConditionalOT)
cs.store(group=f'{Config.model.key}/{ModelFlowMatching.conditional_flow.key}', name=ConditionalSDE.__name__, node=ConditionalSDE)
cs.store(group=f'{Config.model.key}/{ModelFlowMatching.conditional_flow.key}/{ConditionalSDE.sde_diffusion.key}', name=SDEVarianceExploding.__name__, node=SDEVarianceExploding)
cs.store(name=Config.__name__, node=Config)


def get_engine(dir=str(DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


def generate_random_string(k=8, chars=string.ascii_lowercase+string.digits):
    return ''.join(random.SystemRandom().choices(chars, k=k))


def get_new_config_alt_id():
    engine = get_engine()
    create_all(engine)
    with orm.Session(engine, expire_on_commit=False) as session:
        while True:
            alt_id = generate_random_string()
            if session.execute(
                sa.select(Config.alt_id).where(Config.alt_id == alt_id)
            ).first() is None:
                return alt_id


# OmegaConf.register_new_resolver('config_alt_id', get_new_config_alt_id, use_cache=True)


def create_all(engine):
    mapper_registry.metadata.create_all(engine)


def instantiate_and_insert_config(session, cfg):
    if not isinstance(cfg, (omegaconf.DictConfig, dict)):
        raise ValueError(f'Tried to instantiate: {cfg=}')
    record = {}
    m2m = {}
    table = globals()[cfg['_target_'].split('.')[1]]
    table_fields = {f.name: f for f in dataclasses.fields(table)}
    for k, v in cfg.items():
        if isinstance(v, enum.Enum):
            record[k] = v
        elif isinstance(v, (dict, omegaconf.DictConfig)):
            row = instantiate_and_insert_config(session, v)
            record[k] = row
        elif isinstance(v, (list, omegaconf.ListConfig)):
            if hasattr(table, f'transform_{k}') and callable(getattr(table, f'transform_{k}')):
                transform = getattr(table, f'transform_{k}')
                rows = transform(session, v)
            else:
                rows = [
                    instantiate_and_insert_config(session, vv) for vv in v
                ]
            m2m[k] = rows
        elif k != '_target_' and table_fields[k].init and 'sa' in table_fields[k].metadata:
            if hasattr(table, f'transform_{k}') and callable(getattr(table, f'transform_{k}')):
                transform = getattr(table, f'transform_{k}')
                v = transform(session, v)
            record[k] = v

    if len(m2m) > 0:
        if hasattr(table, '__mapper_args__') and 'polymorphic_identity' in table.__mapper_args__:
            table_alias_candidates = orm.aliased(
                table, sa.select(table).filter_by(**record, sa_inheritance=table.__mapper_args__['polymorphic_identity']).subquery('candidates')
            )
        else:
            table_alias_candidates = orm.aliased(
                table, sa.select(table).filter_by(**record).subquery('candidates')
            )
        subqueries = []
        for k, v in m2m.items():
            if len(v) > 0:
                table_related = v[0].__class__
                if hasattr(table_related, '__mapper_args__') and 'polymorphic_identity' in table.__mapper_args__:
                    table_related = table_related.__mro__[-3]
                has_subset_of_relations = orm.aliased(
                    table, (
                        sa.select(table_alias_candidates.id)
                        .join(getattr(table_alias_candidates, k))
                        .where(table_related.id.in_([vv.id for vv in v]))
                        .distinct()
                    ).subquery('has_subset_of_relations')
                )
                subquery = (
                    sa.select(has_subset_of_relations.id)
                    .join(getattr(has_subset_of_relations, k))
                    .group_by(has_subset_of_relations.id)
                    .having(sa.func.count(table_related.id) == len(v))
                )
                subqueries.append(subquery)
            else:
                m2m_rel = table_fields[k].metadata['sa']
                m2m_table_name = m2m_rel.parent.class_.__name__
                m2m_table_col = getattr(m2m_rel.secondary.c, m2m_table_name)
                # m2m_related_col = getattr(m2m_rel.secondary.c, m2m_rel.argument)
                has_relation = sa.select(m2m_table_col)
                subquery = (
                    sa.select(table_alias_candidates.id)
                    .where(table_alias_candidates.id.notin_(has_relation))
                )
                subqueries.append(subquery)
        query = sa.intersect(*subqueries)
        candidates_query = sa.select(table_alias_candidates).where(table_alias_candidates.id.in_(query))
        candidates = session.execute(candidates_query)
        candidates = list(zip(range(2), candidates))
        assert len(candidates) <= 1
        if len(candidates) == 1:
            row = candidates[0][1][0]
            return row

    # with session.no_autoflush:
    if len(m2m) == 0:
        if hasattr(table, '__mapper_args__') and 'polymorphic_identity' in table.__mapper_args__:
            saved_rows = session.execute(sa.select(table).filter_by(**record, sa_inheritance=table.__mapper_args__['polymorphic_identity']))
        else:
            saved_rows = session.execute(sa.select(table).filter_by(**record))
        saved_rows = list(zip(range(2), saved_rows))
        assert len(saved_rows) <= 1
        if len(saved_rows) == 1:
            row = saved_rows[0][1][0]
        else:
            row = table(**record)
            session.add(row)
            session.flush()
    else:
        for k, v in m2m.items():
            record[k] = v
        row = table(**record)
        session.add(row)
        session.flush()

    return row


def detach_config_from_session(table, row_id, session):
    stmt = sa.select(table).where(table.id == row_id).options(orm.joinedload('*'))
    sc = session.execute(stmt).unique().first()[0]
    return sc


def _map_enums(mapper, connection, target):
    for f in dataclasses.fields(target):
        if isinstance(f.type, enum.EnumMeta):
            table = f.type.table
            stmt = sa.select(table).where(getattr(table.c, f.type.__name__) == getattr(target, f.name))
            rows = connection.execute(stmt)
            _, rows = zip(*list(zip(range(2), rows)))
            assert len(rows) == 1
            setattr(target, f.name, rows[0].id)


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg):
    engine = get_engine()
    create_all(engine)



if __name__ == '__main__':
    main()
