import pytest
import hydra

from userfm import cs

@pytest.fixture
def engine():
    engine = cs.sa.create_engine('sqlite+pysqlite:///:memory:')
    cs.create_all(engine)
    return engine


def init_hydra_cfg(config_name, overrides, config_dir=str(cs.DIR_ROOT/'conf')):
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=overrides)
