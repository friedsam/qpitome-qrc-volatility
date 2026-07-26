import numpy as np

from baselines.garch import GARCHConfig, variance_path_to_log_volatility_path


def test_garch_config_and_logvol_conversion():
    config = GARCHConfig(backend="arch")
    assert config.p == config.q == 1
    assert config.distribution == "StudentsT"
    path = variance_path_to_log_volatility_path(
        np.array([4.0, 9.0]),
        return_scale=100.0,
    )
    assert np.allclose(path, np.log([0.02, 0.03]))
