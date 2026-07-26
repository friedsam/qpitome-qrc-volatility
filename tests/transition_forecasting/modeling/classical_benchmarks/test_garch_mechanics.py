from __future__ import annotations

import numpy as np

from baselines.garch import GARCHConfig, fit_garch_variance_path, variance_path_to_log_volatility_path


def test_scipy_garch_forecast_is_finite() -> None:
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0, 0.01, 500)
    result = fit_garch_variance_path(returns, horizon=10, config=GARCHConfig(backend="scipy"))
    assert result.converged
    assert result.variance_path.shape == (10,)
    assert np.all(result.variance_path > 0)
    logvol = variance_path_to_log_volatility_path(result.variance_path)
    assert np.isfinite(logvol).all()
