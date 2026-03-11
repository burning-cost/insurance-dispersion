"""
Tests for diagnostics.py.

These tests verify that residual functions return correct shapes,
have approximately correct distributional properties under simulation,
and that the diagnostic DataFrame has the expected columns.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_dispersion import DGLM
from insurance_dispersion import families as fam
from insurance_dispersion import diagnostics

RNG = np.random.default_rng(333)


def make_fitted_result(n=400, seed=99):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = np.exp(1.5 + 0.5 * x1)
    phi = np.exp(-0.5 + 0.8 * z1)
    shape = 1.0 / phi
    y = rng.gamma(shape, mu * phi)
    df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})
    model = DGLM(
        formula="y ~ x1",
        dformula="~ z1",
        family=fam.Gamma(),
        data=df,
    )
    return model.fit(), df


class TestDiagnostics:
    def setup_method(self):
        self.result, self.df = make_fitted_result()

    def test_pearson_residuals_shape(self):
        r = diagnostics.pearson_residuals(self.result)
        assert r.shape == (len(self.df),)

    def test_pearson_residuals_finite(self):
        r = diagnostics.pearson_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_deviance_residuals_shape(self):
        r = diagnostics.deviance_residuals(self.result)
        assert r.shape == (len(self.df),)

    def test_deviance_residuals_finite(self):
        r = diagnostics.deviance_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_quantile_residuals_shape(self):
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (len(self.df),)

    def test_quantile_residuals_approx_normal(self):
        """Mean and std of quantile residuals should be close to 0 and 1."""
        r = diagnostics.quantile_residuals(self.result)
        # Loose bounds — sample size is modest
        assert abs(np.mean(r)) < 0.3, f"mean={np.mean(r):.3f}"
        assert 0.5 < np.std(r) < 2.0, f"std={np.std(r):.3f}"

    def test_qq_plot_data_shape(self):
        qq = diagnostics.qq_plot_data(self.result)
        assert isinstance(qq, pd.DataFrame)
        assert "theoretical" in qq.columns
        assert "observed" in qq.columns
        assert len(qq) == len(self.df)

    def test_dispersion_diagnostic_columns(self):
        diag = diagnostics.dispersion_diagnostic(self.result)
        assert isinstance(diag, pd.DataFrame)
        for col in ["fitted_mu", "fitted_phi", "unit_deviance", "scaled_deviance"]:
            assert col in diag.columns

    def test_dispersion_diagnostic_shape(self):
        diag = diagnostics.dispersion_diagnostic(self.result)
        assert len(diag) == len(self.df)

    def test_unit_deviances_nonneg(self):
        diag = diagnostics.dispersion_diagnostic(self.result)
        assert np.all(diag["unit_deviance"].values >= -1e-10)

    def test_scaled_deviances_positive(self):
        diag = diagnostics.dispersion_diagnostic(self.result)
        assert np.all(diag["scaled_deviance"].values > 0)


# ---------------------------------------------------------------------------
# Gaussian quantile residuals should be exactly normal
# ---------------------------------------------------------------------------

def test_gaussian_quantile_residuals():
    """For Gaussian with correct model, quantile residuals ~ N(0,1)."""
    rng = np.random.default_rng(77)
    n = 1000
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = 5.0 + 2.0 * x1
    phi = np.exp(0.3 + 0.5 * z1)
    y = rng.normal(mu, np.sqrt(phi))
    df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})

    model = DGLM(
        formula="y ~ x1",
        dformula="~ z1",
        family=fam.Gaussian(link="identity"),
        data=df,
    )
    result = model.fit()
    r = diagnostics.quantile_residuals(result)
    # With well-specified model, these should be N(0,1)
    assert abs(np.mean(r)) < 0.15, f"mean={np.mean(r):.3f}"
    assert 0.7 < np.std(r) < 1.3, f"std={np.std(r):.3f}"
