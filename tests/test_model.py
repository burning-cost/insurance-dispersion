"""
Tests for the DGLM class (model.py) and DGLMResult (results.py).

These tests use formulaic for design matrix construction and verify the full
end-to-end workflow: formula parsing, fitting, predictions, factor tables,
and the overdispersion LRT.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_dispersion import DGLM
from insurance_dispersion import families as fam

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_gamma_df(n=600, seed=42) -> pd.DataFrame:
    """DataFrame for a Gamma DGLM with categorical + continuous covariates."""
    rng = np.random.default_rng(seed)
    n_each = n // 3
    channel = np.array(["direct"] * n_each + ["broker"] * n_each + ["online"] * n_each)
    x_cont = rng.uniform(0, 1, n)

    # Mean: log(mu) = 2 + 0.3*broker + 0.5*online + 0.4*x_cont
    beta = {"direct": 2.0, "broker": 0.3, "online": 0.5}
    mu = np.exp(
        np.array([beta[c] for c in channel]) + 0.4 * x_cont
    )

    # Dispersion: log(phi) = -1 + 1.0*(channel==broker)
    phi = np.exp(
        -1.0 + 1.0 * (channel == "broker").astype(float)
    )

    shape = 1.0 / phi
    y = rng.gamma(shape, mu * phi)

    return pd.DataFrame({
        "claim": y,
        "channel": channel,
        "x_cont": x_cont,
    })


def make_gaussian_df(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = 3.0 + 2.0 * x1
    phi = np.exp(0.5 + 1.0 * z1)
    y = rng.normal(mu, np.sqrt(phi))
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1})


# ---------------------------------------------------------------------------
# Basic model instantiation and fit
# ---------------------------------------------------------------------------

class TestDGLMFit:
    def test_fit_gamma(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        assert result.converged
        assert result.loglik < 0  # log-likelihood negative for Gamma

    def test_fit_gaussian(self):
        df = make_gaussian_df()
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Gaussian(link="identity"),
            data=df,
        )
        result = model.fit()
        assert result.converged

    def test_fit_reml(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            method="reml",
            data=df,
        )
        result = model.fit()
        assert result.converged

    def test_fit_stores_mu_phi(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ 1",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        assert result.mu_.shape == (len(df),)
        assert result.phi_.shape == (len(df),)
        assert np.all(result.mu_ > 0)
        assert np.all(result.phi_ > 0)

    def test_fit_no_data_raises(self):
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Gamma(),
        )
        with pytest.raises(ValueError, match="No data"):
            model.fit()

    def test_fit_with_exposure(self):
        rng = np.random.default_rng(33)
        n = 300
        exposure = rng.uniform(0.5, 2.0, n)
        x1 = rng.uniform(0, 1, n)
        mu = exposure * np.exp(1.5 + 0.5 * x1)
        phi = 0.3
        y = rng.gamma(1.0 / phi, mu * phi)
        df = pd.DataFrame({"y": y, "x1": x1, "exposure": exposure})
        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.Gamma(),
            data=df,
            exposure="exposure",
        )
        result = model.fit()
        assert result.converged
        # Intercept should be ~1.5
        assert abs(result.mean_model.coef[0] - 1.5) < 0.2


# ---------------------------------------------------------------------------
# Factor tables
# ---------------------------------------------------------------------------

class TestRelativities:
    def test_mean_relativities_shape(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        rel = result.mean_relativities()
        assert isinstance(rel, pd.DataFrame)
        assert "coef" in rel.columns
        assert "exp_coef" in rel.columns
        assert "se" in rel.columns
        assert "p_value" in rel.columns

    def test_dispersion_relativities_shape(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        rel = result.dispersion_relativities()
        assert isinstance(rel, pd.DataFrame)
        assert "exp_coef" in rel.columns

    def test_exp_coef_matches_exp_coef(self):
        """exp_coef column must equal exp(coef) within float precision."""
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        rel = result.mean_relativities()
        assert np.allclose(rel["exp_coef"].values, np.exp(rel["coef"].values), rtol=1e-10)

    def test_dispersion_exp_coef_matches(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        rel = result.dispersion_relativities()
        assert np.allclose(rel["exp_coef"].values, np.exp(rel["coef"].values), rtol=1e-10)

    def test_se_positive(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        assert np.all(result.mean_relativities()["se"].values > 0)
        assert np.all(result.dispersion_relativities()["se"].values > 0)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestPredictions:
    def setup_method(self):
        df = make_gamma_df(n=600, seed=10)
        self.df = df
        self.model = DGLM(
            formula="claim ~ C(channel) + x_cont",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        self.result = self.model.fit()

    def test_predict_mean_shape(self):
        pred = self.result.predict(self.df, which="mean")
        assert pred.shape == (len(self.df),)

    def test_predict_dispersion_shape(self):
        pred = self.result.predict(self.df, which="dispersion")
        assert pred.shape == (len(self.df),)

    def test_predict_variance_shape(self):
        pred = self.result.predict(self.df, which="variance")
        assert pred.shape == (len(self.df),)

    def test_predict_mean_positive(self):
        pred = self.result.predict(self.df, which="mean")
        assert np.all(pred > 0)

    def test_predict_dispersion_positive(self):
        pred = self.result.predict(self.df, which="dispersion")
        assert np.all(pred > 0)

    def test_predict_variance_positive(self):
        pred = self.result.predict(self.df, which="variance")
        assert np.all(pred > 0)

    def test_predict_mean_matches_mu(self):
        """In-sample predict(which='mean') must match mu_."""
        pred = self.result.predict(self.df, which="mean")
        assert np.allclose(pred, self.result.mu_, rtol=1e-8)

    def test_predict_dispersion_matches_phi(self):
        """In-sample predict(which='dispersion') must match phi_."""
        pred = self.result.predict(self.df, which="dispersion")
        assert np.allclose(pred, self.result.phi_, rtol=1e-8)

    def test_predict_variance_equals_phi_V_mu(self):
        mu_pred = self.result.predict(self.df, which="mean")
        phi_pred = self.result.predict(self.df, which="dispersion")
        var_pred = self.result.predict(self.df, which="variance")
        V = fam.Gamma().variance(mu_pred)
        assert np.allclose(var_pred, phi_pred * V, rtol=1e-8)

    def test_predict_newdata(self):
        """Prediction on a small new dataset should work."""
        newdata = pd.DataFrame({
            "channel": ["direct", "broker", "online"],
            "x_cont": [0.3, 0.7, 0.5],
        })
        pred = self.result.predict(newdata, which="mean")
        assert pred.shape == (3,)
        assert np.all(pred > 0)

    def test_predict_invalid_which(self):
        with pytest.raises(ValueError, match="which must be"):
            self.result.predict(self.df, which="invalid")


# ---------------------------------------------------------------------------
# Overdispersion LRT
# ---------------------------------------------------------------------------

class TestOverdispersionTest:
    def test_lrt_rejects_when_phi_varies(self):
        """LRT should reject H0 (constant phi) when phi genuinely varies."""
        df = make_gamma_df(n=1000, seed=50)
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        test = result.overdispersion_test()
        assert test["p_value"] < 0.05, f"p_value={test['p_value']:.4f}"
        assert "Reject" in test["conclusion"]

    def test_lrt_returns_correct_keys(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        test = result.overdispersion_test()
        assert "statistic" in test
        assert "df" in test
        assert "p_value" in test
        assert "conclusion" in test

    def test_lrt_statistic_positive(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        test = result.overdispersion_test()
        assert test["statistic"] >= 0


# ---------------------------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------------------------

class TestInformationCriteria:
    def test_aic_formula(self):
        df = make_gamma_df()
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        p_mean = len(result.mean_model.coef)
        p_disp = len(result.dispersion_model.coef)
        expected_aic = -2 * result.loglik + 2 * (p_mean + p_disp)
        assert abs(result.aic - expected_aic) < 1e-8

    def test_bic_formula(self):
        df = make_gamma_df(n=600)
        model = DGLM(
            formula="claim ~ C(channel)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=df,
        )
        result = model.fit()
        n = len(df)
        p_mean = len(result.mean_model.coef)
        p_disp = len(result.dispersion_model.coef)
        expected_bic = -2 * result.loglik + np.log(n) * (p_mean + p_disp)
        assert abs(result.bic - expected_bic) < 1e-8


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_str():
    df = make_gamma_df()
    model = DGLM(
        formula="claim ~ C(channel)",
        dformula="~ C(channel)",
        family=fam.Gamma(),
        data=df,
    )
    result = model.fit()
    s = result.summary()
    assert "Double GLM" in s
    assert "Mean Submodel" in s
    assert "Dispersion Submodel" in s
    assert "AIC" in s


# ---------------------------------------------------------------------------
# Intercept-only dispersion vs constant-phi
# ---------------------------------------------------------------------------

def test_intercept_only_disp_constant_phi():
    """
    DGLM with dformula='~1' should produce a constant phi vector.
    """
    df = make_gamma_df()
    model = DGLM(
        formula="claim ~ C(channel)",
        dformula="~ 1",
        family=fam.Gamma(),
        data=df,
    )
    result = model.fit()
    # phi should be constant (all same value)
    assert np.std(result.phi_) < 1e-8, f"phi not constant: std={np.std(result.phi_):.4e}"
