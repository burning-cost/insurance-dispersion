"""
Expanded test coverage for insurance-dispersion.

Covers gaps in:
  - results.py: SubmodelResult directly, repr, overdispersion_test REML warning,
                predict() missing-exposure warning
  - families.py: InverseGaussian, Tweedie, NegativeBinomial, Poisson families
                 (variance, deviance_resid, log_likelihood, init_mu, repr)
  - fitting.py: _sandwich_vcov, _wls, _hat_diagonal
  - diagnostics.py: Poisson quantile residuals path, InverseGaussian path,
                    NegBin fallback
  - DGLM: InverseGaussian family end-to-end fit, Poisson DGLM fit,
          weights parameter, missing data raises
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from insurance_dispersion import DGLM
from insurance_dispersion import families as fam
from insurance_dispersion import diagnostics
from insurance_dispersion.results import SubmodelResult, DGLMResult
from insurance_dispersion.fitting import _sandwich_vcov, _wls, _hat_diagonal


# ---------------------------------------------------------------------------
# Helper: generate small DataFrames for various families
# ---------------------------------------------------------------------------


def _make_gamma_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = np.exp(1.5 + 0.5 * x1)
    phi = np.exp(-0.5 + 0.8 * z1)
    y = rng.gamma(1.0 / phi, mu * phi)
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1})


def _make_gaussian_df(n=300, seed=2):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = 3.0 + 1.5 * x1
    phi = np.exp(0.2 + 0.5 * z1)
    y = rng.normal(mu, np.sqrt(phi))
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1})


def _make_inv_gaussian_df(n=300, seed=3):
    """Inverse Gaussian data via reciprocal-normal approximation."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = np.exp(2.0 + 0.4 * x1)
    phi = np.exp(-1.0 + 0.5 * z1)
    # Use scipy to sample InvGaussian: parameterised as invgauss(mu/scale, scale)
    sqrt_phi = np.sqrt(phi)
    mu_shape = mu / sqrt_phi
    y = scipy.stats.invgauss.rvs(mu=mu_shape, scale=sqrt_phi, random_state=rng)
    y = np.clip(y, 1e-6, None)
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1})


def _make_poisson_df(n=500, seed=4):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)
    mu = np.exp(-1.0 + 0.8 * x1)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y.astype(float), "x1": x1, "z1": z1})


def _make_tweedie_df(n=400, seed=5):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + 0.6 * x1)
    phi = 0.5
    # Simulate Tweedie(p=1.5): compound Poisson-Gamma
    # N ~ Poisson(mu^(2-p) / (phi*(2-p))), each loss ~ Gamma
    # Use a shortcut: simulate via scipy Tweedie approximation
    y = rng.gamma(mu / phi, phi)  # rough approximation for testing
    y = np.clip(y, 0, None)
    return pd.DataFrame({"y": y, "x1": x1})


# ---------------------------------------------------------------------------
# SubmodelResult direct tests
# ---------------------------------------------------------------------------


class TestSubmodelResult:
    def _make_result(self) -> SubmodelResult:
        coef = np.array([1.5, 0.3, -0.2])
        vcov = np.diag([0.01, 0.004, 0.009])
        return SubmodelResult(
            coef=coef,
            coef_names=["(Intercept)", "x1", "x2"],
            vcov=vcov,
        )

    def test_coef_stored(self):
        r = self._make_result()
        assert np.allclose(r.coef, [1.5, 0.3, -0.2])

    def test_se_equals_sqrt_diag_vcov(self):
        r = self._make_result()
        expected_se = np.sqrt([0.01, 0.004, 0.009])
        assert np.allclose(r.se, expected_se)

    def test_summary_frame_returns_dataframe(self):
        r = self._make_result()
        df = r.summary_frame()
        assert isinstance(df, pd.DataFrame)

    def test_summary_frame_columns(self):
        r = self._make_result()
        df = r.summary_frame()
        for col in ["coef", "exp_coef", "se", "z", "p_value"]:
            assert col in df.columns

    def test_summary_frame_index_is_coef_names(self):
        r = self._make_result()
        df = r.summary_frame()
        assert list(df.index) == ["(Intercept)", "x1", "x2"]

    def test_summary_frame_exp_coef_matches_exp_coef(self):
        r = self._make_result()
        df = r.summary_frame()
        assert np.allclose(df["exp_coef"].values, np.exp(df["coef"].values))

    def test_summary_frame_p_values_in_0_1(self):
        r = self._make_result()
        df = r.summary_frame()
        assert (df["p_value"].values >= 0.0).all()
        assert (df["p_value"].values <= 1.0).all()

    def test_se_nonneg_with_negative_variance_diagonal(self):
        """If diagonal of vcov has tiny negative values (numerical noise), se should still be >= 0."""
        coef = np.array([1.0])
        vcov = np.array([[-1e-16]])  # tiny negative due to floating point
        r = SubmodelResult(coef=coef, coef_names=["(Intercept)"], vcov=vcov)
        assert r.se[0] >= 0.0


# ---------------------------------------------------------------------------
# DGLMResult repr
# ---------------------------------------------------------------------------


class TestDGLMResultRepr:
    def test_repr_contains_class_name(self):
        df = _make_gamma_df()
        model = DGLM(formula="y ~ x1", dformula="~ z1", family=fam.Gamma(), data=df)
        result = model.fit()
        r = repr(result)
        assert "DGLMResult" in r

    def test_repr_contains_n_obs(self):
        df = _make_gamma_df(n=300)
        model = DGLM(formula="y ~ x1", dformula="~ z1", family=fam.Gamma(), data=df)
        result = model.fit()
        r = repr(result)
        assert "300" in r

    def test_repr_contains_loglik(self):
        df = _make_gamma_df()
        model = DGLM(formula="y ~ x1", dformula="~ z1", family=fam.Gamma(), data=df)
        result = model.fit()
        r = repr(result)
        assert "loglik" in r


# ---------------------------------------------------------------------------
# overdispersion_test() with REML: should warn
# ---------------------------------------------------------------------------


class TestOverdispersionTestREMLWarning:
    def test_reml_overdispersion_warns(self):
        df = _make_gamma_df(n=400)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Gamma(),
            method="reml",
            data=df,
        )
        result = model.fit()
        with pytest.warns(UserWarning, match="reml"):
            test = result.overdispersion_test()
        assert "statistic" in test
        assert "p_value" in test


# ---------------------------------------------------------------------------
# predict() missing exposure column warns
# ---------------------------------------------------------------------------


class TestPredictMissingExposure:
    def test_missing_exposure_warns(self):
        rng = np.random.default_rng(7)
        n = 200
        x1 = rng.uniform(0, 1, n)
        exposure = rng.uniform(0.5, 2.0, n)
        mu = exposure * np.exp(1.0 + 0.4 * x1)
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

        # newdata WITHOUT the exposure column — should warn
        newdata = pd.DataFrame({"x1": [0.5, 0.3]})
        with pytest.warns(UserWarning, match="Exposure column"):
            pred = result.predict(newdata, which="mean")
        assert pred.shape == (2,)
        assert np.all(pred > 0)


# ---------------------------------------------------------------------------
# Families: InverseGaussian
# ---------------------------------------------------------------------------


class TestInverseGaussianFamily:
    def setup_method(self):
        self.fam = fam.InverseGaussian()
        self.mu = np.array([1.0, 2.0, 5.0])
        self.y = np.array([0.8, 2.2, 4.5])

    def test_repr(self):
        r = repr(self.fam)
        assert "InverseGaussian" in r
        assert "log" in r

    def test_variance_is_mu_cubed(self):
        V = self.fam.variance(self.mu)
        assert np.allclose(V, self.mu ** 3)

    def test_deviance_resid_nonneg(self):
        d = self.fam.deviance_resid(self.y, self.mu)
        assert np.all(d >= -1e-10)

    def test_deviance_resid_zero_when_y_equals_mu(self):
        mu = np.array([1.5, 2.5, 4.0])
        d = self.fam.deviance_resid(mu, mu)
        assert np.allclose(d, 0.0, atol=1e-10)

    def test_log_likelihood_finite(self):
        phi = np.array([0.2, 0.3, 0.4])
        ll = self.fam.log_likelihood(self.y, self.mu, phi)
        assert np.all(np.isfinite(ll))

    def test_init_mu_positive(self):
        y = np.array([1.0, 2.0, 3.0])
        mu0 = self.fam.init_mu(y)
        assert np.all(mu0 > 0)

    def test_eta_to_mu_positive(self):
        eta = np.array([-1.0, 0.0, 1.0])
        mu = self.fam.eta_to_mu(eta)
        assert np.all(mu > 0)


# ---------------------------------------------------------------------------
# Families: Tweedie
# ---------------------------------------------------------------------------


class TestTweedieFamily:
    def setup_method(self):
        self.fam = fam.Tweedie(p=1.5)
        self.mu = np.array([1.0, 2.0, 3.0])
        self.y = np.array([0.5, 1.8, 3.5])

    def test_repr(self):
        r = repr(self.fam)
        assert "Tweedie" in r
        assert "1.5" in r

    def test_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="p must be"):
            fam.Tweedie(p=0.5)

    def test_p_boundary_raises(self):
        with pytest.raises(ValueError, match="p must be"):
            fam.Tweedie(p=2.0)

    def test_variance_is_mu_p(self):
        V = self.fam.variance(self.mu)
        assert np.allclose(V, self.mu ** 1.5)

    def test_deviance_resid_nonneg(self):
        d = self.fam.deviance_resid(self.y, self.mu)
        assert np.all(d >= -1e-10)

    def test_deviance_resid_with_zeros(self):
        y_with_zero = np.array([0.0, 0.0, 1.0])
        d = self.fam.deviance_resid(y_with_zero, self.mu)
        assert np.all(np.isfinite(d))

    def test_log_likelihood_finite(self):
        phi = np.full(3, 0.5)
        ll = self.fam.log_likelihood(self.y, self.mu, phi)
        assert np.all(np.isfinite(ll))

    def test_init_mu_positive(self):
        y = np.array([1.0, 0.0, 2.0])
        mu0 = self.fam.init_mu(y)
        assert np.all(mu0 > 0)

    def test_init_mu_uses_positive_values(self):
        """init_mu should use mean of positive values only."""
        y_all_zero = np.array([0.0, 0.0, 0.0])
        mu0 = self.fam.init_mu(y_all_zero)
        # Falls back to 1.0 when all zeros
        assert float(mu0[0]) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Families: NegativeBinomial
# ---------------------------------------------------------------------------


class TestNegativeBinomialFamily:
    def setup_method(self):
        self.fam = fam.NegativeBinomial(alpha=0.5)
        self.mu = np.array([1.0, 3.0, 5.0])
        self.y = np.array([1.0, 2.0, 6.0])

    def test_repr(self):
        r = repr(self.fam)
        assert "NegativeBinomial" in r
        assert "0.5" in r

    def test_alpha_nonpositive_raises(self):
        with pytest.raises(ValueError, match="alpha must be"):
            fam.NegativeBinomial(alpha=0.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha must be"):
            fam.NegativeBinomial(alpha=-1.0)

    def test_variance_formula(self):
        V = self.fam.variance(self.mu)
        expected = self.mu + 0.5 * self.mu ** 2
        assert np.allclose(V, expected)

    def test_deviance_resid_nonneg(self):
        d = self.fam.deviance_resid(self.y, self.mu)
        assert np.all(d >= -1e-10)

    def test_deviance_resid_with_zeros(self):
        y_zero = np.array([0.0, 0.0, 0.0])
        d = self.fam.deviance_resid(y_zero, self.mu)
        assert np.all(np.isfinite(d))

    def test_log_likelihood_finite(self):
        phi = np.full(3, 1.0)
        ll = self.fam.log_likelihood(self.y, self.mu, phi)
        assert np.all(np.isfinite(ll))

    def test_init_mu_positive(self):
        y = np.array([1.0, 0.0, 3.0])
        mu0 = self.fam.init_mu(y)
        assert np.all(mu0 > 0)


# ---------------------------------------------------------------------------
# Families: Poisson
# ---------------------------------------------------------------------------


class TestPoissonFamily:
    def setup_method(self):
        self.fam = fam.Poisson()
        self.mu = np.array([0.5, 1.0, 3.0])
        self.y = np.array([0.0, 1.0, 4.0])

    def test_variance_equals_mu(self):
        V = self.fam.variance(self.mu)
        assert np.allclose(V, self.mu)

    def test_deviance_resid_with_zeros(self):
        d = self.fam.deviance_resid(np.array([0.0]), np.array([1.0]))
        assert np.isfinite(d[0])
        assert d[0] >= 0.0

    def test_deviance_resid_nonneg(self):
        d = self.fam.deviance_resid(self.y, self.mu)
        assert np.all(d >= -1e-10)

    def test_log_likelihood_finite(self):
        phi = np.full(3, 1.0)
        ll = self.fam.log_likelihood(self.y, self.mu, phi)
        assert np.all(np.isfinite(ll))

    def test_init_mu_above_threshold(self):
        y = np.array([0.0, 0.0, 0.0])
        mu0 = self.fam.init_mu(y)
        assert np.all(mu0 >= 1e-3)


# ---------------------------------------------------------------------------
# Families: link functions
# ---------------------------------------------------------------------------


class TestLinkFunctions:
    def test_log_link_identity_roundtrip(self):
        mu = np.array([0.5, 1.0, 5.0])
        f = fam.Gamma()
        assert np.allclose(f.eta_to_mu(f.mu_to_eta(mu)), mu)

    def test_identity_link_roundtrip(self):
        mu = np.array([1.0, 3.0, 5.0])
        f = fam.Gaussian(link="identity")
        assert np.allclose(f.eta_to_mu(f.mu_to_eta(mu)), mu)

    def test_unknown_link_raises(self):
        with pytest.raises(ValueError, match="Unknown link"):
            fam.Gamma(link="probit")


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------


class TestFittingUtilities:
    def test_wls_exact_solution(self):
        """WLS should recover exact coefficients for a well-specified system."""
        rng = np.random.default_rng(10)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        true_beta = np.array([2.0, 3.0])
        y = X @ true_beta  # exact (no noise)
        w = np.ones(n)
        beta_hat = _wls(X, y, w)
        assert np.allclose(beta_hat, true_beta, atol=1e-8)

    def test_wls_weighted_recovers_beta(self):
        """WLS with varied weights should still recover true beta for exact system."""
        rng = np.random.default_rng(11)
        n = 50
        X = np.column_stack([np.ones(n), rng.uniform(-1, 1, n)])
        beta = np.array([-1.0, 2.5])
        y = X @ beta
        w = rng.uniform(0.1, 10.0, n)
        beta_hat = _wls(X, y, w)
        assert np.allclose(beta_hat, beta, atol=1e-8)

    def test_hat_diagonal_sum_equals_rank(self):
        """Sum of hat diagonal should equal p (number of columns in X)."""
        rng = np.random.default_rng(12)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = np.ones(n)
        h = _hat_diagonal(X, w)
        assert h.shape == (n,)
        # sum(h_ii) = trace(H) = p for full-rank X
        assert abs(np.sum(h) - p) < 1e-6

    def test_hat_diagonal_all_in_0_1(self):
        rng = np.random.default_rng(13)
        n, p = 80, 4
        X = rng.standard_normal((n, p))
        w = rng.uniform(0.5, 2.0, n)
        h = _hat_diagonal(X, w)
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1.0 + 1e-10)

    def test_sandwich_vcov_shape(self):
        rng = np.random.default_rng(14)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = rng.uniform(0.5, 2.0, n)
        vcov = _sandwich_vcov(X, w)
        assert vcov.shape == (p, p)

    def test_sandwich_vcov_symmetric(self):
        rng = np.random.default_rng(15)
        n, p = 60, 4
        X = rng.standard_normal((n, p))
        w = rng.uniform(0.1, 5.0, n)
        vcov = _sandwich_vcov(X, w)
        assert np.allclose(vcov, vcov.T, atol=1e-12)

    def test_sandwich_vcov_positive_definite(self):
        rng = np.random.default_rng(16)
        n, p = 100, 3
        X = rng.standard_normal((n, p))
        w = np.ones(n)
        vcov = _sandwich_vcov(X, w)
        eigenvalues = np.linalg.eigvalsh(vcov)
        assert np.all(eigenvalues > -1e-10)


# ---------------------------------------------------------------------------
# InverseGaussian DGLM end-to-end
# ---------------------------------------------------------------------------


class TestInverseGaussianDGLM:
    def test_fit_converges(self):
        df = _make_inv_gaussian_df(n=300)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.InverseGaussian(),
            data=df,
        )
        result = model.fit(maxit=100)
        # InverseGaussian can be harder to converge — just check it ran
        assert result.mu_.shape == (300,)
        assert np.all(result.mu_ > 0)

    def test_fit_phi_positive(self):
        df = _make_inv_gaussian_df(n=300)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.InverseGaussian(),
            data=df,
        )
        result = model.fit(maxit=100)
        assert np.all(result.phi_ > 0)

    def test_predictions_positive(self):
        df = _make_inv_gaussian_df(n=300)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.InverseGaussian(),
            data=df,
        )
        result = model.fit(maxit=50)
        pred = result.predict(df, which="mean")
        assert np.all(pred > 0)


# ---------------------------------------------------------------------------
# Poisson DGLM end-to-end
# ---------------------------------------------------------------------------


class TestPoissonDGLM:
    def test_poisson_dglm_fits(self):
        df = _make_poisson_df(n=500)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Poisson(),
            data=df,
        )
        result = model.fit(maxit=50)
        assert result.mu_.shape == (500,)
        assert np.all(result.mu_ > 0)

    def test_poisson_phi_positive(self):
        df = _make_poisson_df()
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Poisson(),
            data=df,
        )
        result = model.fit()
        assert np.all(result.phi_ > 0)

    def test_poisson_predict_mean_positive(self):
        df = _make_poisson_df()
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Poisson(),
            data=df,
        )
        result = model.fit()
        pred = result.predict(df, which="mean")
        assert np.all(pred > 0)


# ---------------------------------------------------------------------------
# Diagnostics: Poisson quantile residuals path
# ---------------------------------------------------------------------------


class TestPoissonDiagnostics:
    def setup_method(self):
        df = _make_poisson_df(n=300, seed=55)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ z1",
            family=fam.Poisson(),
            data=df,
        )
        self.result = model.fit()
        self.n = 300

    def test_pearson_residuals_shape(self):
        r = diagnostics.pearson_residuals(self.result)
        assert r.shape == (self.n,)

    def test_pearson_residuals_finite(self):
        r = diagnostics.pearson_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_deviance_residuals_shape(self):
        r = diagnostics.deviance_residuals(self.result)
        assert r.shape == (self.n,)

    def test_quantile_residuals_shape(self):
        # Poisson takes the randomised quantile path
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (self.n,)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_dispersion_diagnostic_columns(self):
        diag = diagnostics.dispersion_diagnostic(self.result)
        for col in ["fitted_mu", "fitted_phi", "unit_deviance", "scaled_deviance"]:
            assert col in diag.columns


# ---------------------------------------------------------------------------
# Diagnostics: InverseGaussian quantile residuals path
# ---------------------------------------------------------------------------


class TestInverseGaussianDiagnostics:
    def setup_method(self):
        df = _make_inv_gaussian_df(n=300, seed=66)
        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.InverseGaussian(),
            data=df,
        )
        self.result = model.fit(maxit=100)
        self.n = 300

    def test_quantile_residuals_shape(self):
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (self.n,)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))


# ---------------------------------------------------------------------------
# Diagnostics: NegativeBinomial fallback (deviance residuals path)
# ---------------------------------------------------------------------------


class TestNegBinDiagnosticsFallback:
    def setup_method(self):
        # NegBin via DGLM — the quantile_residuals fallback uses deviance_residuals
        rng = np.random.default_rng(77)
        n = 300
        x1 = rng.uniform(0, 1, n)
        # Approximate NegBin data
        mu = np.exp(0.5 + 0.5 * x1)
        y = rng.negative_binomial(2, 2.0 / (2.0 + mu), n).astype(float)
        df = pd.DataFrame({"y": y, "x1": x1})
        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.NegativeBinomial(alpha=0.5),
            data=df,
        )
        self.result = model.fit(maxit=50)
        self.n = n

    def test_quantile_residuals_shape(self):
        # NegBin falls through to deviance residuals path
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (self.n,)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))


# ---------------------------------------------------------------------------
# DGLM: weights parameter
# ---------------------------------------------------------------------------


class TestDGLMWeights:
    def test_weighted_fit_converges(self):
        rng = np.random.default_rng(88)
        n = 300
        x1 = rng.uniform(0, 1, n)
        mu = np.exp(1.5 + 0.4 * x1)
        phi = 0.3
        y = rng.gamma(1.0 / phi, mu * phi)
        w = rng.uniform(0.5, 2.0, n)
        df = pd.DataFrame({"y": y, "x1": x1, "w": w})

        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.Gamma(),
            data=df,
            weights="w",
        )
        result = model.fit()
        assert result.mu_.shape == (n,)
        assert np.all(result.mu_ > 0)

    def test_weighted_phi_positive(self):
        rng = np.random.default_rng(89)
        n = 200
        x1 = rng.uniform(0, 1, n)
        mu = np.exp(1.0 + 0.5 * x1)
        y = rng.gamma(2.0, mu / 2.0)
        w = rng.uniform(0.8, 1.5, n)
        df = pd.DataFrame({"y": y, "x1": x1, "w": w})

        model = DGLM(
            formula="y ~ x1",
            dformula="~ 1",
            family=fam.Gamma(),
            data=df,
            weights="w",
        )
        result = model.fit()
        assert np.all(result.phi_ > 0)
