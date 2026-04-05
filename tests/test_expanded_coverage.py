"""
Expanded test coverage for insurance-dispersion.

Covers the gaps identified relative to the public API surface:

1. families.py — link deriv/inverse_deriv, _get_link errors, non-default
   links, repr, init_mu, log-likelihood correctness for all families,
   Tweedie with zero responses, InverseGaussian and NegBin edge cases.

2. results.py — SubmodelResult.summary_frame() columns, DGLMResult.summary()
   content, overdispersion_test() REML warning, predict() with missing
   exposure column, DGLMResult repr.

3. diagnostics.py — InverseGaussian quantile residuals, Poisson quantile
   residuals (randomised), NegBin fallback to deviance residuals,
   Tweedie quantile residuals fallback.

4. fitting.py — dglm_fit() verbose mode, non-convergence warning,
   _sandwich_vcov() with near-singular matrix, prior_weights, log_offset.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_dispersion import DGLM
from insurance_dispersion import diagnostics
from insurance_dispersion import families as fam
from insurance_dispersion.families import (
    Gaussian,
    Gamma,
    InverseGaussian,
    Tweedie,
    Poisson,
    NegativeBinomial,
    LogLink,
    IdentityLink,
    InverseLink,
    _get_link,
)
from insurance_dispersion.fitting import (
    _wls,
    _hat_diagonal,
    _sandwich_vcov,
    dglm_fit,
)

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gamma_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + 0.5 * x)
    phi = np.exp(-0.5 + 0.6 * z)
    y = rng.gamma(1.0 / phi, mu * phi)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def make_gaussian_df(n: int = 300, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu = 3.0 + 2.0 * x
    phi = np.exp(0.2 + 0.4 * z)
    y = rng.normal(mu, np.sqrt(phi))
    return pd.DataFrame({"y": y, "x": x, "z": z})


def make_poisson_df(n: int = 300, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + 0.5 * x)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x": x})


def make_ig_df(n: int = 300, seed: int = 10) -> pd.DataFrame:
    """InverseGaussian data using scipy.stats.invgauss."""
    from scipy.stats import invgauss
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.0 + 0.3 * x)
    phi = 0.4 * np.ones(n)
    # invgauss(mu/scale, scale=sqrt(phi)) has mean mu when scale=sqrt(phi)
    sqrt_phi = np.sqrt(phi)
    y = invgauss.rvs(mu / sqrt_phi, scale=sqrt_phi, random_state=rng)
    y = np.clip(y, 1e-6, None)
    return pd.DataFrame({"y": y, "x": x})


# ===========================================================================
# 1. Link functions
# ===========================================================================

class TestLinkDeriv:
    """Test the deriv() and inverse_deriv() methods on all link classes."""

    def test_log_link_deriv_equals_1_over_mu(self):
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        link = LogLink()
        d = link.deriv(mu)
        assert np.allclose(d, 1.0 / mu)

    def test_log_link_inverse_deriv_equals_exp_eta(self):
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        link = LogLink()
        d = link.inverse_deriv(eta)
        assert np.allclose(d, np.exp(eta))

    def test_identity_link_deriv_is_one(self):
        mu = np.array([1.0, 2.0, 5.0])
        link = IdentityLink()
        assert np.allclose(link.deriv(mu), 1.0)

    def test_identity_link_inverse_deriv_is_one(self):
        eta = np.array([1.0, 2.0, 5.0])
        link = IdentityLink()
        assert np.allclose(link.inverse_deriv(eta), 1.0)

    def test_inverse_link_deriv_equals_neg_1_over_mu_sq(self):
        mu = np.array([1.0, 2.0, 4.0])
        link = InverseLink()
        d = link.deriv(mu)
        assert np.allclose(d, -1.0 / mu ** 2)

    def test_inverse_link_inverse_deriv_equals_neg_1_over_eta_sq(self):
        eta = np.array([0.5, 1.0, 2.0])
        link = InverseLink()
        d = link.inverse_deriv(eta)
        assert np.allclose(d, -1.0 / eta ** 2)

    def test_log_link_numerically_stable_at_near_zero(self):
        """Deriv should not raise or return inf at near-zero mu."""
        mu = np.array([1e-300])
        link = LogLink()
        d = link.deriv(mu)
        assert np.all(np.isfinite(d))

    def test_inverse_link_numerically_stable_at_near_zero(self):
        """Inverse link should not raise at near-zero eta."""
        eta = np.array([1e-300])
        link = InverseLink()
        result = link.inverse(eta)
        assert np.all(np.isfinite(result))


class TestGetLink:
    def test_get_link_log(self):
        link = _get_link("log")
        assert isinstance(link, LogLink)

    def test_get_link_identity(self):
        link = _get_link("identity")
        assert isinstance(link, IdentityLink)

    def test_get_link_inverse(self):
        link = _get_link("inverse")
        assert isinstance(link, InverseLink)

    def test_get_link_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown link"):
            _get_link("logit")

    def test_get_link_case_sensitive(self):
        with pytest.raises(ValueError, match="Unknown link"):
            _get_link("Log")


# ===========================================================================
# 2. Family repr, init_mu, non-default links
# ===========================================================================

class TestFamilyRepr:
    def test_gaussian_repr(self):
        r = repr(Gaussian())
        assert "Gaussian" in r
        assert "identity" in r

    def test_gamma_repr(self):
        r = repr(Gamma())
        assert "Gamma" in r
        assert "log" in r

    def test_inverse_gaussian_repr(self):
        r = repr(InverseGaussian())
        assert "InverseGaussian" in r

    def test_tweedie_repr(self):
        r = repr(Tweedie(p=1.7))
        assert "Tweedie" in r
        assert "1.7" in r

    def test_negative_binomial_repr(self):
        r = repr(NegativeBinomial(alpha=2.0))
        assert "NegativeBinomial" in r
        assert "2.0" in r

    def test_poisson_repr(self):
        r = repr(Poisson())
        assert "Poisson" in r


class TestFamilyInitMu:
    def test_gaussian_init_mu_is_mean(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu0 = Gaussian().init_mu(y)
        assert np.allclose(mu0, np.mean(y))

    def test_gamma_init_mu_is_mean(self):
        y = np.array([0.5, 1.5, 2.5])
        mu0 = Gamma().init_mu(y)
        assert np.allclose(mu0, np.mean(y))

    def test_tweedie_init_mu_uses_positive_mean(self):
        """Tweedie with zeros: init_mu should use mean of positive values."""
        y = np.array([0.0, 0.0, 2.0, 3.0])
        mu0 = Tweedie(p=1.5).init_mu(y)
        expected = np.mean(y[y > 0])
        assert np.allclose(mu0, expected)

    def test_tweedie_init_mu_all_zero_returns_one(self):
        """If all y=0, init_mu should return 1.0 (the fallback)."""
        y = np.array([0.0, 0.0, 0.0])
        mu0 = Tweedie(p=1.5).init_mu(y)
        assert np.allclose(mu0, 1.0)

    def test_poisson_init_mu_at_least_min(self):
        """Poisson init_mu clamps to a minimum > 0."""
        y = np.zeros(10)
        mu0 = Poisson().init_mu(y)
        assert np.all(mu0 > 0)


class TestNonDefaultLinks:
    def test_gaussian_with_log_link(self):
        """Gaussian accepts log link — unusual but valid."""
        family = Gaussian(link="log")
        mu = np.array([1.0, 2.0])
        eta = family.mu_to_eta(mu)
        mu_back = family.eta_to_mu(eta)
        assert np.allclose(mu, mu_back, atol=1e-10)

    def test_gamma_with_identity_link(self):
        """Gamma can be fitted with identity link."""
        family = Gamma(link="identity")
        mu = np.array([1.0, 2.0, 3.0])
        eta = family.mu_to_eta(mu)
        assert np.allclose(eta, mu)

    def test_inverse_gaussian_with_inverse_link(self):
        """InverseGaussian accepts its canonical inverse link."""
        family = InverseGaussian(link="inverse")
        mu = np.array([1.0, 2.0])
        eta = family.mu_to_eta(mu)
        mu_back = family.eta_to_mu(eta)
        assert np.allclose(mu, mu_back, atol=1e-8)


# ===========================================================================
# 3. Log-likelihood correctness for all families
# ===========================================================================

class TestLogLikelihoods:
    """Verify that each family's log-likelihood is maximised at y == mu."""

    def test_inverse_gaussian_ll_maximised_at_mu(self):
        rng = np.random.default_rng(101)
        y = rng.gamma(2, 1, 50)
        phi = np.ones(50) * 0.5
        family = InverseGaussian()
        ll_at_mu = np.sum(family.log_likelihood(y, y, phi))
        mu_wrong = y * rng.uniform(0.9, 1.1, 50)
        ll_wrong = np.sum(family.log_likelihood(y, mu_wrong, phi))
        assert ll_at_mu > ll_wrong

    def test_poisson_ll_finite(self):
        rng = np.random.default_rng(102)
        y = rng.poisson(3, 50).astype(float)
        mu = rng.uniform(1, 5, 50)
        phi = np.ones(50)
        ll = Poisson().log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))

    def test_negbin_ll_finite(self):
        rng = np.random.default_rng(103)
        y = rng.negative_binomial(2, 0.5, 50).astype(float)
        mu = rng.uniform(0.5, 5, 50)
        phi = np.ones(50)
        ll = NegativeBinomial(alpha=0.5).log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))

    def test_tweedie_ll_at_zero_response(self):
        """Zero responses are valid for Tweedie: log f should be finite."""
        y = np.zeros(10)
        mu = np.ones(10) * 2.0
        phi = np.ones(10) * 0.5
        ll = Tweedie(p=1.5).log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))

    def test_tweedie_ll_mixed_zeros_and_positives(self):
        y = np.array([0.0, 0.0, 1.5, 2.0, 3.0])
        mu = np.ones(5) * 1.5
        phi = np.ones(5) * 0.4
        ll = Tweedie(p=1.5).log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))

    def test_gaussian_ll_returns_per_obs_array(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        phi = np.ones(3) * 0.5
        ll = Gaussian().log_likelihood(y, mu, phi)
        assert ll.shape == (3,)

    def test_gamma_ll_zero_y_clamped_no_crash(self):
        """Gamma clips y to avoid log(0)."""
        y = np.array([0.0, 1.0, 2.0])
        mu = np.ones(3) * 1.0
        phi = np.ones(3) * 0.5
        ll = Gamma().log_likelihood(y, mu, phi)
        assert np.all(np.isfinite(ll))


class TestVarianceFunctions:
    def test_inverse_gaussian_variance(self):
        mu = np.array([1.0, 2.0, 3.0])
        V = InverseGaussian().variance(mu)
        assert np.allclose(V, mu ** 3)

    def test_tweedie_variance(self):
        mu = np.array([1.0, 2.0, 4.0])
        tw = Tweedie(p=1.5)
        V = tw.variance(mu)
        assert np.allclose(V, mu ** 1.5)

    def test_poisson_variance_equals_mu(self):
        mu = np.array([0.5, 1.0, 3.0])
        V = Poisson().variance(mu)
        assert np.allclose(V, mu)

    def test_negbin_variance_greater_than_poisson(self):
        mu = np.array([1.0, 2.0, 3.0])
        V_pois = mu
        V_nb = NegativeBinomial(alpha=1.0).variance(mu)
        assert np.all(V_nb > V_pois)

    def test_negbin_variance_depends_on_alpha(self):
        mu = np.array([2.0])
        V_low = NegativeBinomial(alpha=0.1).variance(mu)
        V_high = NegativeBinomial(alpha=5.0).variance(mu)
        assert V_high[0] > V_low[0]


# ===========================================================================
# 4. Deviance at zero for count families
# ===========================================================================

class TestDevianceAtZero:
    def test_poisson_deviance_at_zero(self):
        """Poisson unit deviance for y=0 should be 2*mu."""
        mu = np.array([1.0, 2.0])
        y = np.zeros(2)
        d = Poisson().deviance_resid(y, mu)
        assert np.allclose(d, 2.0 * mu)

    def test_negbin_deviance_at_zero_is_nonneg(self):
        y = np.zeros(10)
        mu = np.ones(10) * 2.0
        d = NegativeBinomial(alpha=0.5).deviance_resid(y, mu)
        assert np.all(d >= -1e-12)

    def test_negbin_deviance_at_mu_is_zero(self):
        """Unit deviance at y==mu should be (near) zero."""
        rng = np.random.default_rng(104)
        y = rng.uniform(0.5, 5, 50)
        nb = NegativeBinomial(alpha=1.0)
        d = nb.deviance_resid(y, y)
        assert np.allclose(d, 0.0, atol=1e-10)


# ===========================================================================
# 5. Diagnostics for non-Gamma families
# ===========================================================================

def _fit_model(family, df, formula="y ~ x", dformula="~ 1"):
    return DGLM(formula=formula, dformula=dformula, family=family, data=df).fit()


class TestInverseGaussianDiagnostics:
    def setup_method(self):
        self.df = make_ig_df()
        self.result = _fit_model(InverseGaussian(), self.df)

    def test_quantile_residuals_shape(self):
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (len(self.df),)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_pearson_residuals_finite(self):
        r = diagnostics.pearson_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_deviance_residuals_finite(self):
        r = diagnostics.deviance_residuals(self.result)
        assert np.all(np.isfinite(r))


class TestPoissonDiagnostics:
    """Poisson uses randomised quantile residuals."""

    def setup_method(self):
        self.df = make_poisson_df()
        self.result = _fit_model(Poisson(), self.df)

    def test_quantile_residuals_shape(self):
        r = diagnostics.quantile_residuals(self.result)
        assert r.shape == (len(self.df),)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))

    def test_pearson_residuals_finite(self):
        r = diagnostics.pearson_residuals(self.result)
        assert np.all(np.isfinite(r))


class TestNegBinDiagnostics:
    """NegBin falls through to deviance residuals in quantile_residuals."""

    def setup_method(self):
        rng = np.random.default_rng(105)
        n = 200
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.4 * x)
        y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
        self.df = pd.DataFrame({"y": y, "x": x})
        self.result = _fit_model(NegativeBinomial(alpha=0.5), self.df)

    def test_quantile_residuals_falls_back_to_deviance(self):
        """NegBin is not in the special-cased families, falls back."""
        r = diagnostics.quantile_residuals(self.result)
        r_dev = diagnostics.deviance_residuals(self.result)
        assert np.allclose(r, r_dev, atol=1e-10)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))


class TestTweedieDiagnostics:
    """Tweedie also falls through to deviance residuals."""

    def setup_method(self):
        rng = np.random.default_rng(106)
        n = 200
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.3 * x)
        phi = 0.5
        # Generate via compound Poisson-Gamma
        lam = mu ** (2 - 1.5) / (phi * (2 - 1.5))
        alpha_g = (2 - 1.5) / (1.5 - 1)
        beta_g = phi * (1.5 - 1) * mu ** (1.5 - 1)
        n_events = rng.poisson(lam)
        y = np.array([
            rng.gamma(alpha_g, beta_g[i], n_events[i]).sum()
            if n_events[i] > 0 else 0.0
            for i in range(n)
        ])
        self.df = pd.DataFrame({"y": y, "x": x})
        self.result = _fit_model(Tweedie(p=1.5), self.df)

    def test_quantile_residuals_falls_back_to_deviance(self):
        r = diagnostics.quantile_residuals(self.result)
        r_dev = diagnostics.deviance_residuals(self.result)
        assert np.allclose(r, r_dev, atol=1e-10)

    def test_quantile_residuals_finite(self):
        r = diagnostics.quantile_residuals(self.result)
        assert np.all(np.isfinite(r))


# ===========================================================================
# 6. results.py — SubmodelResult, DGLMResult
# ===========================================================================

class TestSubmodelResult:
    def setup_method(self):
        df = make_gamma_df()
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df
        ).fit()

    def test_summary_frame_columns(self):
        sf = self.result.mean_model.summary_frame()
        for col in ["coef", "exp_coef", "se", "z", "p_value"]:
            assert col in sf.columns

    def test_summary_frame_index(self):
        """Index should match the coefficient names."""
        sf = self.result.mean_model.summary_frame()
        assert len(sf) == len(self.result.mean_model.coef_names)

    def test_se_positive(self):
        sf = self.result.mean_model.summary_frame()
        assert (sf["se"] >= 0).all()

    def test_p_values_between_zero_and_one(self):
        sf = self.result.mean_model.summary_frame()
        assert (sf["p_value"] >= 0).all()
        assert (sf["p_value"] <= 1).all()

    def test_exp_coef_positive(self):
        sf = self.result.mean_model.summary_frame()
        assert (sf["exp_coef"] > 0).all()

    def test_dispersion_summary_frame_has_intercept(self):
        sf = self.result.dispersion_model.summary_frame()
        assert any("Intercept" in name for name in sf.index)


class TestDGLMResultSummary:
    def setup_method(self):
        df = make_gamma_df()
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df
        ).fit()

    def test_summary_is_string(self):
        s = self.result.summary()
        assert isinstance(s, str)

    def test_summary_contains_family(self):
        s = self.result.summary()
        assert "Gamma" in s

    def test_summary_contains_aic_bic(self):
        s = self.result.summary()
        assert "AIC" in s
        assert "BIC" in s

    def test_summary_contains_loglik(self):
        s = self.result.summary()
        assert "Log-lik" in s

    def test_summary_contains_observations(self):
        s = self.result.summary()
        assert "Observations" in s or "bservation" in s

    def test_summary_contains_phi_range(self):
        s = self.result.summary()
        assert "phi" in s.lower()

    def test_repr_contains_family(self):
        r = repr(self.result)
        assert "Gamma" in r

    def test_repr_contains_n_obs(self):
        r = repr(self.result)
        assert "n=" in r

    def test_repr_contains_loglik(self):
        r = repr(self.result)
        assert "loglik=" in r


class TestOverdispersionTestREMLWarning:
    """overdispersion_test() should warn when called on a REML-fitted model."""

    def setup_method(self):
        df = make_gamma_df()
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df, method="reml"
        ).fit()

    def test_reml_issues_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.result.overdispersion_test()
            reml_warnings = [x for x in w if issubclass(x.category, UserWarning)
                             and "reml" in str(x.message).lower()]
            assert len(reml_warnings) >= 1, "Expected UserWarning about REML"

    def test_returns_dict_with_correct_keys(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.result.overdispersion_test()
        assert "statistic" in result
        assert "df" in result
        assert "p_value" in result
        assert "conclusion" in result

    def test_statistic_nonneg(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.result.overdispersion_test()
        assert result["statistic"] >= 0.0

    def test_p_value_between_zero_and_one(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.result.overdispersion_test()
        assert 0.0 <= result["p_value"] <= 1.0


class TestOverdispersionTestML:
    """ML method should not warn and LRT should reject when phi varies."""

    def setup_method(self):
        df = make_gamma_df(n=800)
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df, method="ml"
        ).fit()

    def test_no_reml_warning_with_ml(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.result.overdispersion_test()
            reml_warnings = [x for x in w if issubclass(x.category, UserWarning)
                             and "reml" in str(x.message).lower()]
            assert len(reml_warnings) == 0

    def test_conclusion_is_string(self):
        result = self.result.overdispersion_test()
        assert isinstance(result["conclusion"], str)


class TestPredictMissingExposure:
    """predict() on newdata missing the exposure column should warn."""

    def setup_method(self):
        rng = np.random.default_rng(107)
        n = 200
        x = rng.uniform(0, 1, n)
        exp = rng.uniform(0.5, 2.0, n)
        mu = exp * np.exp(1.0 + 0.5 * x)
        phi = 0.4 * np.ones(n)
        y = rng.gamma(1.0 / phi, mu * phi)
        self.df = pd.DataFrame({"y": y, "x": x, "exposure": exp})
        self.result = DGLM(
            formula="y ~ x", dformula="~ 1",
            family=Gamma(), data=self.df, exposure="exposure"
        ).fit()

    def test_missing_exposure_warns(self):
        newdata = pd.DataFrame({"x": [0.3, 0.5]})  # no 'exposure' column
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pred = self.result.predict(newdata)
            exposure_warnings = [x for x in w if issubclass(x.category, UserWarning)
                                  and "exposure" in str(x.message).lower()]
            assert len(exposure_warnings) >= 1

    def test_missing_exposure_still_returns_predictions(self):
        newdata = pd.DataFrame({"x": [0.3, 0.5]})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pred = self.result.predict(newdata)
        assert len(pred) == 2
        assert np.all(pred > 0)


class TestPredictWhich:
    def setup_method(self):
        df = make_gamma_df()
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df
        ).fit()
        self.newdata = df.head(5)

    def test_predict_mean_positive(self):
        pred = self.result.predict(self.newdata, which="mean")
        assert np.all(pred > 0)

    def test_predict_dispersion_positive(self):
        pred = self.result.predict(self.newdata, which="dispersion")
        assert np.all(pred > 0)

    def test_predict_variance_positive(self):
        pred = self.result.predict(self.newdata, which="variance")
        assert np.all(pred > 0)

    def test_predict_invalid_which_raises(self):
        with pytest.raises(ValueError, match="which must be"):
            self.result.predict(self.newdata, which="quantile")


# ===========================================================================
# 7. fitting.py — internal functions and edge cases
# ===========================================================================

class TestWLS:
    def test_exact_recovery_with_identity_matrix(self):
        """WLS with X=I and unit weights: beta = z."""
        n = 5
        X = np.eye(n)
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(n)
        beta = _wls(X, z, w)
        assert np.allclose(beta, z, atol=1e-10)

    def test_weighted_regression_standard_result(self):
        """Single-variable WLS: beta should match numpy's closed form."""
        rng = np.random.default_rng(200)
        n = 50
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        true_beta = np.array([2.0, 1.5])
        z = X @ true_beta + rng.normal(0, 0.05, n)
        w = np.ones(n)
        beta = _wls(X, z, w)
        assert np.allclose(beta, true_beta, atol=0.2)


class TestHatDiagonal:
    def test_sum_equals_rank(self):
        """Sum of hat diagonal = rank of X (number of columns)."""
        rng = np.random.default_rng(201)
        n, p = 100, 3
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, (n, p - 1))])
        w = np.ones(n)
        h = _hat_diagonal(X, w)
        # sum(h) = trace(H) = p for OLS
        assert abs(np.sum(h) - p) < 1e-6

    def test_hat_diagonal_between_0_and_1(self):
        rng = np.random.default_rng(202)
        n, p = 80, 4
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, (n, p - 1))])
        w = np.ones(n)
        h = _hat_diagonal(X, w)
        assert np.all(h >= -1e-10)
        assert np.all(h <= 1.0 + 1e-10)


class TestSandwichVcov:
    def test_positive_diagonal_for_full_rank(self):
        rng = np.random.default_rng(203)
        n, p = 100, 3
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, (n, p - 1))])
        w = np.ones(n)
        vcov = _sandwich_vcov(X, w)
        assert np.all(np.diag(vcov) > 0)

    def test_vcov_shape(self):
        rng = np.random.default_rng(204)
        n, p = 80, 4
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, (n, p - 1))])
        w = np.ones(n)
        vcov = _sandwich_vcov(X, w)
        assert vcov.shape == (p, p)

    def test_vcov_symmetric(self):
        rng = np.random.default_rng(205)
        n, p = 60, 3
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, (n, p - 1))])
        w = np.ones(n)
        vcov = _sandwich_vcov(X, w)
        assert np.allclose(vcov, vcov.T, atol=1e-10)

    def test_near_singular_does_not_crash(self):
        """Nearly singular X should not raise — uses pinv fallback."""
        n = 50
        x = np.ones(n)  # rank-1 design
        X = np.column_stack([x, x])  # rank-deficient
        w = np.ones(n)
        # Should not raise
        vcov = _sandwich_vcov(X, w)
        assert vcov.shape == (2, 2)


class TestDglmFitVerbose:
    def test_verbose_produces_output(self, capsys):
        rng = np.random.default_rng(206)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.ones((n, 1))
        y = rng.gamma(2, 1, n)
        result = dglm_fit(Gamma(), X, Z, y, verbose=True, maxit=3)
        captured = capsys.readouterr()
        assert "iter" in captured.out.lower() or "loglik" in captured.out.lower()

    def test_verbose_false_no_output(self, capsys):
        rng = np.random.default_rng(207)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.ones((n, 1))
        y = rng.gamma(2, 1, n)
        dglm_fit(Gamma(), X, Z, y, verbose=False, maxit=5)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestDglmFitNonConvergence:
    def test_warns_when_no_convergence(self):
        """With maxit=1 the model should not converge and should warn."""
        rng = np.random.default_rng(208)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = rng.gamma(2, 1, n)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dglm_fit(Gamma(), X, Z, y, maxit=1)
            convergence_warnings = [
                x for x in w
                if issubclass(x.category, RuntimeWarning)
                and "converge" in str(x.message).lower()
            ]
            assert len(convergence_warnings) >= 1
        assert not result.converged

    def test_result_still_has_valid_mu_phi(self):
        rng = np.random.default_rng(209)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.ones((n, 1))
        y = rng.gamma(2, 1, n)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = dglm_fit(Gamma(), X, Z, y, maxit=1)
        assert np.all(result.mu > 0)
        assert np.all(result.phi > 0)


class TestDglmFitInvalidMethod:
    def test_invalid_method_raises(self):
        rng = np.random.default_rng(210)
        n = 50
        X = np.ones((n, 1))
        Z = np.ones((n, 1))
        y = rng.gamma(2, 1, n)
        with pytest.raises(ValueError, match="method"):
            dglm_fit(Gamma(), X, Z, y, method="bayes")


class TestDglmFitWithPriorWeights:
    def test_prior_weights_accepted(self):
        rng = np.random.default_rng(211)
        n = 100
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.ones((n, 1))
        y = rng.gamma(2, 1, n)
        w = rng.uniform(0.5, 2.0, n)
        result = dglm_fit(Gamma(), X, Z, y, prior_weights=w, maxit=10)
        assert np.all(result.mu > 0)
        assert np.all(result.phi > 0)


class TestDglmFitWithLogOffset:
    def test_log_offset_accepted(self):
        rng = np.random.default_rng(212)
        n = 150
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        Z = np.ones((n, 1))
        exposure = rng.uniform(0.5, 2.0, n)
        mu_base = np.exp(1.0 + 0.5 * X[:, 1]) * exposure
        y = rng.gamma(2.0, mu_base / 2.0)
        offset = np.log(exposure)
        result = dglm_fit(Gamma(), X, Z, y, log_offset=offset, maxit=10)
        assert np.all(result.mu > 0)


# ===========================================================================
# 8. AIC / BIC formulas
# ===========================================================================

class TestAICBIC:
    def setup_method(self):
        df = make_gamma_df(n=500)
        self.result = DGLM(
            formula="y ~ x", dformula="~ z",
            family=Gamma(), data=df
        ).fit()

    def test_aic_formula(self):
        """AIC = -2 * loglik + 2 * n_params"""
        p_mean = len(self.result.mean_model.coef)
        p_disp = len(self.result.dispersion_model.coef)
        k = p_mean + p_disp
        expected_aic = -2.0 * self.result.loglik + 2.0 * k
        assert abs(self.result.aic - expected_aic) < 1e-6

    def test_bic_formula(self):
        """BIC = -2 * loglik + log(n) * n_params"""
        p_mean = len(self.result.mean_model.coef)
        p_disp = len(self.result.dispersion_model.coef)
        k = p_mean + p_disp
        expected_bic = -2.0 * self.result.loglik + np.log(self.result.n_obs) * k
        assert abs(self.result.bic - expected_bic) < 1e-6

    def test_bic_ge_aic_for_large_n(self):
        """For n > 7, log(n) > 2, so BIC > AIC always."""
        assert self.result.bic > self.result.aic


# ===========================================================================
# 9. DGLM with all supported families end-to-end
# ===========================================================================

class TestEndToEndAllFamilies:
    """Smoke tests: DGLM fits without crash for all supported families."""

    def test_gamma_intercept_only(self):
        df = make_gamma_df()
        result = DGLM("y ~ 1", "~ 1", family=Gamma(), data=df).fit()
        assert result.converged

    def test_gaussian_intercept_only(self):
        df = make_gaussian_df()
        result = DGLM("y ~ 1", "~ 1", family=Gaussian(link="identity"), data=df).fit()
        assert result.converged

    def test_inverse_gaussian_fits(self):
        df = make_ig_df()
        result = DGLM("y ~ x", "~ 1", family=InverseGaussian(), data=df).fit()
        assert np.all(result.mu_ > 0)

    def test_poisson_fits(self):
        df = make_poisson_df()
        result = DGLM("y ~ x", "~ 1", family=Poisson(), data=df).fit()
        assert np.all(result.mu_ > 0)

    def test_negbin_fits(self):
        rng = np.random.default_rng(213)
        n = 200
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.4 * x)
        y = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
        df = pd.DataFrame({"y": y, "x": x})
        result = DGLM("y ~ x", "~ 1", family=NegativeBinomial(alpha=0.5), data=df).fit()
        assert np.all(result.mu_ > 0)

    def test_tweedie_with_zeros_fits(self):
        rng = np.random.default_rng(214)
        n = 200
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.3 * x)
        phi = 0.5
        # Compound Poisson-Gamma
        lam = mu ** 0.5 / (phi * 0.5)
        alpha_g = 0.5 / 0.5
        beta_g = phi * 0.5 * mu ** 0.5
        n_events = rng.poisson(lam)
        y = np.array([
            rng.gamma(alpha_g, beta_g[i], n_events[i]).sum()
            if n_events[i] > 0 else 0.0
            for i in range(n)
        ])
        df = pd.DataFrame({"y": y, "x": x})
        result = DGLM("y ~ x", "~ 1", family=Tweedie(p=1.5), data=df).fit()
        assert np.all(result.mu_ > 0)
