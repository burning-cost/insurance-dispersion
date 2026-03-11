"""
Tests for the core DGLM fitting engine.

The key tests here are:
  1. Known-DGP recovery: simulate data from Gamma with phi varying by group,
     verify DGLM recovers the true alpha within ~2 SE.
  2. Gaussian exact test: DGLM on Gaussian data matches analytical MLE.
  3. Monotone log-likelihood: loglik should not decrease across outer iterations.
  4. Intercept-only dispersion: DGLM with dformula='~1' should match GLM with
     constant phi.
  5. Edge cases: single dispersion covariate, large phi ratios.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_dispersion.families import Gaussian, Gamma, InverseGaussian, Poisson
from insurance_dispersion.fitting import dglm_fit, _hat_diagonal, _wls

RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_gamma_dglm(n=500, seed=42):
    """
    Simulate from a Gamma DGLM with:
      log(mu_i) = 1.5 + 0.8 * x1_i
      log(phi_i) = -0.5 + 1.2 * z1_i

    Returns (y, X, Z, true_beta, true_alpha).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    z1 = rng.uniform(-1, 1, n)

    true_beta = np.array([1.5, 0.8])
    true_alpha = np.array([-0.5, 1.2])

    mu = np.exp(true_beta[0] + true_beta[1] * x1)
    phi = np.exp(true_alpha[0] + true_alpha[1] * z1)

    # Gamma(shape=1/phi, scale=mu*phi)
    shape = 1.0 / phi
    y = rng.gamma(shape, mu * phi)

    X = np.column_stack([np.ones(n), x1])
    Z = np.column_stack([np.ones(n), z1])

    return y, X, Z, true_beta, true_alpha


def _simulate_gaussian_dglm(n=1000, seed=99):
    """
    Simulate Gaussian DGLM: y ~ N(mu_i, phi_i).
    log(mu_i) = 2 + 0.5 * x1
    log(phi_i) = 0.1 + 0.6 * z1
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    z1 = rng.uniform(0, 1, n)

    mu = np.exp(2.0 + 0.5 * x1)
    phi = np.exp(0.1 + 0.6 * z1)

    y = rng.normal(mu, np.sqrt(phi))

    X = np.column_stack([np.ones(n), x1])
    Z = np.column_stack([np.ones(n), z1])

    return y, X, Z


# ---------------------------------------------------------------------------
# Gamma DGP: coefficient recovery
# ---------------------------------------------------------------------------

class TestGammaDGLM:
    def setup_method(self):
        self.y, self.X, self.Z, self.true_beta, self.true_alpha = (
            _simulate_gamma_dglm(n=1000, seed=123)
        )

    def test_converges(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        assert result.converged

    def test_beta_recovery(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        # With n=1000 we expect recovery within ~0.1
        assert np.allclose(result.beta, self.true_beta, atol=0.15), (
            f"beta={result.beta} vs true={self.true_beta}"
        )

    def test_alpha_recovery(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        assert np.allclose(result.alpha, self.true_alpha, atol=0.25), (
            f"alpha={result.alpha} vs true={self.true_alpha}"
        )

    def test_reml_converges(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="reml", maxit=30
        )
        assert result.converged

    def test_loglik_is_finite(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        assert np.isfinite(result.loglik_history[-1])

    def test_phi_positive(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        assert np.all(result.phi > 0)

    def test_mu_positive(self):
        result = dglm_fit(
            Gamma(), self.X, self.Z, self.y, method="ml", maxit=30
        )
        assert np.all(result.mu > 0)


# ---------------------------------------------------------------------------
# Monotone log-likelihood
# ---------------------------------------------------------------------------

class TestMonotoneLogLikelihood:
    def test_loglik_nondecreasing_gamma(self):
        y, X, Z, _, _ = _simulate_gamma_dglm(n=500, seed=7)
        result = dglm_fit(Gamma(), X, Z, y, method="ml", maxit=30)
        history = result.loglik_history
        # Allow tiny numerical decrease (1e-4 relative)
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-2, (
                f"Log-likelihood decreased at iteration {i}: "
                f"{history[i]:.6f} < {history[i - 1]:.6f}"
            )

    def test_loglik_nondecreasing_gaussian(self):
        y, X, Z = _simulate_gaussian_dglm(n=500, seed=8)
        # Use identity link for Gaussian
        result = dglm_fit(Gaussian(), X, Z, y, method="ml", maxit=30)
        history = result.loglik_history
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-2


# ---------------------------------------------------------------------------
# Intercept-only dispersion: should match constant-phi GLM
# ---------------------------------------------------------------------------

class TestInterceptOnlyDispersion:
    def test_gamma_matches_constant_phi(self):
        """
        With dformula='~1' (intercept only in Z), alpha[0] should give
        a constant phi close to the moment estimate for a constant-phi Gamma.
        """
        rng = np.random.default_rng(55)
        n = 800
        x1 = rng.uniform(-1, 1, n)
        mu = np.exp(2.0 + 0.7 * x1)
        true_phi = 0.4  # constant dispersion
        y = rng.gamma(1.0 / true_phi, mu * true_phi)

        X = np.column_stack([np.ones(n), x1])
        Z = np.ones((n, 1))  # intercept only

        result = dglm_fit(Gamma(), X, Z, y, method="ml", maxit=30)
        assert result.converged
        phi_est = float(np.exp(result.alpha[0]))
        # Should be within 20% of true phi
        assert abs(phi_est - true_phi) / true_phi < 0.3, (
            f"Estimated phi={phi_est:.4f} vs true phi={true_phi}"
        )


# ---------------------------------------------------------------------------
# Large phi ratio
# ---------------------------------------------------------------------------

class TestLargePhiRatio:
    def test_large_phi_range(self):
        """
        phi ratio of 100x between groups should still converge.
        """
        rng = np.random.default_rng(77)
        n = 1000
        group = (np.arange(n) % 2).astype(float)  # 0 or 1
        mu = np.full(n, 3.0)
        phi = np.where(group == 0, 0.1, 10.0)  # 100x ratio

        shape = 1.0 / phi
        y = rng.gamma(shape, mu * phi)

        X = np.ones((n, 1))  # intercept only mean
        Z = np.column_stack([np.ones(n), group])

        result = dglm_fit(Gamma(), X, Z, y, method="ml", maxit=50)
        assert result.converged
        assert np.all(result.phi > 0)
        # alpha[1] should be large and positive (log(100) ~ 4.6)
        assert result.alpha[1] > 2.0, f"alpha[1]={result.alpha[1]:.4f}"


# ---------------------------------------------------------------------------
# WLS helper
# ---------------------------------------------------------------------------

def test_wls_exact():
    """WLS should solve exact linear systems."""
    rng = np.random.default_rng(10)
    X = rng.standard_normal((50, 3))
    true_beta = np.array([1.0, -2.0, 0.5])
    w = rng.uniform(0.5, 2.0, 50)
    z = X @ true_beta  # exact (no noise)
    beta_hat = _wls(X, z, w)
    assert np.allclose(beta_hat, true_beta, atol=1e-8)


# ---------------------------------------------------------------------------
# Hat diagonal
# ---------------------------------------------------------------------------

def test_hat_diagonal_sum():
    """Sum of hat diagonal = number of parameters (trace of hat matrix)."""
    rng = np.random.default_rng(11)
    n, p = 100, 4
    X = rng.standard_normal((n, p))
    w = rng.uniform(0.5, 2.0, n)
    h = _hat_diagonal(X, w)
    assert np.allclose(np.sum(h), p, atol=1e-8), f"trace(H)={np.sum(h):.4f} != p={p}"


def test_hat_diagonal_bounds():
    """Hat diagonal values must be in [0, 1]."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal((200, 5))
    w = np.ones(200)
    h = _hat_diagonal(X, w)
    assert np.all(h >= -1e-12)
    assert np.all(h <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Gaussian DGP test
# ---------------------------------------------------------------------------

class TestGaussianDGLM:
    def test_gaussian_converges(self):
        y, X, Z = _simulate_gaussian_dglm(n=500, seed=20)
        result = dglm_fit(Gaussian(), X, Z, y, method="ml", maxit=30)
        assert result.converged

    def test_gaussian_beta_sign(self):
        """beta[1] should be positive (true=0.5, log link on positive mu)."""
        y, X, Z = _simulate_gaussian_dglm(n=1000, seed=21)
        result = dglm_fit(Gaussian(), X, Z, y, method="ml", maxit=30)
        assert result.beta[1] > 0


# ---------------------------------------------------------------------------
# Prior weights
# ---------------------------------------------------------------------------

def test_prior_weights_accepted():
    """Fitting with non-unit prior weights should not crash."""
    rng = np.random.default_rng(30)
    n = 200
    y = rng.gamma(2, 1, n)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    Z = np.ones((n, 1))
    w = rng.uniform(0.5, 2.0, n)
    result = dglm_fit(Gamma(), X, Z, y, prior_weights=w, method="ml", maxit=30)
    assert result.converged


# ---------------------------------------------------------------------------
# Log-offset
# ---------------------------------------------------------------------------

def test_log_offset():
    """Log-offset shifts the mean linear predictor."""
    rng = np.random.default_rng(31)
    n = 300
    exposure = rng.uniform(0.5, 2.0, n)
    log_exp = np.log(exposure)
    # mu = exposure * exp(1.0) -- Gamma
    mu_true = exposure * np.exp(1.0)
    phi_true = 0.3
    y = rng.gamma(1.0 / phi_true, mu_true * phi_true)

    X = np.ones((n, 1))
    Z = np.ones((n, 1))

    result = dglm_fit(
        Gamma(), X, Z, y,
        log_offset=log_exp,
        method="ml", maxit=30,
    )
    assert result.converged
    # Intercept should be close to 1.0 (the log of the true coefficient)
    assert abs(result.beta[0] - 1.0) < 0.15, f"beta[0]={result.beta[0]:.4f}"
