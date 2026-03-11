"""
Tests for the exponential family implementations.

Each family is tested for:
  - deviance_resid is zero when y == mu
  - deviance_resid is non-negative
  - log_likelihood is maximised at y == mu (for continuous families)
  - variance function is correct
"""

import numpy as np
import pytest

from insurance_dispersion.families import (
    Gaussian,
    Gamma,
    InverseGaussian,
    Tweedie,
    Poisson,
    NegativeBinomial,
)

RNG = np.random.default_rng(42)


def test_gaussian_deviance_at_mu():
    y = RNG.uniform(1, 10, 100)
    mu = y.copy()
    d = Gaussian().deviance_resid(y, mu)
    assert np.allclose(d, 0.0)


def test_gaussian_deviance_nonneg():
    y = RNG.uniform(1, 10, 100)
    mu = RNG.uniform(1, 10, 100)
    d = Gaussian().deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_gaussian_variance():
    mu = RNG.uniform(1, 5, 50)
    V = Gaussian().variance(mu)
    assert np.allclose(V, 1.0)


def test_gamma_deviance_at_mu():
    y = RNG.gamma(2, 1, 100)
    mu = y.copy()
    d = Gamma().deviance_resid(y, mu)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_gamma_deviance_nonneg():
    y = RNG.gamma(2, 1, 100)
    mu = RNG.gamma(2, 1, 100)
    d = Gamma().deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_gamma_variance():
    mu = RNG.uniform(1, 5, 50)
    V = Gamma().variance(mu)
    assert np.allclose(V, mu ** 2)


def test_inverse_gaussian_deviance_at_mu():
    y = RNG.gamma(2, 1, 100)
    mu = y.copy()
    d = InverseGaussian().deviance_resid(y, mu)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_inverse_gaussian_deviance_nonneg():
    y = RNG.gamma(2, 1, 100)
    mu = RNG.gamma(2, 1, 100)
    d = InverseGaussian().deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_tweedie_deviance_at_mu():
    y = RNG.gamma(2, 1, 100)
    mu = y.copy()
    tweedie = Tweedie(p=1.5)
    d = tweedie.deviance_resid(y, mu)
    assert np.allclose(d, 0.0, atol=1e-10)


def test_tweedie_deviance_nonneg():
    y = RNG.gamma(2, 1, 100)
    mu = RNG.gamma(2, 1, 100)
    tweedie = Tweedie(p=1.5)
    d = tweedie.deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_tweedie_p_validation():
    with pytest.raises(ValueError, match="p must be in"):
        Tweedie(p=2.5)
    with pytest.raises(ValueError, match="p must be in"):
        Tweedie(p=0.5)


def test_poisson_deviance_nonneg():
    y = RNG.poisson(3, 100).astype(float)
    mu = RNG.uniform(0.5, 5, 100)
    d = Poisson().deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_negative_binomial_deviance_nonneg():
    y = RNG.negative_binomial(2, 0.5, 100).astype(float)
    mu = RNG.uniform(0.5, 5, 100)
    nb = NegativeBinomial(alpha=0.5)
    d = nb.deviance_resid(y, mu)
    assert np.all(d >= -1e-12)


def test_negative_binomial_alpha_validation():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        NegativeBinomial(alpha=0.0)


def test_log_likelihood_maximised_at_mu_gaussian():
    y = RNG.normal(3, 1, 100)
    phi = np.ones(100) * 0.5
    family = Gaussian()
    ll_at_mu = np.sum(family.log_likelihood(y, y, phi))
    mu_wrong = y + RNG.normal(0, 0.1, 100)
    ll_wrong = np.sum(family.log_likelihood(y, mu_wrong, phi))
    assert ll_at_mu > ll_wrong


def test_log_likelihood_maximised_at_mu_gamma():
    y = RNG.gamma(2, 1, 100)
    phi = np.ones(100) * 0.5
    family = Gamma()
    ll_at_mu = np.sum(family.log_likelihood(y, y, phi))
    mu_wrong = y * RNG.uniform(0.9, 1.1, 100)
    ll_wrong = np.sum(family.log_likelihood(y, mu_wrong, phi))
    assert ll_at_mu > ll_wrong


def test_links_round_trip():
    """Link and inverse_link should be exact inverses."""
    from insurance_dispersion.families import LogLink, IdentityLink, InverseLink

    mu = np.array([0.5, 1.0, 2.0, 5.0])

    for LinkCls in [LogLink, IdentityLink, InverseLink]:
        link = LinkCls()
        eta = link.link(mu)
        mu_back = link.inverse(eta)
        assert np.allclose(mu, mu_back, atol=1e-10), f"Round-trip failed for {LinkCls.__name__}"
