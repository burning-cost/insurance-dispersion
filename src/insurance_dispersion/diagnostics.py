"""
Diagnostics for the Double GLM.

Residuals, QQ plot data, and dispersion diagnostic summaries.
These return data (arrays, DataFrames) rather than producing plots directly
so the library has no mandatory matplotlib dependency.

For plots, pass the returned data to matplotlib or any other visualisation
library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats

if TYPE_CHECKING:
    from insurance_dispersion.results import DGLMResult


def pearson_residuals(result: "DGLMResult") -> np.ndarray:
    """
    Pearson residuals adjusted for observation-level phi_i.

    r_i = (y_i - mu_i) / sqrt(phi_i * V(mu_i))

    Under the model these should be approximately N(0, 1). Systematic
    patterns against fitted values indicate mean model misspecification.
    """
    mu = result.mu_
    phi = result.phi_
    y = result._dglm._data[result._dglm.formula.split("~")[0].strip()].to_numpy(
        dtype=float
    )
    V = result._dglm.family.variance(mu)
    return (y - mu) / np.sqrt(np.clip(phi * V, 1e-300, None))


def deviance_residuals(result: "DGLMResult") -> np.ndarray:
    """
    Deviance residuals adjusted for phi_i.

    r_i = sign(y_i - mu_i) * sqrt(d_i / phi_i)

    where d_i is the unit deviance. These are the square roots of the
    dispersion pseudo-responses (before REML correction).
    """
    mu = result.mu_
    phi = result.phi_
    y = _get_y(result)
    d = result._dglm.family.deviance_resid(y, mu)
    return np.sign(y - mu) * np.sqrt(np.clip(d / np.clip(phi, 1e-300, None), 0, None))


def quantile_residuals(result: "DGLMResult") -> np.ndarray:
    """
    Randomised quantile residuals (Dunn & Smyth 1996).

    For continuous families these are exactly N(0, 1) under the true model.
    For discrete families a randomisation step is applied. Returns approximate
    standard normal residuals suitable for QQ plots.

    For continuous families (Gaussian, Gamma, InverseGaussian, Tweedie):
      q_i = Phi^{-1}(F(y_i; mu_i, phi_i))
    where F is the CDF and Phi is the standard normal CDF.
    """
    from insurance_dispersion import families as fam

    mu = result.mu_
    phi = result.phi_
    y = _get_y(result)
    family = result._dglm.family

    if isinstance(family, fam.Gaussian):
        # CDF = Phi((y - mu) / sqrt(phi))
        std = np.sqrt(np.clip(phi, 1e-300, None))
        p = scipy.stats.norm.cdf(y, loc=mu, scale=std)

    elif isinstance(family, fam.Gamma):
        # Gamma(shape=1/phi, scale=mu*phi)
        shape = 1.0 / np.clip(phi, 1e-300, None)
        scale = mu * phi
        p = scipy.stats.gamma.cdf(y, a=shape, scale=scale)

    elif isinstance(family, fam.InverseGaussian):
        # Inverse-Gaussian CDF via scipy
        mu_ig = mu
        lam = mu_ig ** 2 / np.clip(phi * mu_ig ** 3, 1e-300, None)
        # scipy uses mu, lam parametrisation
        p = scipy.stats.invgauss.cdf(y, mu=mu_ig / lam, scale=lam)

    elif isinstance(family, fam.Poisson):
        # Discrete: randomise between F(y-1) and F(y)
        shape = 1.0 / np.clip(phi, 1e-300, None)
        # Quasi-Poisson: use scaled Poisson approx
        p_lo = scipy.stats.poisson.cdf(np.maximum(y - 1, 0), mu=mu)
        p_hi = scipy.stats.poisson.cdf(y, mu=mu)
        u = np.random.uniform(size=len(y))
        p = p_lo + u * (p_hi - p_lo)

    else:
        # Fallback: use deviance residuals
        return deviance_residuals(result)

    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return scipy.stats.norm.ppf(p)


def qq_plot_data(result: "DGLMResult") -> pd.DataFrame:
    """
    Data for a QQ plot of quantile residuals against N(0,1).

    Returns DataFrame with columns: theoretical, observed, sorted residuals.
    Pass to matplotlib.pyplot.scatter or similar.
    """
    resid = quantile_residuals(result)
    n = len(resid)
    resid_sorted = np.sort(resid)
    theoretical = scipy.stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    return pd.DataFrame(
        {"theoretical": theoretical, "observed": resid_sorted}
    )


def dispersion_diagnostic(result: "DGLMResult") -> pd.DataFrame:
    """
    Data for the dispersion diagnostic plot.

    Returns DataFrame with:
      - fitted_mu: predicted mean
      - fitted_phi: predicted dispersion
      - unit_deviance: d_i = deviance_resid(y_i, mu_i)
      - scaled_deviance: d_i / phi_i (the pseudo-response; should be ~ Gamma(1/2, 2))

    Use to check whether the dispersion submodel captures the pattern in
    unit deviances.
    """
    mu = result.mu_
    phi = result.phi_
    y = _get_y(result)
    family = result._dglm.family
    d = family.deviance_resid(y, mu)

    return pd.DataFrame(
        {
            "fitted_mu": mu,
            "fitted_phi": phi,
            "unit_deviance": d,
            "scaled_deviance": d / np.clip(phi, 1e-300, None),
        }
    )


def _get_y(result: "DGLMResult") -> np.ndarray:
    """Extract response from stored data."""
    formula = result._dglm.formula
    response_name = formula.split("~")[0].strip()
    return result._dglm._data[response_name].to_numpy(dtype=float)
