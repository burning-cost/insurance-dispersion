"""
DGLMResult: the object returned by DGLM.fit().

This wraps the raw output of the alternating IRLS and provides:
  - Coefficient tables for mean and dispersion submodels
  - Relativities (exp(coefficients)) for log-linked models
  - Predictions on new data
  - Likelihood ratio test for non-constant dispersion
  - A summary string for interactive use
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import scipy.stats

from insurance_dispersion.fitting import _sandwich_vcov

if TYPE_CHECKING:
    from insurance_dispersion.model import DGLM


class SubmodelResult:
    """
    GLM-like result object for one submodel (mean or dispersion).

    Attributes
    ----------
    coef : ndarray, shape (p,)
    coef_names : list[str]
    vcov : ndarray, shape (p, p) — covariance matrix of coefficients
    se : ndarray, shape (p,) — standard errors
    """

    def __init__(
        self,
        coef: np.ndarray,
        coef_names: list[str],
        vcov: np.ndarray,
    ):
        self.coef = coef
        self.coef_names = coef_names
        self.vcov = vcov
        self.se = np.sqrt(np.clip(np.diag(vcov), 0, None))

    def summary_frame(self) -> pd.DataFrame:
        """Coefficient table with SE, z-stat, and two-sided p-value."""
        z = self.coef / np.where(self.se > 0, self.se, np.nan)
        p = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z)))
        return pd.DataFrame(
            {
                "coef": self.coef,
                "exp_coef": np.exp(self.coef),
                "se": self.se,
                "z": z,
                "p_value": p,
            },
            index=self.coef_names,
        )


class DGLMResult:
    """
    Result from fitting a Double GLM.

    The mean submodel gives factor relativities for the expected loss;
    the dispersion submodel gives factor relativities for uncertainty.
    Together they let you price risk AND quantify per-risk uncertainty.

    Attributes
    ----------
    mean_model : SubmodelResult
    dispersion_model : SubmodelResult
    mu_ : ndarray — fitted means (n,)
    phi_ : ndarray — fitted dispersions (n,)
    loglik : float — joint log-likelihood at convergence
    aic : float
    bic : float
    converged : bool
    n_iter : int
    loglik_history : list[float]
    """

    def __init__(
        self,
        mean_model: SubmodelResult,
        dispersion_model: SubmodelResult,
        mu_: np.ndarray,
        phi_: np.ndarray,
        loglik: float,
        loglik_history: list[float],
        converged: bool,
        n_iter: int,
        # stored for predict() and LRT
        _dglm: "DGLM",
        _fit_raw,
    ):
        self.mean_model = mean_model
        self.dispersion_model = dispersion_model
        self.mu_ = mu_
        self.phi_ = phi_
        self.loglik = loglik
        self.loglik_history = loglik_history
        self.converged = converged
        self.n_iter = n_iter
        self._dglm = _dglm
        self._fit_raw = _fit_raw

        n = len(mu_)
        p_mean = len(mean_model.coef)
        p_disp = len(dispersion_model.coef)
        k = p_mean + p_disp
        self.aic = -2.0 * loglik + 2.0 * k
        self.bic = -2.0 * loglik + np.log(n) * k
        self.n_obs = n

    # ------------------------------------------------------------------
    # Factor tables
    # ------------------------------------------------------------------

    def mean_relativities(self) -> pd.DataFrame:
        """
        Coefficient table for the mean submodel.

        Returns a DataFrame with coef, exp(coef) (the relativity for log-link
        models), standard error, z-statistic, and p-value.
        """
        return self.mean_model.summary_frame()

    def dispersion_relativities(self) -> pd.DataFrame:
        """
        Coefficient table for the dispersion submodel.

        exp(coef) gives the multiplicative effect of each factor on phi_i.
        A factor with exp(coef) = 1.5 inflates per-observation dispersion by 50%.
        """
        return self.dispersion_model.summary_frame()

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict(
        self,
        newdata: pd.DataFrame,
        which: str = "mean",
    ) -> np.ndarray:
        """
        Predict on new data.

        Parameters
        ----------
        newdata : DataFrame
        which : {'mean', 'dispersion', 'variance'}
            'mean'       -> predicted mu_i = g^{-1}(x_i^T beta)
            'dispersion' -> predicted phi_i = exp(z_i^T alpha)
            'variance'   -> predicted Var[Y_i] = phi_i * V(mu_i)

        Returns
        -------
        ndarray, shape (n_new,)
        """
        dglm = self._dglm
        X_new = dglm._build_mean_matrix(newdata)
        Z_new = dglm._build_disp_matrix(newdata)

        eta_mu = X_new @ self.mean_model.coef
        if dglm.exposure is not None and dglm.exposure in newdata.columns:
            eta_mu = eta_mu + np.log(np.clip(newdata[dglm.exposure].to_numpy(dtype=float), 1e-300, None))

        mu_new = dglm.family.eta_to_mu(eta_mu)

        if which == "mean":
            return mu_new

        phi_new = np.exp(np.clip(Z_new @ self.dispersion_model.coef, -500, 500))

        if which == "dispersion":
            return phi_new

        if which == "variance":
            return phi_new * dglm.family.variance(mu_new)

        raise ValueError(
            f"which must be 'mean', 'dispersion', or 'variance'. Got '{which}'."
        )

    # ------------------------------------------------------------------
    # LRT for non-constant dispersion
    # ------------------------------------------------------------------

    def overdispersion_test(self) -> dict:
        """
        Likelihood ratio test: constant phi vs. phi = f(Z).

        The null model has phi = scalar (estimated by the intercept-only
        dispersion GLM). The alternative is the fitted DGLM.

        Returns
        -------
        dict with keys: statistic, df, p_value, conclusion
        """
        from insurance_dispersion.model import DGLM

        dglm = self._dglm
        # Fit null: same mean formula, dformula='~1'
        null_model = DGLM(
            formula=dglm.formula,
            dformula="~1",
            family=dglm.family,
            dlink=dglm.dlink,
            data=dglm._data,
            exposure=dglm.exposure,
            weights=dglm._weights_arr,
            method=dglm.method,
        )
        null_result = null_model.fit(maxit=dglm._maxit, epsilon=dglm._epsilon)

        lr_stat = 2.0 * (self.loglik - null_result.loglik)
        df = len(self.dispersion_model.coef) - 1  # -1 for intercept
        df = max(df, 1)
        p_value = float(1.0 - scipy.stats.chi2.cdf(lr_stat, df=df))

        return {
            "statistic": float(lr_stat),
            "df": int(df),
            "p_value": p_value,
            "conclusion": (
                "Reject constant phi (dispersion varies by covariates)"
                if p_value < 0.05
                else "Fail to reject constant phi"
            ),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of both submodels."""
        lines = [
            "Double GLM (DGLM) Results",
            "=" * 60,
            f"Family:      {self._dglm.family}",
            f"Method:      {self._dglm.method.upper()}",
            f"Observations:{self.n_obs}",
            f"Converged:   {self.converged} (after {self.n_iter} iterations)",
            f"Log-lik:     {self.loglik:.4f}",
            f"AIC:         {self.aic:.4f}",
            f"BIC:         {self.bic:.4f}",
            "",
            "Mean Submodel Coefficients:",
            "-" * 60,
        ]
        mean_df = self.mean_relativities()
        lines.append(mean_df.to_string(float_format="{:.4f}".format))
        lines += [
            "",
            "Dispersion Submodel Coefficients:",
            "-" * 60,
        ]
        disp_df = self.dispersion_relativities()
        lines.append(disp_df.to_string(float_format="{:.4f}".format))
        lines += [
            "",
            f"Fitted phi range: [{self.phi_.min():.4f}, {self.phi_.max():.4f}]",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DGLMResult(family={self._dglm.family}, "
            f"n={self.n_obs}, loglik={self.loglik:.4f}, "
            f"converged={self.converged})"
        )
