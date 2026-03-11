"""
DGLM: the main user-facing class.

The DGLM class accepts R-style formula strings for both the mean and
dispersion submodels, uses formulaic to build design matrices, and
delegates fitting to the alternating IRLS engine in fitting.py.

Design choices:
  - formulaic not patsy: actively maintained, cleaner API, better handling
    of prediction on new data via the same model matrix schema
  - exposure as a column name in data, not a pre-computed array, to keep
    the API consistent with predict()
  - method='reml' default: REML correction is almost always preferable in
    insurance data where mean models often have many parameters
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import formulaic

from insurance_dispersion.families import Family
from insurance_dispersion.fitting import dglm_fit, _sandwich_vcov
from insurance_dispersion.results import DGLMResult, SubmodelResult


class DGLM:
    """
    Double GLM for joint modelling of mean and dispersion.

    Parameters
    ----------
    formula : str
        R-style formula for the mean submodel, e.g.
        "claim_amount ~ C(age_band) + C(vehicle_class)".
        Do NOT include the response in dformula.
    dformula : str
        Formula for the dispersion submodel, e.g. "~ C(channel)".
        Left-hand side is ignored if present; always fits to the
        unit deviances of the mean submodel.
    family : Family
        Mean family: Gamma(), Gaussian(), InverseGaussian(), Tweedie(p=1.5),
        Poisson(), NegativeBinomial().
    dlink : str
        Link function for the dispersion submodel. Default 'log' ensures
        phi_i > 0. The only other practical choice is 'identity' (unconstrained,
        use with care for very large phi).
    data : DataFrame, optional
        Training data. Can be passed at fit time instead.
    exposure : str, optional
        Column name in data containing exposure (policy years, etc.).
        Added as log-offset in the mean linear predictor only.
    weights : array-like or str, optional
        Prior weights for the mean submodel. Can be a column name in data
        or a 1-D array.
    method : {'reml', 'ml'}
        REML (recommended) applies the hat-matrix correction to the
        dispersion pseudo-response. ML fits without correction.
    """

    def __init__(
        self,
        formula: str,
        dformula: str,
        family: Family,
        dlink: str = "log",
        data: Optional[pd.DataFrame] = None,
        exposure: Optional[str] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        method: str = "reml",
    ):
        self.formula = formula
        self.dformula = dformula
        self.family = family
        self.dlink = dlink
        self.exposure = exposure
        self.method = method
        # Stored for LRT in overdispersion_test
        self._data = data
        self._weights_input = weights
        self._weights_arr: Optional[np.ndarray] = None
        self._maxit = 30
        self._epsilon = 1e-7

        # Cache design matrix schemas after first fit (for predict)
        self._mean_schema = None
        self._disp_schema = None

    def fit(
        self,
        data: Optional[pd.DataFrame] = None,
        maxit: int = 30,
        epsilon: float = 1e-7,
        verbose: bool = False,
    ) -> DGLMResult:
        """
        Fit the DGLM on data.

        Parameters
        ----------
        data : DataFrame, optional
            If not provided, uses the data passed to __init__.
        maxit : int
            Maximum outer iterations.
        epsilon : float
            Convergence threshold (relative change in -2*loglik).
        verbose : bool
            Print log-likelihood at each outer iteration.

        Returns
        -------
        DGLMResult
        """
        if data is not None:
            self._data = data

        if self._data is None:
            raise ValueError("No data provided. Pass data= to fit() or DGLM().")

        df = self._data
        self._maxit = maxit
        self._epsilon = epsilon

        # ------------------------------------------------------------------
        # Build design matrices via formulaic
        # ------------------------------------------------------------------
        y, X, self._mean_schema = self._parse_mean_formula(df)
        Z, self._disp_schema = self._parse_disp_formula(df)

        # ------------------------------------------------------------------
        # Prior weights
        # ------------------------------------------------------------------
        n = len(y)
        w_input = self._weights_input
        if isinstance(w_input, str):
            prior_weights = df[w_input].to_numpy(dtype=float)
        elif w_input is not None:
            prior_weights = np.asarray(w_input, dtype=float)
            self._weights_arr = prior_weights
        else:
            prior_weights = np.ones(n)
        self._weights_arr = prior_weights

        # ------------------------------------------------------------------
        # Log-offset from exposure column
        # ------------------------------------------------------------------
        log_offset = None
        if self.exposure is not None:
            if self.exposure not in df.columns:
                raise ValueError(
                    f"Exposure column '{self.exposure}' not found in data."
                )
            log_offset = np.log(np.clip(df[self.exposure].to_numpy(dtype=float), 1e-300, None))

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------
        fit_raw = dglm_fit(
            family=self.family,
            X=X,
            Z=Z,
            y=y,
            prior_weights=prior_weights,
            log_offset=log_offset,
            method=self.method,
            maxit=maxit,
            epsilon=epsilon,
            verbose=verbose,
        )

        # ------------------------------------------------------------------
        # Build result objects
        # ------------------------------------------------------------------
        mean_vcov = _sandwich_vcov(X, fit_raw.irls_weights)
        disp_vcov = _sandwich_vcov(Z, fit_raw.disp_irls_weights)

        mean_names = self._get_column_names(self._mean_schema, X)
        disp_names = self._get_column_names(self._disp_schema, Z)

        mean_model = SubmodelResult(fit_raw.beta, mean_names, mean_vcov)
        disp_model = SubmodelResult(fit_raw.alpha, disp_names, disp_vcov)

        loglik = fit_raw.loglik_history[-1] if fit_raw.loglik_history else float("nan")

        return DGLMResult(
            mean_model=mean_model,
            dispersion_model=disp_model,
            mu_=fit_raw.mu,
            phi_=fit_raw.phi,
            loglik=loglik,
            loglik_history=fit_raw.loglik_history,
            converged=fit_raw.converged,
            n_iter=fit_raw.n_iter,
            _dglm=self,
            _fit_raw=fit_raw,
        )

    # ------------------------------------------------------------------
    # Formula parsing helpers
    # ------------------------------------------------------------------

    def _parse_mean_formula(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Parse mean formula, return (y, X, model_matrix_spec)."""
        model_matrix = formulaic.model_matrix(self.formula, df)
        if isinstance(model_matrix, tuple):
            # formulaic returns (lhs, rhs) for two-sided formulas
            lhs, rhs = model_matrix
            y = lhs.to_numpy(dtype=float).ravel()
            X = rhs.to_numpy(dtype=float)
            schema = rhs.model_spec
        else:
            raise ValueError(
                f"Formula '{self.formula}' must be a two-sided formula "
                "with a response variable, e.g. 'y ~ x1 + x2'."
            )
        return y, X, schema

    def _parse_disp_formula(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, object]:
        """
        Parse dispersion formula. The LHS is ignored — we always fit to
        the unit deviances computed during the mean step.
        """
        # Strip LHS if present
        dform = self.dformula.strip()
        if "~" in dform:
            dform = "~" + dform.split("~", 1)[1]

        # Dummy response column to satisfy formulaic for one-sided formula
        # We use a column of ones — they are discarded
        df_aug = df.copy()
        df_aug["__disp_dummy__"] = 1.0
        full_formula = f"__disp_dummy__ {dform}"

        model_matrix = formulaic.model_matrix(full_formula, df_aug)
        if isinstance(model_matrix, tuple):
            _, rhs = model_matrix
        else:
            rhs = model_matrix

        Z = rhs.to_numpy(dtype=float)
        schema = rhs.model_spec
        return Z, schema

    def _build_mean_matrix(self, newdata: pd.DataFrame) -> np.ndarray:
        """Build mean design matrix on new data using the fitted schema."""
        if self._mean_schema is None:
            raise RuntimeError("Call fit() before predict().")
        X_new = self._mean_schema.get_model_matrix(newdata)
        return X_new.to_numpy(dtype=float)

    def _build_disp_matrix(self, newdata: pd.DataFrame) -> np.ndarray:
        """Build dispersion design matrix on new data using the fitted schema."""
        if self._disp_schema is None:
            raise RuntimeError("Call fit() before predict().")
        newdata_aug = newdata.copy()
        newdata_aug["__disp_dummy__"] = 1.0
        Z_new = self._disp_schema.get_model_matrix(newdata_aug)
        return Z_new.to_numpy(dtype=float)

    @staticmethod
    def _get_column_names(schema, matrix: np.ndarray) -> list[str]:
        """Extract column names from formulaic model matrix."""
        try:
            return list(schema.column_names)
        except AttributeError:
            return [f"x{i}" for i in range(matrix.shape[1])]

    def __repr__(self) -> str:
        return (
            f"DGLM(formula='{self.formula}', dformula='{self.dformula}', "
            f"family={self.family}, method='{self.method}')"
        )
