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
from formulaic import Formula

from insurance_dispersion.families import Family
from insurance_dispersion.fitting import dglm_fit, _sandwich_vcov
from insurance_dispersion.results import DGLMResult, SubmodelResult


def _formulaic_model_matrices(formula_str: str, df: pd.DataFrame):
    """
    Parse a two-sided formula with formulaic and return (lhs, rhs).

    Handles different formulaic versions robustly. In all versions,
    a two-sided formula 'y ~ x1 + x2' produces a result with 'lhs' and
    'rhs' attributes (ModelMatrices named tuple).
    """
    result = formulaic.model_matrix(formula_str, df)
    # formulaic returns ModelMatrices (namedtuple-like) for two-sided formulas
    # Check for lhs/rhs attributes
    if hasattr(result, "lhs") and hasattr(result, "rhs"):
        return result.lhs, result.rhs
    # Older versions may return a plain tuple
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0], result[1]
    # If it's just a single matrix, the formula was one-sided
    raise ValueError(
        f"Formula '{formula_str}' must be a two-sided formula with a response, "
        "e.g. 'y ~ x1 + x2'. Got a one-sided result from formulaic."
    )


def _formulaic_rhs(formula_str: str, df: pd.DataFrame):
    """
    Parse a formula and return only the RHS design matrix.

    Works for both one-sided ('~ x1 + x2') and two-sided formulas.
    """
    result = formulaic.model_matrix(formula_str, df)
    if hasattr(result, "rhs"):
        return result.rhs
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[1]
    # Single matrix = one-sided formula
    return result


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
        self._data: Optional[pd.DataFrame] = data
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

        df = self._data.reset_index(drop=True)  # ensure clean integer index
        self._data = df  # store reset version so predict() indices align
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
            log_offset = np.log(
                np.clip(df[self.exposure].to_numpy(dtype=float), 1e-300, None)
            )

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
        lhs, rhs = _formulaic_model_matrices(self.formula, df)
        y = lhs.to_numpy(dtype=float).ravel()
        X = rhs.to_numpy(dtype=float)
        schema = rhs.model_spec
        return y, X, schema

    def _parse_disp_formula(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, object]:
        """
        Parse dispersion formula. The LHS is ignored — always fits to unit
        deviances from the mean step.
        """
        dform = self.dformula.strip()
        # Normalise: extract RHS only, build as dummy_response ~ rhs
        if "~" in dform:
            rhs_str = dform.split("~", 1)[1].strip()
        else:
            rhs_str = dform

        # Build a two-sided formula with a dummy LHS so formulaic parses it
        df_aug = df.copy()
        df_aug["__disp_dummy__"] = 1.0
        full_formula = f"__disp_dummy__ ~ {rhs_str}"

        lhs, rhs = _formulaic_model_matrices(full_formula, df_aug)
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
            pass
        # Fallback for different formulaic versions
        if hasattr(matrix, "columns"):
            return list(matrix.columns)
        return [f"x{i}" for i in range(matrix.shape[1])]

    def __repr__(self) -> str:
        return (
            f"DGLM(formula='{self.formula}', dformula='{self.dformula}', "
            f"family={self.family}, method='{self.method}')"
        )
