"""
Alternating IRLS engine for the Double GLM.

This is the core algorithm from Smyth (1989). The two submodels are fitted
in alternation: fix phi, update beta (mean step); fix mu, update alpha
(dispersion step). Repeat until convergence.

The dispersion step is a Gamma GLM on the pseudo-response delta_i = d_i / phi_i,
where d_i is the unit deviance. The Gamma GLM is fitted with log link and
dispersion fixed at 2 (as justified by the saddlepoint approximation
d_i/phi_i ~ chi^2(1) ~ Gamma(1/2, 2)).

REML correction (Smyth & Verbyla 1999): subtract phi_i * h_ii from delta_i
before fitting the dispersion step.

CONVERGENCE NOTE:
The alternating algorithm does not guarantee monotone increase of the JOINT
log-likelihood. Each step maximises its sub-objective, but the joint function
can oscillate slightly. We therefore use RELATIVE PARAMETER CHANGE (||beta_new
- beta_old|| + ||alpha_new - alpha_old||) as the convergence criterion, not
log-likelihood change. Log-likelihood is tracked for diagnostic purposes.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np

from insurance_dispersion.families import Family


# ---------------------------------------------------------------------------
# WLS helper
# ---------------------------------------------------------------------------

def _wls(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted least squares: solve (X^T W X) beta = X^T W z.

    Uses the square-root trick for numerical stability. Returns beta of shape (p,).
    """
    sqrt_w = np.sqrt(np.clip(w, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]
    zw = z * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
    return beta


# ---------------------------------------------------------------------------
# Hat matrix diagonal
# ---------------------------------------------------------------------------

def _hat_diagonal(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute h_ii = diag(X (X^T W X)^{-1} X^T W) for the weighted mean model.

    Used for the REML correction. Computed via QR decomposition of the
    weighted design matrix to avoid forming X^T W X explicitly.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    w : ndarray, shape (n,) — IRLS weights

    Returns
    -------
    h : ndarray, shape (n,)
    """
    sqrt_w = np.sqrt(np.clip(w, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]  # (n, p)
    Q, _ = np.linalg.qr(Xw, mode="reduced")
    h = np.sum(Q ** 2, axis=1)
    return h


# ---------------------------------------------------------------------------
# Gamma GLM IRLS for dispersion step
# ---------------------------------------------------------------------------

def _gamma_glm_irls(
    Z: np.ndarray,
    delta: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha_init: Optional[np.ndarray] = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Gamma GLM with log link to pseudo-response delta.

    This is the dispersion submodel. The Gamma GLM has:
      E[delta_i] = phi_i = exp(z_i^T alpha)
      Var[delta_i] / phi_i^2 = 2  (dispersion fixed at 2 by saddlepoint)

    IRLS for Gamma(log):
      weight_i = mu_i^2 * obs_weight_i / 2
      working response: z = eta + (delta - mu) / mu

    Parameters
    ----------
    Z : design matrix, shape (n, q)
    delta : pseudo-response (unit deviances / phi), shape (n,)
    weights : optional prior weights, shape (n,)
    alpha_init : warm-start coefficients, shape (q,)
    """
    n = len(delta)
    if weights is None:
        weights = np.ones(n)

    # Initialise
    if alpha_init is not None:
        alpha = alpha_init.copy()
    else:
        # Robust init: log of clipped delta
        eta = np.log(np.clip(delta, 1e-6, None))
        alpha, _, _, _ = np.linalg.lstsq(Z, eta, rcond=None)

    prev_alpha = alpha.copy()

    for _ in range(max_iter):
        eta = Z @ alpha
        mu = np.exp(np.clip(eta, -500, 500))  # E[delta] = exp(eta) = phi

        # Gamma IRLS weights: w = mu^2 * obs_weights / 2
        irls_w = mu ** 2 * weights / 2.0
        irls_w = np.clip(irls_w, 1e-14, None)

        # Working response for log-link Gamma IRLS
        z = eta + (delta - mu) / mu

        alpha = _wls(Z, z, irls_w)

        # Convergence: relative change in alpha
        denom = max(np.linalg.norm(prev_alpha), 1e-8)
        if np.linalg.norm(alpha - prev_alpha) / denom < tol:
            break
        prev_alpha = alpha.copy()

    phi_fitted = np.exp(np.clip(Z @ alpha, -500, 500))
    return alpha, phi_fitted


# ---------------------------------------------------------------------------
# Mean submodel IRLS
# ---------------------------------------------------------------------------

def _fit_mean(
    family: Family,
    X: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
    log_offset: Optional[np.ndarray],
    beta_init: Optional[np.ndarray],
    max_iter: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the mean submodel via IRLS with observation-level weights 1/phi_i.

    Convergence is assessed by relative change in beta, not log-likelihood,
    because phi changes between outer iterations making LL-based convergence
    unreliable for the inner loop.
    """
    n = len(y)
    offset = log_offset if log_offset is not None else np.zeros(n)

    if beta_init is not None:
        beta = beta_init.copy()
    else:
        mu0 = family.init_mu(y)
        eta0 = family.mu_to_eta(mu0) - offset
        beta, _, _, _ = np.linalg.lstsq(X, eta0, rcond=None)

    prev_beta = beta.copy()

    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = family.eta_to_mu(eta)
        mu = np.clip(mu, 1e-300, None)

        # Variance function and link derivative
        V = family.variance(mu)
        deta_dmu = family.link.deriv(mu)

        # IRLS weight: w_i = prior_w / (phi_i * V(mu_i) * (g'(mu_i))^2)
        irls_w = prior_weights / (phi * np.clip(V, 1e-300, None) * deta_dmu ** 2)
        irls_w = np.clip(irls_w, 1e-14, None)

        # Working response (without offset for WLS regression)
        z_full = eta + (y - mu) * deta_dmu
        z = z_full - offset

        beta = _wls(X, z, irls_w)

        # Convergence: relative change in beta
        denom = max(np.linalg.norm(prev_beta), 1e-8)
        if np.linalg.norm(beta - prev_beta) / denom < tol:
            break
        prev_beta = beta.copy()

    # Compute final mu and irls_weights
    eta = X @ beta + offset
    mu = family.eta_to_mu(eta)
    mu = np.clip(mu, 1e-300, None)
    V = family.variance(mu)
    deta_dmu = family.link.deriv(mu)
    irls_w_final = prior_weights / (phi * np.clip(V, 1e-300, None) * deta_dmu ** 2)
    irls_w_final = np.clip(irls_w_final, 1e-14, None)

    return beta, mu, irls_w_final


# ---------------------------------------------------------------------------
# Joint log-likelihood
# ---------------------------------------------------------------------------

def _joint_loglik(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
) -> float:
    """Weighted joint log-likelihood summed over observations."""
    ll_arr = family.log_likelihood(y, mu, phi)
    ll_arr = np.where(np.isfinite(ll_arr), ll_arr, 0.0)
    return float(np.sum(ll_arr * prior_weights))


# ---------------------------------------------------------------------------
# Main alternating IRLS
# ---------------------------------------------------------------------------

class DGLMFitResult(NamedTuple):
    """Raw result from the alternating IRLS, before wrapping in DGLMResult."""
    beta: np.ndarray          # mean model coefficients (p,)
    alpha: np.ndarray         # dispersion model coefficients (q,)
    mu: np.ndarray            # fitted means (n,)
    phi: np.ndarray           # fitted dispersions (n,)
    irls_weights: np.ndarray  # IRLS weights at convergence (n,), for std error
    disp_irls_weights: np.ndarray  # dispersion IRLS weights (n,)
    loglik_history: list[float]
    converged: bool
    n_iter: int


def dglm_fit(
    family: Family,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    prior_weights: Optional[np.ndarray] = None,
    log_offset: Optional[np.ndarray] = None,
    method: str = "reml",
    maxit: int = 30,
    epsilon: float = 1e-7,
    verbose: bool = False,
) -> DGLMFitResult:
    """
    Fit a Double GLM via alternating IRLS (Smyth 1989).

    Parameters
    ----------
    family : Family
        Mean family (Gamma, Gaussian, etc.).
    X : ndarray, shape (n, p)
        Mean submodel design matrix.
    Z : ndarray, shape (n, q)
        Dispersion submodel design matrix.
    y : ndarray, shape (n,)
        Response observations.
    prior_weights : ndarray, shape (n,), optional
        Observation weights (earned exposure for frequency models, etc.).
    log_offset : ndarray, shape (n,), optional
        log(exposure) added to the mean linear predictor only.
    method : {'reml', 'ml'}
        Whether to apply the REML correction to the dispersion pseudo-response.
    maxit : int
        Maximum outer iterations.
    epsilon : float
        Convergence threshold: relative change in (||beta|| + ||alpha||).
    verbose : bool
        Print log-likelihood per iteration.

    Returns
    -------
    DGLMFitResult
    """
    n = len(y)

    if prior_weights is None:
        prior_weights = np.ones(n)

    method = method.lower()
    if method not in ("reml", "ml"):
        raise ValueError(f"method must be 'reml' or 'ml', got '{method}'.")

    # -----------------------------------------------------------------------
    # Initialise: fit mean GLM with phi=1, then fit constant phi
    # -----------------------------------------------------------------------
    beta, mu, irls_w = _fit_mean(
        family, X, y,
        phi=np.ones(n),
        prior_weights=prior_weights,
        log_offset=log_offset,
        beta_init=None,
        max_iter=50, tol=1e-8,
    )

    # Initial dispersion estimate from unit deviances
    d = family.deviance_resid(y, mu)
    d = np.clip(d, 1e-14, None)
    # Use constant phi from moment estimate (Gamma GLM on intercept-only)
    Z_int = np.ones((n, 1))
    alpha0, phi = _gamma_glm_irls(Z_int, d, weights=prior_weights, max_iter=50)
    # Now initialise dispersion model on full Z
    alpha, phi = _gamma_glm_irls(Z, d, weights=prior_weights, max_iter=50)

    # -----------------------------------------------------------------------
    # Outer alternating loop
    # -----------------------------------------------------------------------
    loglik_history: list[float] = []
    converged = False
    n_iter = 0
    prev_params_norm = None

    for iteration in range(maxit):
        n_iter = iteration + 1

        beta_old = beta.copy()
        alpha_old = alpha.copy()

        # ------------------------------------------------------------------
        # Mean step: update beta given current phi
        # ------------------------------------------------------------------
        beta, mu, irls_w = _fit_mean(
            family, X, y,
            phi=phi,
            prior_weights=prior_weights,
            log_offset=log_offset,
            beta_init=beta,
            max_iter=25, tol=1e-6,
        )

        # ------------------------------------------------------------------
        # Dispersion step: fit Gamma GLM on delta_i = d_i / phi_i
        # ------------------------------------------------------------------
        d = family.deviance_resid(y, mu)
        d = np.clip(d, 1e-14, None)
        delta = d / np.clip(phi, 1e-300, None)

        # REML correction: subtract hat-matrix leverage
        if method == "reml":
            h = _hat_diagonal(X, irls_w)
            h = np.clip(h, 0.0, 1.0 - 1e-6)
            delta = delta - h
            delta = np.clip(delta, 1e-14, None)

        alpha, phi = _gamma_glm_irls(
            Z, delta,
            weights=prior_weights,
            alpha_init=alpha,
            max_iter=25, tol=1e-6,
        )

        # ------------------------------------------------------------------
        # Track log-likelihood
        # ------------------------------------------------------------------
        ll = _joint_loglik(family, y, mu, phi, prior_weights)
        loglik_history.append(ll)

        if verbose:
            print(f"  DGLM iter {n_iter}: loglik = {ll:.6f}")

        # ------------------------------------------------------------------
        # Convergence: relative change in parameter vector
        # ------------------------------------------------------------------
        params_norm = np.linalg.norm(beta) + np.linalg.norm(alpha)
        delta_params = np.linalg.norm(beta - beta_old) + np.linalg.norm(alpha - alpha_old)
        denom = max(params_norm, 1e-8)
        rel_change = delta_params / denom

        if rel_change < epsilon:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"DGLM did not converge after {maxit} iterations. "
            f"Final relative parameter change: {rel_change:.2e}. "
            "Try increasing maxit or check for data issues.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Final phi
    eta_z = Z @ alpha
    phi_final = np.exp(np.clip(eta_z, -500, 500))
    disp_irls_w = phi_final ** 2 * prior_weights / 2.0

    return DGLMFitResult(
        beta=beta,
        alpha=alpha,
        mu=mu,
        phi=phi_final,
        irls_weights=irls_w,
        disp_irls_weights=disp_irls_w,
        loglik_history=loglik_history,
        converged=converged,
        n_iter=n_iter,
    )


# ---------------------------------------------------------------------------
# Standard error computation
# ---------------------------------------------------------------------------

def _sandwich_vcov(
    X: np.ndarray,
    irls_weights: np.ndarray,
) -> np.ndarray:
    """
    Asymptotic covariance matrix of beta: (X^T W X)^{-1}.

    This is the standard GLM information matrix inverse. Returns (p, p) matrix.
    """
    sqrt_w = np.sqrt(np.clip(irls_weights, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]
    XtWX = Xw.T @ Xw
    try:
        return np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(XtWX)
