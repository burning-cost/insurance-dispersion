"""
Alternating IRLS engine for the Double GLM.

This is the core algorithm from Smyth (1989). The two submodels are fitted
in alternation:
  - Fix phi, update beta via weighted IRLS (mean step)
  - Fix mu, update alpha via Gamma GLM IRLS (dispersion step)
Repeat until the change in total deviance is small.

Convergence criterion: relative change in total deviance (R dglm convention).

The dispersion step is a Gamma GLM on pseudo-response delta_i = d_i / phi_i.
Dispersion of the Gamma GLM is fixed at 2 (saddlepoint: d_i/phi_i ~ Gamma(1/2,2)).

REML correction (Smyth & Verbyla 1999): delta_i -= h_ii before the Gamma fit.
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
    Weighted least squares: beta = argmin sum_i w_i * (z_i - x_i^T beta)^2.

    Returns beta of shape (p,). Handles near-singular systems robustly.
    """
    sqrt_w = np.sqrt(np.clip(w, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]
    zw = z * sqrt_w
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
    except Exception:
        beta = np.linalg.pinv(Xw) @ zw
    return beta


# ---------------------------------------------------------------------------
# Hat matrix diagonal
# ---------------------------------------------------------------------------

def _hat_diagonal(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Hat matrix diagonal: h_ii = diag(X (X^T W X)^{-1} X^T W).
    """
    sqrt_w = np.sqrt(np.clip(w, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]
    try:
        Q, _ = np.linalg.qr(Xw, mode="reduced")
        h = np.sum(Q ** 2, axis=1)
    except Exception:
        h = np.zeros(X.shape[0])
    return h


# ---------------------------------------------------------------------------
# Gamma GLM IRLS for dispersion step
# ---------------------------------------------------------------------------

def _gamma_glm_irls(
    Z: np.ndarray,
    delta: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha_init: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Gamma GLM with log link and fixed dispersion=2 to delta.

    E[delta_i] = exp(z_i^T alpha). Returns (alpha, exp(Z @ alpha)).
    """
    n = len(delta)
    if weights is None:
        weights = np.ones(n)

    delta_fit = np.clip(delta, 1e-8, None)

    if alpha_init is not None:
        alpha = alpha_init.copy()
    else:
        # Init from log(mean(delta))
        m = max(float(np.mean(delta_fit)), 1e-8)
        alpha, _, _, _ = np.linalg.lstsq(
            Z, np.full(n, np.log(m)), rcond=None
        )

    for _ in range(max_iter):
        eta = Z @ alpha
        mu = np.exp(np.clip(eta, -30, 30))

        irls_w = mu ** 2 * weights / 2.0
        irls_w = np.clip(irls_w, 1e-14, None)

        z = eta + (delta_fit - mu) / mu
        alpha_new = _wls(Z, z, irls_w)

        denom = max(np.linalg.norm(alpha), 1e-8)
        if np.linalg.norm(alpha_new - alpha) / denom < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    phi = np.exp(np.clip(Z @ alpha, -30, 30))
    return alpha, phi


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
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit mean submodel via IRLS with weights prior_w / (phi * V(mu) * g'(mu)^2).

    Returns (beta, mu, irls_weights).
    """
    n = len(y)
    offset = log_offset if log_offset is not None else np.zeros(n)

    if beta_init is not None:
        beta = beta_init.copy()
    else:
        mu0 = family.init_mu(y)
        eta0 = family.mu_to_eta(mu0) - offset
        beta, _, _, _ = np.linalg.lstsq(X, eta0, rcond=None)

    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = family.eta_to_mu(eta)
        mu = np.clip(mu, 1e-300, None)

        V = family.variance(mu)
        g_prime = family.link.deriv(mu)

        irls_w = prior_weights / (phi * np.clip(V, 1e-300, None) * g_prime ** 2)
        irls_w = np.clip(irls_w, 1e-14, None)

        z_full = eta + (y - mu) * g_prime
        z = z_full - offset  # regress without offset

        beta_new = _wls(X, z, irls_w)

        denom = max(np.linalg.norm(beta), 1e-8)
        if np.linalg.norm(beta_new - beta) / denom < tol:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta + offset
    mu = family.eta_to_mu(eta)
    mu = np.clip(mu, 1e-300, None)
    V = family.variance(mu)
    g_prime = family.link.deriv(mu)
    irls_w_final = prior_weights / (phi * np.clip(V, 1e-300, None) * g_prime ** 2)
    irls_w_final = np.clip(irls_w_final, 1e-14, None)

    return beta, mu, irls_w_final


# ---------------------------------------------------------------------------
# Total deviance
# ---------------------------------------------------------------------------

def _total_deviance(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
) -> float:
    """Weighted sum of unit deviances d_i = deviance_resid(y_i, mu_i)."""
    d = family.deviance_resid(y, mu)
    d = np.where(np.isfinite(d), d, 0.0)
    return float(np.sum(d * prior_weights))


def _joint_loglik(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
) -> float:
    """Weighted joint log-likelihood."""
    ll_arr = family.log_likelihood(y, mu, phi)
    ll_arr = np.where(np.isfinite(ll_arr), ll_arr, 0.0)
    return float(np.sum(ll_arr * prior_weights))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class DGLMFitResult(NamedTuple):
    beta: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    phi: np.ndarray
    irls_weights: np.ndarray
    disp_irls_weights: np.ndarray
    loglik_history: list[float]
    converged: bool
    n_iter: int


# ---------------------------------------------------------------------------
# Main alternating IRLS
# ---------------------------------------------------------------------------

def dglm_fit(
    family: Family,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    prior_weights: Optional[np.ndarray] = None,
    log_offset: Optional[np.ndarray] = None,
    method: str = "reml",
    maxit: int = 30,
    epsilon: float = 1e-6,
    verbose: bool = False,
) -> DGLMFitResult:
    """
    Fit a Double GLM via alternating IRLS (Smyth 1989).

    Convergence criterion: relative change in the WITHIN-DISPERSION-STEP
    deviance (i.e. the Gamma GLM deviance), following R dglm. The algorithm
    converges when the dispersion deviance stops changing between outer
    iterations.

    Parameters
    ----------
    family : Family
    X : ndarray (n, p)
    Z : ndarray (n, q)
    y : ndarray (n,)
    prior_weights : ndarray (n,), optional
    log_offset : ndarray (n,), optional
    method : {'reml', 'ml'}
    maxit : int
    epsilon : float — convergence tolerance on deviance change
    verbose : bool
    """
    n = len(y)
    if prior_weights is None:
        prior_weights = np.ones(n)

    method = method.lower()
    if method not in ("reml", "ml"):
        raise ValueError(f"method must be 'reml' or 'ml'.")

    # -----------------------------------------------------------------------
    # Initialise
    # -----------------------------------------------------------------------
    # Step 1: Fit mean GLM with phi=1
    beta, mu, irls_w = _fit_mean(
        family, X, y,
        phi=np.ones(n),
        prior_weights=prior_weights,
        log_offset=log_offset,
        beta_init=None,
        max_iter=100, tol=1e-8,
    )

    # Step 2: Initialise dispersion from unit deviances (phi_init=1)
    d_init = np.clip(family.deviance_resid(y, mu), 1e-8, None)
    alpha, phi = _gamma_glm_irls(
        Z, d_init,
        weights=prior_weights,
        alpha_init=None,
        max_iter=100, tol=1e-8,
    )

    # -----------------------------------------------------------------------
    # Outer alternating loop
    # -----------------------------------------------------------------------
    loglik_history: list[float] = []
    converged = False
    n_iter = 0

    # Track the Gamma deviance from the dispersion step for convergence
    # (following R dglm: convergence on dispersion deviance change)
    prev_disp_dev = np.inf

    for iteration in range(maxit):
        n_iter = iteration + 1

        # ------------------------------------------------------------------
        # Mean step: update beta with current phi
        # ------------------------------------------------------------------
        beta, mu, irls_w = _fit_mean(
            family, X, y,
            phi=phi,
            prior_weights=prior_weights,
            log_offset=log_offset,
            beta_init=beta,
            max_iter=100, tol=1e-8,
        )

        # ------------------------------------------------------------------
        # Dispersion step: Gamma GLM on delta = d/phi
        # ------------------------------------------------------------------
        d = np.clip(family.deviance_resid(y, mu), 1e-8, None)
        delta = d / np.clip(phi, 1e-300, None)

        if method == "reml":
            h = _hat_diagonal(X, irls_w)
            h = np.clip(h, 0.0, 0.99)
            delta = delta - h
            delta = np.clip(delta, 1e-8, None)

        alpha, phi = _gamma_glm_irls(
            Z, delta,
            weights=prior_weights,
            alpha_init=alpha,
            max_iter=100, tol=1e-8,
        )

        # ------------------------------------------------------------------
        # Log-likelihood and convergence
        # ------------------------------------------------------------------
        ll = _joint_loglik(family, y, mu, phi, prior_weights)
        loglik_history.append(ll)

        # Compute Gamma GLM deviance for the dispersion step
        # (sum of Gamma unit deviances: 2*(delta/mu - 1 - log(delta/mu)))
        mu_disp = np.exp(np.clip(Z @ alpha, -30, 30))
        gamma_dev = float(np.sum(
            prior_weights * 2.0 * (
                np.clip(delta, 1e-300, None) / mu_disp - 1.0
                - np.log(np.clip(delta / mu_disp, 1e-300, None))
            )
        ))

        if verbose:
            print(f"  DGLM iter {n_iter}: loglik = {ll:.6f}, disp_dev = {gamma_dev:.6f}")

        # Convergence: relative change in Gamma deviance
        rel_change = abs(gamma_dev - prev_disp_dev) / (abs(prev_disp_dev) + 1e-3)
        if rel_change < epsilon:
            converged = True
            break

        prev_disp_dev = gamma_dev

    if not converged:
        warnings.warn(
            f"DGLM did not converge after {maxit} iterations. "
            f"Final relative dispersion deviance change: {rel_change:.2e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    phi_final = np.exp(np.clip(Z @ alpha, -30, 30))
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

def _sandwich_vcov(X: np.ndarray, irls_weights: np.ndarray) -> np.ndarray:
    """Asymptotic covariance: (X^T W X)^{-1}."""
    sqrt_w = np.sqrt(np.clip(irls_weights, 1e-14, None))
    Xw = X * sqrt_w[:, np.newaxis]
    XtWX = Xw.T @ Xw
    try:
        return np.linalg.inv(XtWX)
    except Exception:
        return np.linalg.pinv(XtWX)
