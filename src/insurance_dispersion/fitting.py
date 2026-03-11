"""
Alternating IRLS engine for the Double GLM.

This is the core algorithm from Smyth (1989). The two submodels are fitted
in alternation:
  - Fix phi, update beta via weighted IRLS (mean step)
  - Fix mu, update alpha via Gamma GLM IRLS (dispersion step)
Repeat until convergence.

Dispersion pseudo-response:
  The dispersion submodel Gamma GLM is fitted to the unit deviances d_i
  directly (NOT d_i / phi_i). The Gamma family has E[d_i] = phi_i, so the
  Gamma GLM log(E[d_i]) = z_i^T alpha recovers log(phi_i). Dividing by phi_i
  (as sometimes done in saddlepoint implementations for non-Gamma families)
  would produce a biased fixed point: phi_fixed = sqrt(phi_true) ≠ phi_true.

REML correction (Smyth & Verbyla 1999):
  Subtract h_ii * phi_i from d_i before the dispersion fit. This corrects for
  the effective degrees of freedom consumed by the mean model.

Numerical stability:
  - phi clamped to [PHI_MIN, PHI_MAX] to prevent weight explosion in mean IRLS
  - IRLS weights clamped to [1e-10, 1e10]
  - Step damping on alpha update prevents oscillation; does not bias the
    fixed point (since the fixed point of alpha = d*step + (1-d)*old is
    step = old, i.e. the IRLS solution = current alpha, i.e. the ML solution)
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np

from insurance_dispersion.families import Family


# Clamp phi to this range to prevent IRLS weight explosion in the mean step.
PHI_MIN = 1e-4
PHI_MAX = 1e4

# Step damping for dispersion update: alpha_new = DAMP * alpha_step + (1-DAMP) * alpha_old
DAMP = 0.7


# ---------------------------------------------------------------------------
# WLS helper
# ---------------------------------------------------------------------------

def _wls(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted least squares: beta = argmin sum_i w_i * (z_i - x_i^T beta)^2.
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
    """Hat diagonal h_ii = diag(H) where H = X(X^T W X)^{-1} X^T W."""
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

    delta should be the unit deviances d_i (or REML-adjusted d_i - h_ii*phi_i).
    Models E[delta_i] = phi_i = exp(z_i^T alpha).

    Returns (alpha, phi) where phi = exp(Z @ alpha), clamped to [PHI_MIN, PHI_MAX].
    """
    n = len(delta)
    if weights is None:
        weights = np.ones(n)

    delta_fit = np.clip(delta, 1e-8, None)

    if alpha_init is not None:
        alpha = alpha_init.copy()
    else:
        m = np.log(np.clip(np.mean(delta_fit), 1e-8, None))
        alpha, _, _, _ = np.linalg.lstsq(Z, np.full(n, m), rcond=None)

    for _ in range(max_iter):
        eta = Z @ alpha
        # Clamp eta to prevent overflow in exp
        eta_clamped = np.clip(eta, np.log(PHI_MIN), np.log(PHI_MAX))
        mu = np.exp(eta_clamped)

        # Gamma IRLS weights: mu^2 / (2 * phi_gamma) = mu^2 / 2 (phi_gamma=2 fixed)
        irls_w = np.clip(mu ** 2 * weights / 2.0, 1e-14, None)
        # Working response for Gamma with log link: eta + (y - mu)/mu
        z_irls = eta_clamped + (delta_fit - mu) / mu
        alpha_new = _wls(Z, z_irls, irls_w)

        denom = max(np.linalg.norm(alpha), 1e-8)
        if np.linalg.norm(alpha_new - alpha) / denom < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    eta_final = np.clip(Z @ alpha, np.log(PHI_MIN), np.log(PHI_MAX))
    phi = np.exp(eta_final)
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
    Fit mean submodel via IRLS. Returns (beta, mu, irls_weights).

    phi is clamped to [PHI_MIN, PHI_MAX] before computing weights to prevent
    weight explosion when the dispersion step overshoots.
    """
    n = len(y)
    offset = log_offset if log_offset is not None else np.zeros(n)
    phi_safe = np.clip(phi, PHI_MIN, PHI_MAX)

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

        # Weight: prior_w / (phi * V(mu) * g'(mu)^2); clamp to [1e-10, 1e10]
        raw_w = prior_weights / (phi_safe * np.clip(V, 1e-300, None) * g_prime ** 2)
        irls_w = np.clip(raw_w, 1e-10, 1e10)

        z_full = eta + (y - mu) * g_prime
        z = z_full - offset

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
    raw_w = prior_weights / (phi_safe * np.clip(V, 1e-300, None) * g_prime ** 2)
    irls_w_final = np.clip(raw_w, 1e-10, 1e10)

    return beta, mu, irls_w_final


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def _joint_loglik(
    family: Family,
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
) -> float:
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

    The dispersion pseudo-response is the unit deviance d_i (NOT d_i/phi_i).
    The Gamma GLM models E[d_i] = phi_i. Using d_i/phi_i would give a biased
    fixed point phi_fixed = sqrt(phi_true) due to the self-referential scaling.

    Convergence: relative change in Gamma deviance between outer iterations,
    following the R dglm convention. This is tracked on the UNDAMPED alpha_step
    to avoid the slow-convergence issue of measuring a permanently damped sequence.

    Step damping (DAMP=0.7) is applied to alpha updates for numerical stability.
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
    beta, mu, irls_w = _fit_mean(
        family, X, y,
        phi=np.ones(n),
        prior_weights=prior_weights,
        log_offset=log_offset,
        beta_init=None,
        max_iter=100, tol=1e-8,
    )

    # Init dispersion: fit Gamma GLM to raw unit deviances
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
    prev_disp_dev = np.inf
    rel_change = np.inf

    for iteration in range(maxit):
        n_iter = iteration + 1

        # ------------------------------------------------------------------
        # Mean step: update beta, mu given current phi
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
        # Dispersion step
        #
        # The pseudo-response is the raw unit deviance d_i.
        # E[d_i] = phi_i, so Gamma GLM with log link recovers log(phi_i).
        #
        # REML correction: d_i -= h_ii * phi_i  (Smyth & Verbyla 1999)
        # ------------------------------------------------------------------
        d = np.clip(family.deviance_resid(y, mu), 1e-8, None)
        delta = d.copy()  # pseudo-response: raw unit deviances

        if method == "reml":
            h = _hat_diagonal(X, irls_w)
            h = np.clip(h, 0.0, 0.99)
            phi_safe = np.clip(phi, PHI_MIN, PHI_MAX)
            delta = delta - h * phi_safe
            delta = np.clip(delta, 1e-8, None)

        alpha_old = alpha.copy()
        alpha_step, _ = _gamma_glm_irls(
            Z, delta,
            weights=prior_weights,
            alpha_init=alpha,
            max_iter=100, tol=1e-8,
        )

        # Damped update: alpha moves toward alpha_step at rate DAMP
        alpha = DAMP * alpha_step + (1.0 - DAMP) * alpha_old
        phi = np.exp(np.clip(Z @ alpha, np.log(PHI_MIN), np.log(PHI_MAX)))

        # ------------------------------------------------------------------
        # Convergence: Gamma deviance using alpha_step (undamped proposal)
        # Measure against the proposed fit, not the damped alpha, so that
        # convergence fires when the IRLS solution is stable.
        # ------------------------------------------------------------------
        ll = _joint_loglik(family, y, mu, phi, prior_weights)
        loglik_history.append(ll)

        # Gamma deviance of dispersion pseudo-response at alpha_step
        mu_step = np.exp(np.clip(Z @ alpha_step, np.log(PHI_MIN), np.log(PHI_MAX)))
        delta_safe = np.clip(delta, 1e-300, None)
        mu_step_safe = np.clip(mu_step, 1e-300, None)
        gamma_dev = float(np.sum(
            prior_weights * 2.0 * (
                delta_safe / mu_step_safe - 1.0
                - np.log(delta_safe / mu_step_safe)
            )
        ))

        if verbose:
            print(f"  DGLM iter {n_iter}: loglik={ll:.4f}, disp_dev={gamma_dev:.4f}")

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

    phi_final = np.exp(np.clip(Z @ alpha, np.log(PHI_MIN), np.log(PHI_MAX)))
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
