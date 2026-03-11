"""
Alternating IRLS engine for the Double GLM.

This is the core algorithm from Smyth (1989). The two submodels are fitted
in alternation: fix phi, update beta (mean step); fix mu, update alpha
(dispersion step). Repeat until the joint log-likelihood converges.

The dispersion step is a Gamma GLM on the pseudo-response delta_i = d_i / phi_i,
where d_i is the unit deviance. The Gamma GLM is fitted with log link and
dispersion fixed at 2 (as justified by the saddlepoint approximation
d_i/phi_i ~ chi^2(1) ~ Gamma(1/2, 2)).

REML correction (Smyth & Verbyla 1999): subtract phi_i * h_ii from delta_i
before fitting the dispersion step, where h_ii is the hat-matrix diagonal from
the mean model. This removes the contribution of estimating beta from the
dispersion score, analogous to REML in linear mixed models.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg

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
    # h_ii = ||Q[i,:]||^2 (rows of Q are orthonormal in W^{1/2} space)
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
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Gamma GLM with log link to pseudo-response delta.

    This is the dispersion submodel. The Gamma GLM has:
      E[delta_i] = 1   (since delta_i = d_i/phi_i and E[d_i] = phi_i)
      Var[delta_i] = 2 (dispersion fixed at 2, the saddlepoint constant)

    IRLS for Gamma(log): weight = mu^2 / (2) where mu = exp(eta).
    Working response: z = eta + (delta - mu) / mu  (= eta + delta/mu - 1)

    Parameters
    ----------
    Z : design matrix, shape (n, q)
    delta : pseudo-response (unit deviances / phi), shape (n,)
    weights : optional prior weights, shape (n,)
    alpha_init : warm-start coefficients, shape (q,)

    Returns
    -------
    alpha : shape (q,)
    phi_fitted : exp(Z @ alpha), shape (n,)
    """
    n = len(delta)
    if weights is None:
        weights = np.ones(n)

    # Initialise from warm start or from delta directly
    if alpha_init is not None:
        alpha = alpha_init.copy()
    else:
        eta = np.log(np.clip(delta, 1e-6, None))
        alpha, _, _, _ = np.linalg.lstsq(Z, eta, rcond=None)

    prev_ll = -np.inf

    for _ in range(max_iter):
        eta = Z @ alpha
        mu = np.exp(np.clip(eta, -500, 500))  # E[delta] = exp(eta)

        # Gamma IRLS weights: w = mu^2 * obs_weights / phi_dispersion
        # phi_dispersion = 2 (fixed by saddlepoint theory)
        irls_w = mu ** 2 * weights / 2.0

        # Working response for log-link Gamma IRLS
        # z = eta + (delta - mu) / mu
        z = eta + (delta - mu) / mu

        alpha = _wls(Z, z, irls_w)

        # Convergence check on Gamma log-likelihood
        eta = Z @ alpha
        mu = np.exp(np.clip(eta, -500, 500))
        # Gamma(shape=1/2, mean=1): log f = const - delta/(2*mu) - (1/2)*log(mu)
        # (phi_dispersion = 2 means shape = 1/2)
        ll = float(np.sum(weights * (-0.5 * np.log(mu) - delta / (2.0 * mu))))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    phi_fitted = np.exp(np.clip(Z @ alpha, -500, 500))
    return alpha, phi_fitted


# ---------------------------------------------------------------------------
# Mean submodel IRLS step
# ---------------------------------------------------------------------------

def _fit_mean(
    family: Family,
    X: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    prior_weights: np.ndarray,
    log_offset: Optional[np.ndarray],
    beta_init: Optional[np.ndarray],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the mean submodel via IRLS with observation-level weights 1/phi_i.

    The mean GLM has:
      g(mu_i) = x_i^T beta + offset_i
      Var[Y_i] = phi_i * V(mu_i)

    So the IRLS weight for obs i is prior_weight_i / (phi_i * V(mu_i) * g'(mu_i)^2).

    Parameters
    ----------
    family : the mean family
    X : design matrix (n, p)
    y : response (n,)
    phi : current dispersion vector (n,)
    prior_weights : (n,)
    log_offset : log(exposure) or None, shape (n,)
    beta_init : warm-start coefficients
    max_iter / tol : convergence control

    Returns
    -------
    beta : (p,)
    mu : (n,)
    irls_weights : (n,) — IRLS weights at convergence (used for hat diagonal)
    """
    n = len(y)
    offset = log_offset if log_offset is not None else np.zeros(n)

    if beta_init is not None:
        beta = beta_init.copy()
    else:
        # Moment init: project log(init_mu) onto X
        mu0 = family.init_mu(y)
        eta0 = family.mu_to_eta(mu0) - offset
        beta, _, _, _ = np.linalg.lstsq(X, eta0, rcond=None)

    prev_ll = -np.inf

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

        # Working response: z = eta + (y - mu) * g'(mu) - offset
        z_full = eta + (y - mu) * deta_dmu
        z = z_full - offset  # regress without offset

        beta = _wls(X, z, irls_w)

        # Convergence check on mean log-likelihood
        eta = X @ beta + offset
        mu = family.eta_to_mu(eta)
        ll_arr = family.log_likelihood(y, mu, phi)
        ll = float(np.sum(np.where(np.isfinite(ll_arr), ll_arr * prior_weights, 0.0)))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

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
        Convergence threshold: relative change in -2*loglik.
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
    # Initialise
    # -----------------------------------------------------------------------
    # Step 0a: Fit intercept-only mean GLM to get initial mu
    X_int = np.ones((n, 1))  # intercept only
    beta0, mu, _ = _fit_mean(
        family, X_int, y,
        phi=np.ones(n),
        prior_weights=prior_weights,
        log_offset=log_offset,
        beta_init=None,
        max_iter=50, tol=1e-8,
    )
    # Now fit with the full design matrix for a better beta start
    beta, mu, irls_w = _fit_mean(
        family, X, y,
        phi=np.ones(n),
        prior_weights=prior_weights,
        log_offset=log_offset,
        beta_init=None,
        max_iter=50, tol=1e-8,
    )

    # Step 0b: Initialise phi from intercept-only dispersion GLM
    d = family.deviance_resid(y, mu)
    d = np.clip(d, 1e-10, None)
    delta = d.copy()  # phi=1 initially so delta = d/1

    Z_int = np.ones((n, 1))
    alpha, phi = _gamma_glm_irls(Z_int, delta, weights=prior_weights, max_iter=50)
    # phi is now a constant (intercept-only) for all obs

    # Expand to full dispersion design
    alpha, phi = _gamma_glm_irls(
        Z, delta, weights=prior_weights, alpha_init=None, max_iter=50
    )

    # -----------------------------------------------------------------------
    # Outer alternating loop
    # -----------------------------------------------------------------------
    loglik_history: list[float] = []
    prev_m2ll = np.inf  # -2 * loglik
    converged = False
    n_iter = 0

    alpha_prev = alpha.copy()

    for iteration in range(maxit):
        n_iter = iteration + 1

        # ------------------------------------------------------------------
        # Mean step: update beta given current phi
        # ------------------------------------------------------------------
        beta, mu, irls_w = _fit_mean(
            family, X, y,
            phi=phi,
            prior_weights=prior_weights,
            log_offset=log_offset,
            beta_init=beta,
            max_iter=50, tol=1e-10,
        )

        # ------------------------------------------------------------------
        # Dispersion step: fit Gamma GLM on pseudo-response delta_i = d_i/phi_i
        # ------------------------------------------------------------------
        d = family.deviance_resid(y, mu)
        d = np.clip(d, 1e-14, None)
        delta = d / np.clip(phi, 1e-300, None)

        # REML correction: subtract leverage from pseudo-response
        if method == "reml":
            h = _hat_diagonal(X, irls_w)
            h = np.clip(h, 0.0, 1.0 - 1e-6)
            delta = delta - h
            # Clamp from below to avoid Gamma GLM seeing negative responses
            delta = np.clip(delta, 1e-14, None)

        alpha, phi = _gamma_glm_irls(
            Z, delta,
            weights=prior_weights,
            alpha_init=alpha,
            max_iter=50, tol=1e-10,
        )

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        ll = _joint_loglik(family, y, mu, phi, prior_weights)
        loglik_history.append(ll)
        m2ll = -2.0 * ll

        if verbose:
            print(f"  DGLM iter {n_iter}: loglik = {ll:.6f}")

        rel_change = abs(m2ll - prev_m2ll) / (abs(prev_m2ll) + 1.0)
        if rel_change < epsilon:
            converged = True
            break

        prev_m2ll = m2ll
        alpha_prev = alpha.copy()

    if not converged:
        warnings.warn(
            f"DGLM did not converge after {maxit} iterations. "
            f"Final relative change in -2*loglik: {rel_change:.2e}. "
            "Try increasing maxit or check for data issues.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Compute dispersion IRLS weights (mu^2 / 2 for Gamma with phi_disp=2)
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
    # (X^T W X) = (Xw^T Xw)
    XtWX = Xw.T @ Xw
    try:
        return np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(XtWX)
