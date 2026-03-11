"""
Exponential family definitions for the DGLM.

Each family provides:
  - variance(mu): V(mu) — the variance function
  - deviance_resid(y, mu): per-observation unit deviance d_i
  - log_likelihood(y, mu, phi): per-observation log-likelihood
  - link / inverse_link: default link function and its inverse
  - init_mu(y): starting value for mu_i

The dispersion submodel always uses a Gamma GLM internally (justified by
the saddlepoint approximation d_i/phi_i ~ chi^2(1) ~ Gamma(1/2, 2)).
For the Gamma family the approximation is exact; for others it is
asymptotically valid.

Supported families:
  Gaussian, Gamma, InverseGaussian, Tweedie, Poisson, NegativeBinomial
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.special import gammaln, digamma


# ---------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------

class LogLink:
    """Log link: eta = log(mu), mu = exp(eta)."""

    name = "log"

    def link(self, mu: np.ndarray) -> np.ndarray:
        return np.log(np.clip(mu, 1e-300, None))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(eta, -500, 500))

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        """d(eta)/d(mu) = 1/mu."""
        return 1.0 / np.clip(mu, 1e-300, None)

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        """d(mu)/d(eta) = exp(eta)."""
        return np.exp(np.clip(eta, -500, 500))


class IdentityLink:
    """Identity link: eta = mu."""

    name = "identity"

    def link(self, mu: np.ndarray) -> np.ndarray:
        return mu.copy()

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return eta.copy()

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)


class InverseLink:
    """Inverse link: eta = 1/mu, mu = 1/eta. Default for InverseGaussian."""

    name = "inverse"

    def link(self, mu: np.ndarray) -> np.ndarray:
        return 1.0 / np.clip(mu, 1e-300, None)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / np.clip(eta, 1e-300, None)

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        return -1.0 / np.clip(mu, 1e-300, None) ** 2

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        return -1.0 / np.clip(eta, 1e-300, None) ** 2


def _get_link(name: str):
    """Return a link object from a string name."""
    options = {"log": LogLink, "identity": IdentityLink, "inverse": InverseLink}
    if name not in options:
        raise ValueError(
            f"Unknown link '{name}'. Supported: {list(options.keys())}"
        )
    return options[name]()


# ---------------------------------------------------------------------------
# Base family class
# ---------------------------------------------------------------------------

class Family(ABC):
    """
    Abstract base for DGLM exponential families.

    Subclasses implement the mean family. The dispersion submodel always
    uses a Gamma GLM — this is handled in fitting.py, not here.
    """

    def __init__(self, link: str | None = None):
        self._link_obj = _get_link(link) if link else _get_link(self._default_link)

    @property
    def _default_link(self) -> str:
        raise NotImplementedError

    @property
    def link(self):
        return self._link_obj

    def eta_to_mu(self, eta: np.ndarray) -> np.ndarray:
        return self._link_obj.inverse(eta)

    def mu_to_eta(self, mu: np.ndarray) -> np.ndarray:
        return self._link_obj.link(mu)

    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """V(mu) — variance function (not Var[Y]; that is phi * V(mu))."""

    @abstractmethod
    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Per-observation unit deviance d_i = 2 * (log f(y|y) - log f(y|mu)).

        This is phi-free. The dispersion step scales by phi_i: delta_i = d_i / phi_i.
        """

    @abstractmethod
    def log_likelihood(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """Per-observation log-likelihood. phi is observation-level (n,)."""

    @abstractmethod
    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """Robust starting values for mu_i before the outer loop."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(link='{self._link_obj.name}')"


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------

class Gaussian(Family):
    """
    Gaussian (normal) family. V(mu) = 1.

    The unit deviance is the squared residual: d_i = (y_i - mu_i)^2.
    DGLM with Gaussian mean submodel is exact — no saddlepoint approximation
    in the dispersion step.
    """

    _default_link = "identity"

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return (y - mu) ** 2

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        return (
            -0.5 * np.log(2.0 * np.pi * phi)
            - 0.5 * (y - mu) ** 2 / phi
        )

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        return np.full_like(y, float(np.mean(y)))


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class Gamma(Family):
    """
    Gamma family. V(mu) = mu^2.

    Natural for claim severity modelling. The log link ensures positive fitted
    means and gives multiplicative factor tables (relativities).

    For Gamma, the dispersion submodel Gamma GLM is exact (not saddlepoint
    approximation), so DGLM is fully justified even at small sample sizes.
    """

    _default_link = "log"

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu ** 2

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-300, None)
        return 2.0 * (np.log(mu / y) + y / mu - 1.0)

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-300, None)
        phi = np.clip(phi, 1e-300, None)
        # Gamma(shape=1/phi, scale=mu*phi): sum over obs
        # log f = (1/phi - 1)*log(y) - y/(mu*phi) - (1/phi)*log(mu*phi) - log Gamma(1/phi)
        k = 1.0 / phi
        return (
            (k - 1.0) * np.log(y)
            - y * k / mu
            - k * np.log(mu)
            + k * np.log(k)
            - gammaln(k)
        )

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        return np.full_like(y, float(np.mean(y)))


# ---------------------------------------------------------------------------
# InverseGaussian
# ---------------------------------------------------------------------------

class InverseGaussian(Family):
    """
    Inverse Gaussian family. V(mu) = mu^3.

    Suitable for heavy-tailed severity where variance grows faster than Gamma.
    The log link is more natural than the canonical inverse-squared link in
    insurance pricing contexts.
    """

    _default_link = "log"

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu ** 3

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-300, None)
        return (y - mu) ** 2 / (mu ** 2 * y)

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        mu = np.clip(mu, 1e-300, None)
        phi = np.clip(phi, 1e-300, None)
        return (
            -0.5 * np.log(2.0 * np.pi * phi * y ** 3)
            - (y - mu) ** 2 / (2.0 * phi * mu ** 2 * y)
        )

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        y = np.clip(y, 1e-300, None)
        return np.full_like(y, float(np.mean(y)))


# ---------------------------------------------------------------------------
# Tweedie
# ---------------------------------------------------------------------------

class Tweedie(Family):
    """
    Tweedie family. V(mu) = mu^p, p in (1, 2).

    The compound Poisson-Gamma distribution: the only continuous distribution
    with point mass at zero in the exponential family. The workhorse for
    pure premium modelling in non-life insurance.

    p=1: Poisson; p=2: Gamma; p in (1,2): compound Poisson-Gamma.

    The dispersion submodel uses the saddlepoint approximation, which is
    asymptotically valid. Accuracy degrades when many responses are exactly
    zero.
    """

    _default_link = "log"

    def __init__(self, p: float = 1.5, link: str | None = None):
        if not (1.0 < p < 2.0):
            raise ValueError(
                f"Tweedie p must be in (1, 2) for compound Poisson-Gamma. Got p={p}."
            )
        self.p = float(p)
        super().__init__(link)

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu ** self.p

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        # Standard Tweedie unit deviance
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        p = self.p
        # For y > 0
        term1 = np.where(
            y > 0,
            y ** (2.0 - p) / ((1.0 - p) * (2.0 - p)),
            0.0,
        )
        term2 = -y * mu ** (1.0 - p) / (1.0 - p)
        term3 = mu ** (2.0 - p) / (2.0 - p)
        return 2.0 * (term1 + term2 + term3)

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        # Saddlepoint approximation (Dunn & Smyth 2005 series is more accurate
        # but complex; use saddlepoint for now).
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        phi = np.clip(phi, 1e-300, None)
        p = self.p
        d = self.deviance_resid(y, mu)
        # log-normalising constant for Tweedie (saddlepoint):
        # log f ~ -d/(2*phi) - 0.5*log(2*pi*phi*V(y)) for y > 0
        # For y = 0, log f = -mu^(2-p) / (phi*(2-p))
        ll = np.where(
            y > 0,
            -d / (2.0 * phi)
            - 0.5 * np.log(2.0 * np.pi * phi * np.clip(y, 1e-300, None) ** p),
            -mu ** (2.0 - p) / (phi * (2.0 - p)),
        )
        return ll

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        y_pos = y[y > 0]
        m = float(np.mean(y_pos)) if len(y_pos) > 0 else 1.0
        return np.full_like(y, m)

    def __repr__(self) -> str:
        return f"Tweedie(p={self.p}, link='{self._link_obj.name}')"


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

class Poisson(Family):
    """
    Poisson family. V(mu) = mu.

    For frequency modelling where overdispersion is present (phi > 1 indicates
    extra-Poisson variation). The standard GLM imposes phi=1 by assumption;
    DGLM relaxes this.

    The dispersion submodel is a saddlepoint approximation. It works well when
    expected counts are not too small (mu*phi >> 1 is the rough rule of thumb).
    """

    _default_link = "log"

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu.copy()

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        return 2.0 * np.where(
            y > 0,
            y * np.log(y / mu) - (y - mu),
            mu,
        )

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        # Quasi-Poisson: treat phi as dispersion parameter
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        phi = np.clip(phi, 1e-300, None)
        # Quasi log-likelihood proportional to true log-likelihood divided by phi
        # For LRT purposes: use scaled deviance -d/(2*phi)
        d = self.deviance_resid(y, mu)
        return -d / (2.0 * phi)

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        return np.full_like(y, max(float(np.mean(y)), 1e-3))


# ---------------------------------------------------------------------------
# Negative Binomial
# ---------------------------------------------------------------------------

class NegativeBinomial(Family):
    """
    Negative Binomial family. V(mu) = mu + alpha * mu^2.

    alpha (> 0) is the additional overdispersion parameter beyond the Poisson.
    phi in the DGLM acts as a further scale on top of the built-in NegBin
    variance — this is the quasi-NegBin interpretation.

    In practice for insurance frequency modelling you often want either NegBin
    with fixed alpha (estimated separately) or Poisson-DGLM. Use NegBin-DGLM
    when you have reason to believe the overdispersion structure itself varies
    by covariate.
    """

    _default_link = "log"

    def __init__(self, alpha: float = 1.0, link: str | None = None):
        if alpha <= 0:
            raise ValueError(f"NegativeBinomial alpha must be > 0. Got {alpha}.")
        self.alpha = float(alpha)
        super().__init__(link)

    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu + self.alpha * mu ** 2

    def deviance_resid(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        a = self.alpha
        r = 1.0 / a
        return 2.0 * np.where(
            y > 0,
            y * np.log(y / mu) - (y + r) * np.log((y + r) / (mu + r)),
            r * np.log((mu + r) / r),
        )

    def log_likelihood(
        self, y: np.ndarray, mu: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        y = np.clip(y, 0, None)
        mu = np.clip(mu, 1e-300, None)
        phi = np.clip(phi, 1e-300, None)
        # Quasi log-likelihood via scaled deviance
        d = self.deviance_resid(y, mu)
        return -d / (2.0 * phi)

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        return np.full_like(y, max(float(np.mean(y)), 1e-3))

    def __repr__(self) -> str:
        return f"NegativeBinomial(alpha={self.alpha}, link='{self._link_obj.name}')"
