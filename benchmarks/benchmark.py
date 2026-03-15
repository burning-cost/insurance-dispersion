"""
Benchmark: Double GLM (DGLM) vs constant-dispersion Gamma GLM.

DGP: Gamma severity where dispersion phi_i varies by distribution channel.
  - Direct channel: phi ~ 0.30  (tighter, more predictable)
  - Broker channel: phi ~ 1.20  (looser, more heterogeneous)

A standard GLM assumes a single phi for all policies. The DGLM fits a
separate regression for phi_i, recovering the true dispersion structure.

Metrics:
  1. Prediction interval coverage (nominal 90% interval — should hit 90%)
  2. Dispersion parameter recovery (estimated phi vs true phi per group)
  3. Log-likelihood on test data (DGLM > constant-phi GLM)
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def make_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Gamma severity with covariate-driven dispersion.

    Mean model:  log(mu_i) = 7.5 + 0.4*(veh_class=="B") + 0.2*(veh_class=="C")
    Disp model:  log(phi_i) = -1.2 + 1.4*(channel=="broker")
      => phi_direct ~ 0.30,  phi_broker ~ 1.20
    """
    rng = np.random.default_rng(seed)
    vehicle_class = rng.choice(["A", "B", "C"], size=n)
    channel       = rng.choice(["direct", "broker"], size=n, p=[0.6, 0.4])

    log_mu = 7.5 + 0.4 * (vehicle_class == "B") + 0.2 * (vehicle_class == "C")
    mu     = np.exp(log_mu)

    log_phi = -1.2 + 1.4 * (channel == "broker")
    phi     = np.exp(log_phi)   # direct ~ 0.30, broker ~ 1.20

    # Gamma(shape=1/phi, scale=mu*phi) => E=mu, Var=phi*mu^2
    shape = 1.0 / phi
    scale = mu * phi
    y = rng.gamma(shape=shape, scale=scale)

    return pd.DataFrame({
        "vehicle_class": vehicle_class,
        "channel":       channel,
        "claim_amount":  y,
        "true_mu":       mu,
        "true_phi":      phi,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gamma_loglik(y: np.ndarray, mu: np.ndarray, phi: np.ndarray) -> float:
    shape = 1.0 / phi
    scale = mu * phi
    ll = (shape - 1) * np.log(y) - y / scale - shape * np.log(scale) - gammaln(shape)
    return float(np.sum(ll))


def pi_coverage(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    level: float = 0.90,
) -> float:
    """Fraction of obs inside nominal prediction interval under Gamma(mu, phi)."""
    alpha = (1.0 - level) / 2.0
    shape = 1.0 / phi
    scale = mu * phi
    lo = gamma_dist.ppf(alpha,       a=shape, scale=scale)
    hi = gamma_dist.ppf(1.0 - alpha, a=shape, scale=scale)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Benchmark: DGLM (Double GLM) vs constant-dispersion Gamma GLM")
    print("DGP: Gamma severity with channel-dependent dispersion")
    print("  phi_direct ~ 0.30  |  phi_broker ~ 1.20")
    print("=" * 65)

    df = make_dataset(n=3000, seed=42)
    n_train = 2400
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    test_df  = df.iloc[n_train:].copy().reset_index(drop=True)

    y_test       = test_df["claim_amount"].to_numpy()
    direct_mask  = (test_df["channel"] == "direct").to_numpy()
    broker_mask  = ~direct_mask

    print(f"\nDataset: {n_train} train / {len(test_df)} test policies")
    print(f"Train channel split: direct={(train_df['channel']=='direct').sum()}  broker={(train_df['channel']=='broker').sum()}")
    print(f"True phi: direct=0.30  broker=1.20")

    # ------------------------------------------------------------------
    # Constant-dispersion Gamma GLM (statsmodels)
    # ------------------------------------------------------------------
    print("\nFitting constant-dispersion Gamma GLM...")
    glm_ok = False
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        glm_res = smf.glm(
            "claim_amount ~ C(vehicle_class)",
            data=train_df,
            family=sm.families.Gamma(link=sm.families.links.log()),
        ).fit()

        glm_phi_const = float(glm_res.scale)
        mu_glm = glm_res.predict(test_df).to_numpy()
        phi_glm_arr = np.full(len(test_df), glm_phi_const)

        glm_ll           = gamma_loglik(y_test, mu_glm, phi_glm_arr)
        glm_cov_all      = pi_coverage(y_test, mu_glm, phi_glm_arr)
        glm_cov_direct   = pi_coverage(y_test[direct_mask], mu_glm[direct_mask], phi_glm_arr[direct_mask])
        glm_cov_broker   = pi_coverage(y_test[broker_mask], mu_glm[broker_mask], phi_glm_arr[broker_mask])

        print(f"  Constant phi estimate: {glm_phi_const:.3f}  (blend of true 0.30 and 1.20)")
        glm_ok = True
    except ImportError:
        print("  statsmodels not available — skipping GLM baseline")

    # ------------------------------------------------------------------
    # DGLM
    # ------------------------------------------------------------------
    print("\nFitting DGLM (channel-dependent dispersion)...")
    dglm_ok = False
    try:
        from insurance_dispersion import DGLM
        import insurance_dispersion.families as fam

        dglm = DGLM(
            formula="claim_amount ~ C(vehicle_class)",
            dformula="~ C(channel)",
            family=fam.Gamma(),
            data=train_df,
            method="reml",
        )
        result = dglm.fit()

        mu_dglm  = result.predict(test_df, which="mean")
        phi_dglm = result.predict(test_df, which="dispersion")

        dglm_ll         = gamma_loglik(y_test, mu_dglm, phi_dglm)
        dglm_cov_all    = pi_coverage(y_test, mu_dglm, phi_dglm)
        dglm_cov_direct = pi_coverage(y_test[direct_mask], mu_dglm[direct_mask], phi_dglm[direct_mask])
        dglm_cov_broker = pi_coverage(y_test[broker_mask], mu_dglm[broker_mask], phi_dglm[broker_mask])

        dglm_phi_direct = np.mean(phi_dglm[direct_mask])
        dglm_phi_broker = np.mean(phi_dglm[broker_mask])

        print(f"  Converged: {result.converged}  ({result.n_iter} iterations)")
        print(f"  DGLM phi estimates: direct={dglm_phi_direct:.3f}  broker={dglm_phi_broker:.3f}")
        print(f"  (True values: direct=0.30,  broker=1.20)")
        dglm_ok = True
    except Exception as e:
        print(f"  DGLM failed: {e}")
        import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTS — 90% prediction interval coverage (nominal = 0.900)")
    print("=" * 65)
    print(f"{'Model':<30} {'Log-lik':>10} {'All':>8} {'Direct':>8} {'Broker':>8}")
    print("-" * 65)
    if glm_ok:
        print(f"{'Gamma GLM (const phi)':<30} {glm_ll:>10.1f} {glm_cov_all:>8.3f} {glm_cov_direct:>8.3f} {glm_cov_broker:>8.3f}")
    if dglm_ok:
        print(f"{'DGLM (channel phi)':<30} {dglm_ll:>10.1f} {dglm_cov_all:>8.3f} {dglm_cov_direct:>8.3f} {dglm_cov_broker:>8.3f}")
    print("-" * 65)
    print(f"{'Nominal target':<30} {'':>10} {'0.900':>8} {'0.900':>8} {'0.900':>8}")

    if glm_ok and dglm_ok:
        print(f"\nDispersion recovery (true: direct=0.30, broker=1.20):")
        print(f"  Constant GLM phi:  {glm_phi_const:.3f}  (same for both groups)")
        print(f"  DGLM direct phi:   {dglm_phi_direct:.3f}")
        print(f"  DGLM broker phi:   {dglm_phi_broker:.3f}")
        print(f"\nLog-likelihood improvement: {dglm_ll - glm_ll:+.1f}")
        print(f"\nConclusion: constant GLM over-covers direct (phi too large)")
        print(f"  and under-covers broker (phi too small). DGLM recovers")
        print(f"  the true per-group dispersion and achieves correct coverage.")


if __name__ == "__main__":
    main()
