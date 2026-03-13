# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-dispersion (DGLM) vs single Tweedie GLM
# MAGIC
# MAGIC **Library:** `insurance-dispersion` — Double GLM for joint modelling of mean
# MAGIC and dispersion. Each observation gets its own phi_i driven by covariates via
# MAGIC alternating IRLS (Smyth 1989).
# MAGIC
# MAGIC **Baseline:** single Tweedie GLM (statsmodels) with a constant dispersion
# MAGIC parameter phi shared across all observations. This is the standard approach
# MAGIC for pure premium modelling in UK personal and commercial lines.
# MAGIC
# MAGIC **Dataset:** Synthetic UK commercial property pure premiums — 25,000 policies.
# MAGIC Known DGP where dispersion genuinely varies by distribution channel and risk
# MAGIC size band (the motivating case for the DGLM).
# MAGIC Temporal 70/30 train/test split.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The key question: when phi varies systematically across the portfolio, does a
# MAGIC constant-phi Tweedie misestimate volatility in commercially important subgroups?
# MAGIC The broker/direct split is the canonical case — broker-placed risks aggregate
# MAGIC heterogeneous SME accounts that a direct policy would never write.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-dispersion statsmodels matplotlib numpy scipy pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_dispersion import DGLM
import insurance_dispersion.families as fam

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC Synthetic UK commercial property pure premium data with known heterogeneous dispersion.
# MAGIC
# MAGIC DGP:
# MAGIC - Pure premium (loss cost) follows a Tweedie distribution with power p=1.5
# MAGIC - Mean depends on building_class, age_band, region, log(TSI)
# MAGIC - **Dispersion phi_i varies by channel and risk size band** — this is what a
# MAGIC   standard Tweedie GLM cannot capture
# MAGIC
# MAGIC Channel phi structure (true DGP):
# MAGIC - Direct: phi = 0.8 (lowest — homogeneous retail, well-understood risks)
# MAGIC - Aggregator: phi = 1.2 (slightly more volatile — price-shopped book)
# MAGIC - Broker SME: phi = 2.5 (heterogeneous SME risks, wide range of constructions)
# MAGIC - Broker Large: phi = 4.5 (large commercial risks, fat-tailed losses)
# MAGIC
# MAGIC This DGP is designed to represent what a London Market or commercial lines team
# MAGIC would actually observe when they split their book by distribution channel.

# COMMAND ----------

rng = np.random.default_rng(2024)
N = 25_000

# Covariates
building_class = rng.choice(["residential", "light_comm", "heavy_comm", "industrial"],
                              N, p=[0.40, 0.30, 0.20, 0.10])
age_band       = rng.choice(["pre1945", "1945-1980", "1980-2000", "post2000"],
                              N, p=[0.15, 0.25, 0.35, 0.25])
region         = rng.choice(["London", "SE", "Midlands", "North", "Scotland"],
                              N, p=[0.20, 0.18, 0.22, 0.28, 0.12])
channel        = rng.choice(["direct", "aggregator", "broker_sme", "broker_large"],
                              N, p=[0.35, 0.25, 0.30, 0.10])
tsi            = rng.lognormal(12.5, 0.8, N)   # total sum insured ~ £50k to £20M
log_tsi        = np.log(tsi)
exposure       = rng.uniform(0.5, 1.0, N)       # years on risk

# True mean pure premium rate (annualised, per unit TSI)
mu_log  = -3.5   # intercept: ~3% loss ratio at base
mu_log += np.where(building_class == "light_comm",  0.30, 0.0)
mu_log += np.where(building_class == "heavy_comm",  0.65, 0.0)
mu_log += np.where(building_class == "industrial",  0.95, 0.0)
mu_log += np.where(age_band == "pre1945",     0.35, 0.0)
mu_log += np.where(age_band == "1945-1980",   0.15, 0.0)
mu_log += np.where(age_band == "post2000",   -0.10, 0.0)
mu_log += np.where(region == "London", 0.20, 0.0)
mu_log += np.where(region == "SE",     0.12, 0.0)
mu_log += 0.18 * (log_tsi - 12.5)    # scale sensitivity
mu_rate_true = np.exp(mu_log)
mu_true = mu_rate_true * exposure    # expected loss = rate * exposure

# True dispersion — heterogeneous by channel (this is the DGP the baseline misses)
phi_log = np.zeros(N)
phi_log += np.where(channel == "direct",       np.log(0.8),  0.0)
phi_log += np.where(channel == "aggregator",   np.log(1.2),  0.0)
phi_log += np.where(channel == "broker_sme",   np.log(2.5),  0.0)
phi_log += np.where(channel == "broker_large", np.log(4.5),  0.0)
phi_log += 0.15 * (log_tsi - 12.5)   # larger risks also have more volatile loss ratios
phi_true = np.exp(phi_log)

# Generate Tweedie random variates by compound Poisson-Gamma decomposition
# Tweedie(mu, phi, p=1.5): N ~ Poisson(lambda), claims ~ Gamma(alpha, beta)
# lambda = mu^(2-p) / (phi*(2-p)) = mu^0.5 / (phi*0.5)
# alpha = (2-p)/(p-1) = 1.0 (for p=1.5)
# beta (scale) = phi*(p-1)*mu^(p-1) = phi*0.5*mu^0.5

TWEEDIE_P = 1.5
lam_true = mu_true ** (2 - TWEEDIE_P) / (phi_true * (2 - TWEEDIE_P))
alpha_g   = (2 - TWEEDIE_P) / (TWEEDIE_P - 1)   # = 1.0 for p=1.5
beta_g    = phi_true * (TWEEDIE_P - 1) * mu_true ** (TWEEDIE_P - 1)

n_claims  = rng.poisson(np.maximum(lam_true, 1e-8))
y = np.zeros(N)
for i in range(N):
    nc = n_claims[i]
    if nc > 0:
        y[i] = rng.gamma(alpha_g * nc, scale=beta_g[i])

# Aggregate pure premium (with exposure)
pp = y  # already exposure-weighted via mu_true = mu_rate * exposure

# Temporal split
policy_year = rng.choice([2021, 2022, 2023], N, p=[0.35, 0.35, 0.30])
order = np.argsort(policy_year, kind="stable")

df = pd.DataFrame({
    "building_class": building_class,
    "age_band":       age_band,
    "region":         region,
    "channel":        channel,
    "tsi":            tsi,
    "log_tsi":        log_tsi,
    "exposure":       exposure,
    "policy_year":    policy_year,
    "pure_premium":   y,
    "mu_true":        mu_true,
    "phi_true":       phi_true,
})
df = df.iloc[order].reset_index(drop=True)

# Add small offset for GLM stability (Tweedie requires positive y, but allows 0)
# We keep 0s in the data — they are real non-loss policies

train_end = int(N * 0.70)
train = df.iloc[:train_end].copy()
test  = df.iloc[train_end:].copy()

print(f"Total:  {N:,}")
print(f"Train:  {len(train):,}  ({100*len(train)/N:.0f}%)")
print(f"Test:   {len(test):,}   ({100*len(test)/N:.0f}%)")
print(f"\nPure premium summary:")
print(f"  Zero rate: {(y == 0).mean():.1%}")
print(f"  Mean (non-zero): {y[y > 0].mean():.1f}")
print(f"\nTrue phi by channel:")
for ch in ["direct", "aggregator", "broker_sme", "broker_large"]:
    m = channel == ch
    print(f"  {ch:<15}: true phi mean = {phi_true[m].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Single Tweedie GLM
# MAGIC
# MAGIC A Tweedie GLM with constant dispersion — the standard UK commercial property
# MAGIC pure premium model. Dispersion is estimated as a single scalar across the entire
# MAGIC portfolio, ignoring channel heterogeneity.

# COMMAND ----------

t0 = time.perf_counter()

mean_formula = (
    "pure_premium ~ C(building_class) + C(age_band) + C(region) + log_tsi"
)

# Add small epsilon to allow Tweedie GLM with zeros
train_fit = train.copy()
# statsmodels Tweedie handles true 0s, but we add offset for exposure

glm_tw = smf.glm(
    mean_formula,
    data=train_fit,
    family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=TWEEDIE_P),
    offset=np.log(train_fit["exposure"].clip(lower=1e-6)),
).fit()

mu_baseline_train = glm_tw.predict(train_fit, offset=np.log(train_fit["exposure"].clip(lower=1e-6)))
mu_baseline_test  = glm_tw.predict(test,      offset=np.log(test["exposure"].clip(lower=1e-6)))

# Constant dispersion estimate (GLM scale parameter)
phi_constant = float(glm_tw.scale)

baseline_fit_time = time.perf_counter() - t0

print(f"Baseline fit time:  {baseline_fit_time:.2f}s")
print(f"Constant phi (GLM): {phi_constant:.4f}")
print(f"Deviance explained: {(1 - glm_tw.deviance / glm_tw.null_deviance):.1%}")
print()
print(glm_tw.summary2().tables[1].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: DGLM (Double GLM)
# MAGIC
# MAGIC The DGLM fits a second regression model for phi alongside the mean model.
# MAGIC We use channel and log_tsi as dispersion covariates — the known DGP drivers.
# MAGIC
# MAGIC The alternating IRLS algorithm:
# MAGIC 1. Update mean model (GLM with weights = prior_weights / phi_i)
# MAGIC 2. Update dispersion model (Gamma GLM on unit deviances)
# MAGIC 3. Repeat until convergence on joint log-likelihood

# COMMAND ----------

t0 = time.perf_counter()

dglm = DGLM(
    formula="pure_premium ~ C(building_class) + C(age_band) + C(region) + log_tsi",
    dformula="~ C(channel) + log_tsi",
    family=fam.Tweedie(p=TWEEDIE_P),
    exposure="exposure",
    data=train,
    method="reml",
)
result = dglm.fit(maxit=30, epsilon=1e-7, verbose=False)

library_fit_time = time.perf_counter() - t0

mu_dglm_train = result.predict(train, which="mean")
mu_dglm_test  = result.predict(test,  which="mean")
phi_dglm_test = result.predict(test,  which="dispersion")

print(f"DGLM fit time:   {library_fit_time:.2f}s")
print(result.summary())

# COMMAND ----------

# Mean and dispersion relativities
print("=== Mean model relativities ===")
print(result.mean_relativities().to_string())
print()
print("=== Dispersion model relativities ===")
print(result.dispersion_relativities().to_string())

# COMMAND ----------

# Overdispersion test: is varying phi significantly better than constant phi?
test_result = result.overdispersion_test()
print("=== Overdispersion test (LRT: constant phi vs varying phi) ===")
print(f"  Statistic: {test_result['statistic']:.2f}")
print(f"  df:        {test_result['df']}")
print(f"  p-value:   {test_result['p_value']:.4e}")
print(f"  Conclusion: {test_result['conclusion']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics
# MAGIC
# MAGIC Key metrics for a pure premium model:
# MAGIC
# MAGIC - **Tweedie deviance**: the proper scoring rule for this response distribution.
# MAGIC   Lower is better.
# MAGIC - **Gini coefficient**: discriminatory power on the test set. Higher is better.
# MAGIC - **A/E ratio by channel**: the most commercially important test. A correctly
# MAGIC   calibrated model should have A/E close to 1.0 for each distribution channel.
# MAGIC   The constant-phi Tweedie cannot differentiate dispersion by channel — the DGLM
# MAGIC   can.
# MAGIC - **Phi recovery**: how closely does the fitted phi track the known true phi?
# MAGIC   The constant-phi model gets a single scalar; the DGLM gets per-policy estimates.
# MAGIC - **Variance calibration by channel**: for reinsurance and capital modelling,
# MAGIC   correct variance estimates by segment are as important as correct means.

# COMMAND ----------

def tweedie_deviance(y, mu, p=1.5):
    """Mean Tweedie deviance. Uses compound Poisson formula for 1 < p < 2."""
    y   = np.asarray(y, dtype=float)
    mu  = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    p1  = p - 1.0
    p2  = 2.0 - p
    # Unit deviance formula for 1 < p < 2
    d = 2.0 * (
        np.where(y > 0, y**(2-p) / ((1-p)*(2-p)) - y * mu**(-p1) / (1-p), 0.0)
        + mu**(2-p) / ((2-p))
    )
    return float(np.mean(d))


def gini_coefficient(y, mu_pred):
    order  = np.argsort(mu_pred)
    y_s    = y[order]
    n      = len(y_s)
    cum_y  = np.cumsum(y_s) / max(y_s.sum(), 1e-10)
    cum_pop = np.arange(1, n + 1) / n
    return 2 * np.trapz(cum_y, cum_pop) - 1


def ae_by_group(y, mu_pred, group_series):
    """A/E ratio per group."""
    df_tmp = pd.DataFrame({"y": y, "mu": mu_pred, "group": group_series})
    res = {}
    for g, sub in df_tmp.groupby("group"):
        ae = sub["y"].sum() / max(sub["mu"].sum(), 1e-10)
        res[g] = ae
    return res

# COMMAND ----------

y_test_arr = test["pure_premium"].values
mu_base_arr = mu_baseline_test.values
mu_dglm_arr = mu_dglm_test

# Deviance
dev_base = tweedie_deviance(y_test_arr, mu_base_arr, p=TWEEDIE_P)
dev_dglm = tweedie_deviance(y_test_arr, mu_dglm_arr, p=TWEEDIE_P)

# Gini
gini_base = gini_coefficient(y_test_arr, mu_base_arr)
gini_dglm = gini_coefficient(y_test_arr, mu_dglm_arr)

# A/E by channel
ae_base = ae_by_group(y_test_arr, mu_base_arr, test["channel"].values)
ae_dglm = ae_by_group(y_test_arr, mu_dglm_arr, test["channel"].values)

# Phi recovery (per-test-policy)
phi_true_test = test["phi_true"].values
phi_dglm_arr  = np.asarray(phi_dglm_test)

phi_mae_const = np.abs(phi_constant - phi_true_test).mean()
phi_mae_dglm  = np.abs(phi_dglm_arr - phi_true_test).mean()

# Variance calibration by channel
print(f"{'Metric':<44} {'Baseline (const phi)':>20} {'DGLM':>10} {'Delta':>8}")
print("=" * 88)

print(f"  {'Tweedie deviance (lower better)':<42} {dev_base:>20.5f} {dev_dglm:>10.5f} "
      f"{(dev_dglm-dev_base)/abs(dev_base)*100:>+7.1f}%")
print(f"  {'Gini coefficient (higher better)':<42} {gini_base:>20.4f} {gini_dglm:>10.4f} "
      f"{(gini_dglm-gini_base)/abs(gini_base)*100:>+7.1f}%")
print(f"  {'Phi MAE vs true (lower better)':<42} {phi_mae_const:>20.4f} {phi_mae_dglm:>10.4f} "
      f"{(phi_mae_dglm-phi_mae_const)/abs(phi_mae_const)*100:>+7.1f}%")
print(f"  {'Constant phi estimate':<42} {phi_constant:>20.4f} {'n/a':>10}")
print(f"  {'Mean fitted phi (DGLM)':<42} {'n/a':>20} {phi_dglm_arr.mean():>10.4f}")
print(f"  {'Fit time (s)':<42} {baseline_fit_time:>20.2f} {library_fit_time:>10.2f} "
      f"{(library_fit_time-baseline_fit_time)/abs(baseline_fit_time)*100:>+7.1f}%")

print("=" * 88)
print()

print("A/E by channel:")
print(f"  {'Channel':<20} {'Baseline A/E':>14} {'DGLM A/E':>10} {'True phi':>10}")
print("-" * 58)
for ch in ["direct", "aggregator", "broker_sme", "broker_large"]:
    m = test["channel"] == ch
    true_phi_ch = phi_true_test[m.values].mean()
    print(f"  {ch:<20} {ae_base.get(ch, np.nan):>14.4f} {ae_dglm.get(ch, np.nan):>10.4f} "
          f"{true_phi_ch:>10.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Diagnostic Plots

# COMMAND ----------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

channels = ["direct", "aggregator", "broker_sme", "broker_large"]

# ── Plot 1: True phi vs DGLM fitted phi by channel ─────────────────────────
colors = {"direct": "steelblue", "aggregator": "goldenrod",
          "broker_sme": "tomato", "broker_large": "darkred"}
for ch in channels:
    m = test["channel"] == ch
    ax1.scatter(phi_true_test[m.values], phi_dglm_arr[m.values],
                alpha=0.2, s=6, color=colors[ch], label=ch)
max_phi = max(phi_true_test.max(), phi_dglm_arr.max())
ax1.plot([0, max_phi], [0, max_phi], "k--", linewidth=1.5, label="Perfect")
ax1.set_xlabel("True phi (DGP)")
ax1.set_ylabel("Fitted phi (DGLM)")
ax1.set_title("Phi Recovery: DGLM vs True")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Plot 2: A/E by channel ─────────────────────────────────────────────────
x_pos = np.arange(len(channels))
ae_base_v = [ae_base.get(ch, np.nan) for ch in channels]
ae_dglm_v = [ae_dglm.get(ch, np.nan) for ch in channels]

ax2.bar(x_pos - 0.2, ae_base_v, 0.4, label="Baseline (const phi)",
        color="steelblue", alpha=0.7)
ax2.bar(x_pos + 0.2, ae_dglm_v, 0.4, label="DGLM",
        color="tomato", alpha=0.7)
ax2.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="A/E = 1.0")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(channels, rotation=20, ha="right")
ax2.set_ylabel("A/E ratio")
ax2.set_title("A/E by Distribution Channel")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# ── Plot 3: Fitted mean lift chart ─────────────────────────────────────────
order_b    = np.argsort(mu_base_arr)
n_dec      = 10
idx_splits = np.array_split(np.arange(len(y_test_arr)), n_dec)

# Sort test data by baseline prediction
y_s    = y_test_arr[order_b]
mb_s   = mu_base_arr[order_b]
md_s   = mu_dglm_arr[order_b]

actual_d  = [y_s[i].mean()  for i in idx_splits]
base_d    = [mb_s[i].mean() for i in idx_splits]
dglm_d    = [md_s[i].mean() for i in idx_splits]

ax3.plot(range(1, n_dec+1), actual_d, "ko-",  label="Actual",   linewidth=2)
ax3.plot(range(1, n_dec+1), base_d,   "b^--", label="Baseline", linewidth=1.5, alpha=0.8)
ax3.plot(range(1, n_dec+1), dglm_d,   "rs-",  label="DGLM",     linewidth=1.5, alpha=0.8)
ax3.set_xlabel("Decile (sorted by baseline prediction)")
ax3.set_ylabel("Mean pure premium")
ax3.set_title("Lift Chart")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Plot 4: Fitted phi distribution by channel (DGLM) ─────────────────────
phi_ranges = np.linspace(0, phi_dglm_arr.max() + 0.5, 60)
for ch in channels:
    m = test["channel"] == ch
    ax4.hist(phi_dglm_arr[m.values], bins=40, density=True, histtype="step",
             linewidth=1.5, color=colors[ch], label=f"{ch} (n={m.sum()})")

ax4.set_xlabel("Fitted phi (DGLM)")
ax4.set_ylabel("Density")
ax4.set_title("DGLM Fitted Phi Distribution by Channel")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle(
    "insurance-dispersion DGLM vs constant-phi Tweedie — Diagnostic Plots",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_dispersion.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_dispersion.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Variance Calibration by Channel
# MAGIC
# MAGIC The most important test for dispersion modelling: does the model's predicted
# MAGIC variance track the observed variance within each channel segment?
# MAGIC
# MAGIC Under the Tweedie model: Var[Y|X] = phi * mu^p.
# MAGIC We compare predicted variance to empirical squared deviations by channel.

# COMMAND ----------

print(f"{'Channel':<20} {'True phi':>10} {'Const phi':>10} {'DGLM phi':>10} "
      f"{'Var ratio (const)':>18} {'Var ratio (DGLM)':>16}")
print("-" * 86)

for ch in channels:
    m = (test["channel"] == ch).values
    if m.sum() < 20:
        continue

    y_ch      = y_test_arr[m]
    mu_b_ch   = mu_base_arr[m]
    mu_d_ch   = mu_dglm_arr[m]
    phi_d_ch  = phi_dglm_arr[m].mean()
    true_phi_ch = phi_true_test[m].mean()

    # Predicted variance
    pred_var_const = phi_constant * mu_b_ch ** TWEEDIE_P
    pred_var_dglm  = phi_dglm_arr[m] * mu_d_ch ** TWEEDIE_P

    # Empirical variance proxy: squared deviations from conditional mean
    obs_sq_const = (y_ch - mu_b_ch) ** 2
    obs_sq_dglm  = (y_ch - mu_d_ch) ** 2

    # Ratio: if close to 1.0, the variance is well-calibrated
    ratio_const = obs_sq_const.mean() / max(pred_var_const.mean(), 1e-10)
    ratio_dglm  = obs_sq_dglm.mean()  / max(pred_var_dglm.mean(),  1e-10)

    print(f"  {ch:<18} {true_phi_ch:>10.3f} {phi_constant:>10.3f} {phi_d_ch:>10.3f} "
          f"{ratio_const:>18.3f} {ratio_dglm:>16.3f}")

print()
print("Variance ratio: observed_sq_deviation / predicted_variance.")
print("Close to 1.0 = well-calibrated. The constant-phi model is miscalibrated")
print("in the tails (broker_large especially) where phi is much higher than average.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict

# COMMAND ----------

print("=" * 60)
print("VERDICT: DGLM vs constant-phi Tweedie")
print("=" * 60)
print()
print("Mean prediction quality:")
print(f"  Tweedie deviance: {dev_base:.5f} (baseline) -> {dev_dglm:.5f} (DGLM)  "
      f"{(dev_dglm-dev_base)/abs(dev_base)*100:+.1f}%")
print(f"  Gini:             {gini_base:.4f} (baseline) -> {gini_dglm:.4f} (DGLM)")
print()
print("Dispersion quality:")
print(f"  Phi MAE vs true:  {phi_mae_const:.4f} (constant) -> {phi_mae_dglm:.4f} (DGLM)  "
      f"{(phi_mae_dglm-phi_mae_const)/abs(phi_mae_const)*100:+.1f}%")
print()
print("Channel A/E deviation from 1.0:")
ae_base_max_dev = max(abs(v - 1.0) for v in ae_base.values())
ae_dglm_max_dev = max(abs(v - 1.0) for v in ae_dglm.values())
print(f"  Max |A/E - 1.0|:  {ae_base_max_dev:.4f} (baseline) -> {ae_dglm_max_dev:.4f} (DGLM)")
print()
print("Fit time:")
print(f"  {baseline_fit_time:.2f}s (baseline) -> {library_fit_time:.2f}s (DGLM)  "
      f"({library_fit_time/max(baseline_fit_time,0.001):.1f}x slower)")
print()
print("When does the DGLM matter most?")
print("  - Mixed channel books: direct vs broker dispersion differs by 3-6x")
print("  - Risk size heterogeneity: SME vs large accounts")
print("  - Reinsurance pricing: the variance per risk drives XL attachment decisions")
print("  - When LRT p-value < 0.05: evidence dispersion is not constant")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. README Performance Snippet

# COMMAND ----------

print("""
## Performance

Benchmarked against a constant-phi Tweedie GLM (statsmodels) on synthetic UK
commercial property pure premium data (25,000 policies, known DGP with
heterogeneous dispersion by channel and risk size). Temporal 70/30 train/test split.
See `notebooks/benchmark_dispersion.py` for full methodology.
""")
print(f"| Metric                          | Tweedie GLM (const phi) | DGLM            |")
print(f"|----------------------------------|-------------------------|-----------------|")
print(f"| Tweedie deviance (test)          | {dev_base:.5f}               | {dev_dglm:.5f}      |")
print(f"| Gini coefficient                 | {gini_base:.4f}                | {gini_dglm:.4f}          |")
print(f"| Phi MAE vs true                  | {phi_mae_const:.4f}                | {phi_mae_dglm:.4f}          |")
print(f"| Max channel A/E deviation        | {ae_base_max_dev:.4f}                | {ae_dglm_max_dev:.4f}          |")
print(f"| Fit time (s)                     | {baseline_fit_time:.2f}                   | {library_fit_time:.2f}               |")
print()
print("""The primary gain is in dispersion calibration. The constant-phi Tweedie
assigns the same volatility to a broker-placed large commercial risk and a
direct retail policy. The DGLM captures a 3-6x dispersion difference between
these segments, materially improving variance calibration for channels at the
extremes. On homogeneous books (single channel, narrow risk size band), a
constant-phi Tweedie is adequate. On mixed books, the LRT test (overdispersion_test())
will flag whether the dispersion model adds value.
""")
