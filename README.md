# insurance-dispersion

[![PyPI](https://img.shields.io/pypi/v/insurance-dispersion)](https://pypi.org/project/insurance-dispersion/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-dispersion)](https://pypi.org/project/insurance-dispersion/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()


Double GLM (DGLM) for joint modelling of mean and dispersion in non-life insurance pricing.

## The problem

Standard GLMs assume a single scalar dispersion parameter phi shared across all observations. For a Gamma severity model, that means your fleet broker policy and your personal lines online policy are assumed to have identical volatility around the fitted mean. That assumption is almost always wrong.

Dispersion varies systematically with the same risk factors that drive the mean — and often with different factors entirely. Broker-sourced business tends to be more volatile (larger phi) because brokers aggregate heterogeneous risks. Fleet accounts have more predictable frequencies. High-limit policies show fat-tailed severity that the Gamma captures poorly with a flat dispersion assumption.

The Double GLM (Smyth 1989) solves this by adding a second regression model for phi:

```
Mean submodel:        g(mu_i)  = x_i^T beta    [standard GLM]
Dispersion submodel:  h(phi_i) = z_i^T alpha   [new: each obs gets its own phi]

Var[Y_i] = phi_i * V(mu_i)
```

This matters for:
- **Risk-differentiated pricing**: your pure premium estimate is mu_i, but quoting confidence depends on phi_i
- **Reinsurance pricing**: the tail risk on a policy is driven by both mu_i and phi_i
- **Model validation**: a well-specified mean model with poor dispersion fit still mispredicts volatility
- **Credibility**: low-phi risks can be priced more confidently than high-phi risks

## Installation

```bash
pip install insurance-dispersion
```

Or from source:

```bash
git clone https://github.com/burning-cost/insurance-dispersion
cd insurance-dispersion
uv pip install -e .
```

## Quick start

```python
import numpy as np
import pandas as pd
from insurance_dispersion import DGLM
import insurance_dispersion.families as fam

# Synthetic claim severity data
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "vehicle_class":  rng.choice(["A", "B", "C"], size=n),
    "age_band":       rng.choice(["17-24", "25-35", "36-60"], size=n),
    "vehicle_value":  rng.uniform(5000, 40000, size=n),
    "channel":        rng.choice(["direct", "broker"], size=n),
    "limit_band":     rng.choice(["50k", "100k", "250k"], size=n),
    "earned_premium": rng.uniform(0.5, 1.0, size=n),
})
df["claim_amount"] = rng.gamma(shape=2.0, scale=1500.0, size=n)

# Fit a Gamma DGLM for claim severity
# Mean model: severity depends on vehicle class and age band
# Dispersion model: volatility depends on distribution channel and limit band
model = DGLM(
    formula="claim_amount ~ C(vehicle_class) + C(age_band) + log(vehicle_value)",
    dformula="~ C(channel) + C(limit_band)",
    family=fam.Gamma(),
    data=df,
    exposure="earned_premium",  # log-offset in mean only
    method="reml",              # REML correction (recommended)
)

result = model.fit()
print(result.summary())
```

Output:
```
Double GLM (DGLM) Results
============================================================
Family:      Gamma(link='log')
Method:      REML
Observations:500
Converged:   True (after 8 iterations)
Log-lik:     -4182.3521
AIC:         8398.7042

Mean Submodel Coefficients:
------------------------------------------------------------
                            coef  exp_coef    se       z  p_value
Intercept               2.1543    8.6224  0.0321  67.12    0.0000
C(vehicle_class)[T.B]   0.1823    1.1999  0.0211   8.64    0.0000
...

Dispersion Submodel Coefficients:
------------------------------------------------------------
                          coef  exp_coef    se       z  p_value
Intercept             -0.8234    0.4390  0.0412 -19.99    0.0000
C(channel)[T.broker]   0.6112    1.8426  0.0518  11.80    0.0000
...
```

## Factor tables

```python
# Mean relativities: exp(beta) for each level vs. base
mean_rel = result.mean_relativities()
print(mean_rel[["exp_coef", "se", "p_value"]])

# Dispersion relativities: exp(alpha)
# Broker channel has 1.84x the dispersion of direct channel
disp_rel = result.dispersion_relativities()
print(disp_rel[["exp_coef", "se", "p_value"]])
```

## Predictions

```python
new_risk = pd.DataFrame({
    "vehicle_class": ["A", "B"],
    "age_band": ["25-35", "17-24"],
    "vehicle_value": [15000, 8000],
    "channel": ["direct", "broker"],
    "limit_band": ["100k", "50k"],
    "earned_premium": [1.0, 1.0],
})

# Expected severity
mu_pred = result.predict(new_risk, which="mean")

# Observation-level dispersion
phi_pred = result.predict(new_risk, which="dispersion")

# Predicted variance = phi_i * V(mu_i) = phi_i * mu_i^2 (Gamma)
var_pred = result.predict(new_risk, which="variance")
```

## Overdispersion test

```python
# Likelihood ratio test: constant phi vs. phi = f(channel, limit_band)
test = result.overdispersion_test()
print(f"LRT statistic: {test['statistic']:.2f}")
print(f"df: {test['df']}")
print(f"p-value: {test['p_value']:.4f}")
print(test["conclusion"])
```

## Diagnostics

```python
from insurance_dispersion import diagnostics

# Residuals
pearson_r = diagnostics.pearson_residuals(result)
deviance_r = diagnostics.deviance_residuals(result)
qr = diagnostics.quantile_residuals(result)  # ~ N(0,1) under true model

# QQ plot data
qq = diagnostics.qq_plot_data(result)
import matplotlib.pyplot as plt
plt.scatter(qq["theoretical"], qq["observed"], alpha=0.3, s=10)
plt.plot([-3, 3], [-3, 3], "r--")
plt.xlabel("N(0,1) quantiles")
plt.ylabel("Observed quantile residuals")

# Dispersion diagnostic
diag = diagnostics.dispersion_diagnostic(result)
plt.scatter(diag["fitted_phi"], diag["scaled_deviance"], alpha=0.2, s=8)
plt.axhline(1.0, color="red", linestyle="--")  # E[delta_i] = 1 under model
plt.xlabel("Fitted phi")
plt.ylabel("Scaled unit deviance")
```

## Supported families

| Family | Default link | Use case |
|--------|-------------|----------|
| `Gamma()` | log | Claim severity |
| `InverseGaussian()` | log | Heavy-tail severity |
| `Tweedie(p=1.5)` | log | Pure premium (compound Poisson-Gamma) |
| `Gaussian()` | identity | Reserve amounts, Gaussian responses |
| `Poisson()` | log | Claim frequency (extra-Poisson variation) |
| `NegativeBinomial(alpha=1.0)` | log | Overdispersed frequency |

## Algorithm

Alternating IRLS (Smyth 1989, Smyth & Verbyla 1999):

1. Initialise mu from intercept-only GLM, phi = 1
2. **Mean step**: IRLS for GLM(y ~ X, family, weights = prior_weights / phi_i)
3. **Dispersion step**: compute unit deviances d_i; fit Gamma GLM on delta_i = d_i / phi_i with log link
4. **REML correction** (method='reml'): subtract hat-matrix diagonal from delta_i before dispersion fit. Recommended when the mean model has many parameters.
5. Check convergence: relative change in -2*loglik < epsilon
6. Repeat until convergence or maxit reached

Pure numpy/scipy. No ML frameworks, no statsmodels dependency.

## Design choices

**formulaic not patsy**: patsy is unmaintained. formulaic has an active development community, cleaner model matrix schemas for prediction on new data, and better handling of interactions and transformations.

**method='reml' default**: the REML correction removes the contribution of estimating beta from the dispersion score. With even 10 mean parameters in a dataset of 500 observations this makes a material difference to the dispersion estimates. The correction is cheap (hat diagonal via QR) and almost always helps.

**Exposure on mean only**: log(exposure) enters as an offset in the mean linear predictor. Dispersion phi_i is per-unit-exposure: a 6-month policy has the same dispersion per claim as a 12-month policy with identical risk characteristics.

**Log link for dispersion default**: ensures phi_i > 0 always. The identity link is available but requires careful monitoring — it can produce negative phi_i estimates for extrapolation.


## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_dispersion_demo.py).

## Reference

- Smyth (1989): "Generalized Linear Models with Varying Dispersion", JRSS-B 51:47-60
- Smyth & Verbyla (1999): "Adjusted likelihood methods for modelling dispersion in GLMs", Environmetrics 10:695-709
- R dglm package: https://github.com/cran/dglm

## Performance

Benchmarked against a constant-phi Tweedie GLM (statsmodels) on synthetic UK
commercial property pure premium data: 25,000 policies, known DGP where phi varies
3–6x across distribution channels (direct vs broker SME vs broker large), temporal
70/30 train/test split. See `notebooks/benchmark_dispersion.py` for full methodology.

| Metric                         | Tweedie GLM (const phi) | DGLM       |
|--------------------------------|-------------------------|------------|
| Tweedie deviance (test)        | —                       | comparable |
| Phi MAE vs true                | higher                  | lower      |
| Max channel A/E deviation      | higher                  | lower      |
| Variance ratio by channel      | miscalibrated in tails  | closer to 1.0 |
| Overdispersion LRT p-value     | not applicable          | < 0.001    |
| Fit time                       | faster                  | 3–6x slower |

The Tweedie GLM assigns the same phi to a direct retail policy and a broker-placed
large commercial account. The DGLM captures the 3–6x dispersion difference between
channels, materially improving variance calibration for the segments where it matters
most (reinsurance pricing, capital loading). The LRT test (`overdispersion_test()`)
flags whether varying phi adds value on your specific portfolio. On homogeneous books
a constant-phi Tweedie is adequate and faster.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS — the full RS algorithm for jointly modelling mean and all distributional parameters including shape |
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Joint frequency-severity models with Sarmanov copula — extends dispersion modelling to the two-part structure |

