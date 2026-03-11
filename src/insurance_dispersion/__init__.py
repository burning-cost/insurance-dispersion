"""
insurance-dispersion: Double GLM (DGLM) for joint modelling of mean and dispersion.

The DGLM (Smyth 1989) extends the standard GLM by adding a second submodel
for the dispersion parameter phi. Instead of a single scalar dispersion shared
across all observations, each observation gets its own phi_i driven by covariates.

This matters in non-life insurance: a broker-channel fleet policy and a direct-
channel private car policy should not share the same uncertainty. The DGLM lets
you model that explicitly.

Quick start::

    from insurance_dispersion import DGLM
    import insurance_dispersion.families as fam

    model = DGLM(
        formula="claim_amount ~ C(age_band) + C(vehicle_class)",
        dformula="~ C(channel) + C(limit_band)",
        family=fam.Gamma(),
        data=df,
        exposure="earned_exposure",
    )
    result = model.fit()
    print(result.summary())
    print(result.mean_relativities())
    print(result.dispersion_relativities())
"""

from insurance_dispersion.model import DGLM
from insurance_dispersion.results import DGLMResult
from insurance_dispersion import families

__version__ = "0.1.0"
__all__ = ["DGLM", "DGLMResult", "families"]
