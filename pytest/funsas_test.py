import numpy as np
import matplotlib
matplotlib.use("Agg")
from BiFuncLib.simulation_data import sas_sim_data
from BiFuncLib.sas_bifunc import sas_bifunc, sas_bifunc_cv
from BiFuncLib.FDPlot import FDPlot


def _check_sas_result(res):
    assert isinstance(res, dict)

def test_sas_full_story():
    sas_simdata_0 = sas_sim_data(0, n_i=20, var_e=1, var_b=0.25)
    sas_simdata_1 = sas_sim_data(1, n_i=20, var_e=1, var_b=0.25)
    sas_simdata_2 = sas_sim_data(2, n_i=20, var_e=1, var_b=0.25)
  
    sas_result = sas_bifunc(
        X=sas_simdata_0["X"],
        grid=sas_simdata_0["grid"],
        lambda_s=1e-6,
        lambda_l=10,
        G=2,
        maxit=5,
        q=10,
        init="hierarchical",
        trace=True,
        plot=True,
    )
    _check_sas_result(sas_result)
  
    sas_result = sas_bifunc(
        X=sas_simdata_1["X"],
        grid=sas_simdata_1["grid"],
        lambda_s=1e-6,
        lambda_l=10,
        G=2,
        maxit=5,
        q=10,
        init="hierarchical",
        trace=False,
        varcon="equal",
    )
    _check_sas_result(sas_result)

    lambda_s_seq = 10.0 ** np.arange(-4, -2, dtype=float)
    lambda_l_seq = 10.0 ** np.arange(-1, 1, dtype=float)
    G_seq = [2, 3]
    sas_cv_result = sas_bifunc_cv(
        X=sas_simdata_2["X"],
        grid=sas_simdata_2["grid"],
        lambda_l_seq=lambda_l_seq,
        lambda_s_seq=lambda_s_seq,
        G_seq=G_seq,
        maxit=20,
        init="model-based",
        K_fold=2,
        q=10,
    )
    assert isinstance(sas_cv_result, dict)
    FDPlot(sas_result).sas_fdplot()
    FDPlot(sas_cv_result).sas_cvplot()
