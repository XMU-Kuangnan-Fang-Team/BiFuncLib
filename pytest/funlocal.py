import numpy as np
import matplotlib
matplotlib.use("Agg")
from BiFuncLib.local_bifunc import local_bifunc
from BiFuncLib.simulation_data import local_sim_data
from BiFuncLib.FDPlot import FDPlot


def _check_local_result(res):
    assert isinstance(res, dict)


def test_local_full_story():
    local_simdata = local_sim_data(n=100, T=100, sigma=0.75, seed=1)
    res = local_bifunc(
        local_simdata["data"],
        local_simdata["location"],
        1.02e-5,
        2,
        0.3,
        opt=False,
    )
    _check_local_result(res)

    opt_res = local_bifunc(
        local_simdata["data"],
        local_simdata["location"],
        np.array([1.02e-5]),
        np.array([2, 3]),
        np.array([0.3, 0.5]),
        opt=True,
    )
    _check_local_result(opt_res)
    FDPlot(opt_res).local_individuals_fdplot()
    FDPlot(opt_res).local_center_fdplot()
