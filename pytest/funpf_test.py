import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from BiFuncLib.pf_bifunc import pf_bifunc
from BiFuncLib.simulation_data import pf_sim_data
from BiFuncLib.FDPlot import FDPlot


def _check_pf_result(res):
    assert isinstance(res, dict)
    assert 'feature cluster' in res
    assert 'sample cluster' in res

def test_pf_sim_data():
    pf_sim = pf_sim_data(n=60, T=10, nknots=3, order=3, seed=123)
    assert isinstance(pf_sim, dict)
    assert pf_sim['data'].ndim == 2

def test_pf_bifunc_basic():
    pf_simdata = pf_sim_data(n=60, T=10, nknots=3, order=3, seed=123)['data']
    res = pf_bifunc(
        pf_simdata,
        nknots=3,
        order=3,
        gamma1=0.023,
        gamma2=3,
        theta=1,
        tau=3,
        max_iter=500,
        eps_abs=1e-3,
        eps_rel=1e-3
    )
    _check_pf_result(res)
    FDPlot(res).pf_fdplot()

def test_pf_bifunc_opt():
    pf_simdata = pf_sim_data(n=60, T=10, nknots=3, order=3, seed=123)['data']
    res_opt = pf_bifunc(
        pf_simdata,
        nknots=3,
        order=3,
        gamma1=[0.023, 0.025],
        gamma2=[2, 3],
        opt=True,
        theta=1,
        tau=3,
        max_iter=500,
        eps_abs=1e-3,
        eps_rel=1e-3
    )
    _check_pf_result(res_opt)
    FDPlot(res_opt).pf_fdplot()
  
