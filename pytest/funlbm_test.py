import numpy as np
import pytest
from BiFuncLib.FDPlot import FDPlot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from BiFuncLib.simulation_data import lbm_sim_data
from BiFuncLib.lbm_bifunc import lbm_bifunc
from BiFuncLib.lbm_main_func import ari

def _check_lbm_result(res: dict):
    assert isinstance(res, dict)
    assert 'row_clust' in res
    assert 'col_clust' in res
    assert len(res['row_clust']) > 0
    assert len(res['col_clust']) > 0

def test_lbm_sim_data():
    lbm_sim = lbm_sim_data(n=50, p=50, t=15, seed=42)
    assert isinstance(lbm_sim, dict)
    assert 'data' in lbm_sim

def test_lbm_bifunc_basic():
    lbm_sim = lbm_sim_data(n=100, p=100, t=30, seed=1)
    data = lbm_sim['data']
    res1 = lbm_bifunc(data, K=2, L=2, display=False, basis_name = 'spline', init='funFEM')
    _check_lbm_result(res1)
    res2 = lbm_bifunc(data, K=2, L=2, display=True, basis_name = 'spline', init='kmeans')
    _check_lbm_result(res2)
    res3 = lbm_bifunc(data, K=2, L=2, display=True, init='funFEM')
    _check_lbm_result(res3)
    row_ari = ari(res3['row_clust'], lbm_sim['row_clust'])
    col_ari = ari(res3['col_clust'], lbm_sim['col_clust'])
    assert 0 <= row_ari <= 1
    assert 0 <= col_ari <= 1

def test_lbm_bifunc_grid():
    lbm_sim = lbm_sim_data(n=50, p=50, t=15, bivariate=True, seed=456)
    data = [lbm_sim['data1'], lbm_sim['data2']]
    res = lbm_bifunc(data, K=[2, 3], L=[2, 3], display=True, init='kmeans')
    FDPlot(res).lbm_fdplot('proportions')
    FDPlot(res).lbm_fdplot('evolution')
    FDPlot(res).lbm_fdplot('likelihood')
    FDPlot(res).lbm_fdplot('blocks')
    FDPlot(res).lbm_fdplot('means')
    _check_lbm_result(res)

