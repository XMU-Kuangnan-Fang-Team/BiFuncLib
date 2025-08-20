import numpy as np
import pytest
from BiFuncLib.FDPlot import FDPlot
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
    assert lbm_sim['data'].shape == (50, 15)

def test_lbm_bifunc_basic():
    lbm_sim = lbm_sim_data(n=60, p=60, t=20, seed=123)
    data = lbm_sim['data']
    res = lbm_bifunc(data, K=4, L=3, display=False, basis_name = 'spline', init='funFEM')
    _check_lbm_result(res)
    row_ari = ari(res['row_clust'], lbm_sim['row_clust'])
    col_ari = ari(res['col_clust'], lbm_sim['col_clust'])
    assert 0 <= row_ari <= 1
    assert 0 <= col_ari <= 1

def test_lbm_bifunc_grid():
    lbm_sim = lbm_sim_data(n=40, p=40, t=10, bivariate=True, seed=456)
    data = [lbm_sim['data1'], lbm_sim['data2']]
    res = lbm_bifunc(data, K=[2, 3], L=[2, 3], display=False)
    _check_lbm_result(res)

def test_lbm_bifunc_user_init():
    lbm_sim = lbm_sim_data(n=30, p=30, t=10, seed=789)
    data = lbm_sim['data']
    res0 = lbm_bifunc(data, K=4, L=3, display=True, init='kmeans')
    FDPlot(res0).lbm_fdplot('proportions')
    FDPlot(res0).lbm_fdplot('evolution')
    FDPlot(res0).lbm_fdplot('likelihood')
    FDPlot(res0).lbm_fdplot('blocks')
    FDPlot(res0).lbm_fdplot('means')
    res1 = lbm_bifunc(data, K=[res0['K']], L=[res0['L']],
                      init='user',
                      row_init=res0['row_clust'],
                      col_init=res0['col_clust'],
                      display=False)
    _check_lbm_result(res1)
