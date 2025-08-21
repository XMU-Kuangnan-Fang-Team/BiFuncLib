import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from BiFuncLib.simulation_data import sparse_sim_data
from BiFuncLib.sparse_bifunc import sparse_bifunc
from BiFuncLib.FDPlot import FDPlot


def _check_sparse_result(res):
    assert isinstance(res, dict)

def test_sparse_sim_data():
    n = 100
    x = np.linspace(0, 1, 1000)
    paramC = 0.7
    sim = sparse_sim_data(n, x, paramC, plot=True)
    assert sim['data'].shape is not None

def test_sparse_bifunc_smoke():
    n, K = 100, 2
    x = np.linspace(0, 1, 1000)
    sim = sparse_sim_data(n, x, 0.7, plot=False)
    res = sparse_bifunc(sim['data'], x, K, true_clus=sim['cluster'])
    _check_sparse_result(res)
    FDPlot(res).sparse_fdplot(x, sim['data'])

def test_sparse_bifunc_pam():
    n, K = 100, 2
    x = np.linspace(0, 1, 1000)
    sim = sparse_sim_data(n, x, 0.7, plot=False)
    res = sparse_bifunc(sim['data'], x, K, method='pam', true_clus=sim['cluster'])
    _check_sparse_result(res)

def test_sparse_bifunc_hier():
    n, K = 100, 2
    x = np.linspace(0, 1, 1000)
    sim = sparse_sim_data(n, x, 0.7, plot=False)
    res = sparse_bifunc(sim['data'], x, K, method='hier')
    _check_sparse_result(res)

