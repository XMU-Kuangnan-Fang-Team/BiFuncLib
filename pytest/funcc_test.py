import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from BiFuncLib.simulation_data import cc_sim_data
from BiFuncLib.cc_bifunc import cc_bifunc, cc_bifunc_cv
from BiFuncLib.FDPlot import FDPlot


def _check_cc_result(res: dict, aligned: bool):
    assert isinstance(res, dict)
    assert 'RowxNumber' in res
    assert 'NumberxCol' in res

def test_cc_sim_data():
    fun_mat = cc_sim_data()
    assert fun_mat.shape == (30, 7, 240)

def test_cc_bifunc_cv_smoke():
    fun_mat = cc_sim_data()
    delta_list = np.linspace(0.1, 20, num=21)
    res_cv = cc_bifunc_cv(fun_mat, delta_list=delta_list, alpha=1, beta=0, const_alpha=True, plot=False)
    assert isinstance(res_cv, pd.DataFrame)

def test_cc_bifunc_no_shift():
    fun_mat = cc_sim_data()
    res = cc_bifunc(fun_mat, delta=10, alpha=0, beta=0, const_alpha=True, shift_alignment=False)
    _check_cc_result(res)
    FDPlot(res).cc_fdplot(fun_mat, only_mean=True, aligned=False, warping=False)

def test_cc_bifunc_with_shift():
    fun_mat = cc_sim_data()
    res = cc_bifunc(fun_mat, delta=10, alpha=0, beta=1, const_alpha=True, shift_alignment=True, plot=True)
    _check_cc_result(res)
    res = cc_bifunc(fun_mat, delta=10, alpha=1, beta=1, const_alpha=False, shift_alignment=True, plot=True)
    _check_cc_result(res)
    FDPlot(res).cc_fdplot(fun_mat, only_mean=False, aligned=True, warping=True)
