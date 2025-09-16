import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from BiFuncLib.simulation_data import cc_sim_data
from BiFuncLib.cc_bifunc import cc_bifunc, cc_bifunc_cv
from BiFuncLib.FDPlot import FDPlot


def _check_cc_result(res):
    assert isinstance(res, dict)
    assert "RowxNumber" in res
    assert "NumberxCol" in res


def test_cc_sim_data():
    fun_mat = cc_sim_data()
    assert fun_mat.shape == (30, 7, 240)


def test_cc_bifunc_cv_smoke():
    fun_mat = cc_sim_data()
    delta_list = np.linspace(0.1, 20, num=21)
    res_cv = cc_bifunc_cv(
        fun_mat,
        delta_list=delta_list,
        alpha=1,
        beta=0,
        const_alpha=True,
        const_beta=True,
    )
    assert isinstance(res_cv, pd.DataFrame)


def test_cc_bifunc_no_shift():
    fun_mat = cc_sim_data()
    res = cc_bifunc(
        fun_mat,
        delta=10,
        template_type="medoid",
        alpha=0,
        beta=0,
        const_alpha=True,
        shift_alignment=False,
    )
    _check_cc_result(res)
    FDPlot(res).cc_fdplot(fun_mat, only_mean=True, aligned=False, warping=False)


def test_cc_bifunc_with_shift():
    fun_mat = cc_sim_data()
    res = cc_bifunc(
        fun_mat,
        delta=10,
        alpha=0,
        beta=1,
        const_alpha=False,
        shift_alignment=True,
    )
    _check_cc_result(res)
    res = cc_bifunc(
        fun_mat,
        delta=20,
        alpha=1,
        beta=1,
        const_alpha=False,
        const_beta=True,
        shift_alignment=True,
    )
    _check_cc_result(res)
    FDPlot(res).cc_fdplot(fun_mat, only_mean=False, aligned=True, warping=True)


data = cc_sim_data()


def test_wrong_dims():
    with pytest.raises(ValueError, match="three dimensions"):
        cc_bifunc(np.ones((5, 5)), delta=1)


def test_medoid_with_alpha():
    with pytest.raises(ValueError, match="Medoid template.*alpha.*beta"):
        cc_bifunc(data, delta=1, template_type="medoid", alpha=1)


def test_shift_max_out_of_range():
    with pytest.raises(ValueError, match="shift_max must be in"):
        cc_bifunc(data, delta=1, shift_max=1.5)


def test_alpha_not_0_or_1():
    with pytest.raises(ValueError, match="alpha and beta must be 0 or 1"):
        cc_bifunc(data, delta=1, alpha=2)


def test_template_type_invalid():
    with pytest.raises(ValueError, match="template_type.*is not defined"):
        cc_bifunc(data, delta=1, template_type="foo")


def test_number_non_positive():
    with pytest.raises(ValueError, match="number must be.*greater than 0"):
        cc_bifunc(data, delta=1, number=0)


def test_shift_alignment_not_bool():
    with pytest.raises(ValueError, match="shift_alignment should be a logical"):
        cc_bifunc(data, delta=1, shift_alignment=1)


def test_max_iter_align_non_positive():
    with pytest.raises(
        ValueError, match="max.iter.align must be.*greater than 0"
    ):
        cc_bifunc(data, delta=1, max_iter_align=-1)


def test_const_alpha_not_bool():
    with pytest.raises(ValueError, match="const_alpha.*must be TRUE or FALSE"):
        cc_bifunc(data, delta=1, const_alpha="True")


def test_delta_negative():
    with pytest.raises(ValueError, match="delta must be.*greater than 0"):
        cc_bifunc(data, delta=-1)
