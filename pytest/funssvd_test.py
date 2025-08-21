import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import itertools
import pytest
from BiFuncLib.simulation_data import ssvd_sim_data
from BiFuncLib.ssvd_main_func import jaccardmat
from BiFuncLib.ssvd_biclus import ssvd_biclus, s4vd_biclus
from BiFuncLib.bcheatmap import bcheatmap


def ssvd_simdata():
    return ssvd_sim_data()

def test_ssvd_biclus(ssvd_simdata):
    data = ssvd_simdata["data"]
    res_sim = ssvd_simdata["res"]
    res = ssvd_biclus(data, K=1)
    assert isinstance(res, dict)
    _ = jaccardmat(res_sim, res)
    bcheatmap(data, res)

_BOOL_COMBOS = list(itertools.product([True, False], repeat=4))
@pytest.mark.parametrize(
    "cols_nc,rows_nc,row_overlap,col_overlap",
    _BOOL_COMBOS,
)

def test_s4vd_biclus_all_combos(ssvd_simdata, cols_nc, rows_nc, row_overlap, col_overlap):
    data = ssvd_simdata["data"]
    res_sim = ssvd_simdata["res"]
    s4vd_res = s4vd_biclus(
        data,
        pcerv=0.5,
        pceru=0.5,
        pointwise=True,
        nbiclust=1,
        cols_nc=cols_nc,
        rows_nc=rows_nc,
        row_overlap=row_overlap,
        col_overlap=col_overlap,
        savepath=True,
    )
    assert isinstance(s4vd_res, dict)
    _ = jaccardmat(res_sim, s4vd_res)
    bcheatmap(data, s4vd_res)

def test_s4vd_pointwise_false(ssvd_simdata):
    data = ssvd_simdata["data"]
    res_sim = ssvd_simdata["res"]
    s4vd_res = s4vd_biclus(
        data,
        pcerv=0.5,
        pceru=0.5,
        pointwise=False,
        nbiclust=1,
    )
    assert isinstance(s4vd_res, dict)
    print("row  jaccard:", jaccardmat(res_sim, s4vd_res, "row"))
    print("col  jaccard:", jaccardmat(res_sim, s4vd_res, "column"))
    bcheatmap(data, s4vd_res)
  
