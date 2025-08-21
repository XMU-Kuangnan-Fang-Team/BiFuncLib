import matplotlib
matplotlib.use("Agg")
import pytest
from BiFuncLib.simulation_data import ssvd_sim_data
from BiFuncLib.ssvd_main_func import jaccardmat
from BiFuncLib.ssvd_biclus import ssvd_biclus, s4vd_biclus
from BiFuncLib.bcheatmap import bcheatmap
from BiFuncLib.BiclustResult import BiclustResult


@pytest.fixture(scope="session")
def ssvd_data():
    return ssvd_sim_data()

def test_ssvd_biclus(ssvd_data):
    data = ssvd_data["data"]
    res_sim = ssvd_data["res"]
    out = ssvd_biclus(data, K=1)
    assert isinstance(out, BiclustResult)
    print("ssvd jaccard:", jaccardmat(res_sim, out))
    bcheatmap(data, out)

def test_s4vd_pointwise_false(ssvd_data):
    """s4vd pointwise=False"""
    data = ssvd_data["data"]
    res_sim = ssvd_data["res"]
    out = s4vd_biclus(data, pcerv=0.5, pceru=0.5, pointwise=False, nbiclust=1)
    assert isinstance(out, BiclustResult)
    print("row jaccard:", jaccardmat(res_sim, out, "row"))
    print("col jaccard:", jaccardmat(res_sim, out, "column"))
    bcheatmap(data, out, axisR=False, axisC=False, heatcols=None, clustercols=None, allrows=True, allcolumns=True)

def test_s4vd_pointwise_true(ssvd_data):
    data = ssvd_data["data"]
    res_sim = ssvd_data["res"]
    out1 = s4vd_biclus(
        data,
        pcerv=0.5,
        pceru=0.5,
        pointwise=True,
        nbiclust=1,
        cols_nc=False,
        rows_nc=False,
        row_overlap=False,
        col_overlap=False,
        savepath=True,
    )
    out2 = s4vd_biclus(
        data,
        pcerv=0.5,
        pceru=0.5,
        pointwise=True,
        nbiclust=1,
        cols_nc=True,
        rows_nc=True,
        row_overlap=False,
        col_overlap=False,
        savepath=True,
    )
    assert out1 is not None
    assert out2 is not None
    print("s4vd_pw jaccard:", jaccardmat(res_sim, out2))
    bcheatmap(data, out2, axisR=True, axisC=True, heatcols=None, clustercols=None, allrows=False, allcolumns=False)
