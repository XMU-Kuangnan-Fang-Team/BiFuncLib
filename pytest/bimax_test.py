import matplotlib

matplotlib.use("Agg")
import pytest
from BiFuncLib.simulation_data import bimax_sim_data
from BiFuncLib.bimax_biclus import bimax_biclus
from BiFuncLib.bcheatmap import bcheatmap


def test_bimax_full_story():
    bimax_simdata = bimax_sim_data()
    bimax_res = bimax_biclus(
        bimax_simdata,
        minr=4,
        minc=4,
        number=10,
    )
    assert bimax_res is not None
    bcheatmap(
        bimax_simdata,
        bimax_res,
        axisR=False,
        axisC=False,
        heatcols=None,
        clustercols=None,
        allrows=True,
        allcolumns=True,
    )
    bcheatmap(
        bimax_simdata,
        bimax_res,
        axisR=True,
        axisC=True,
        heatcols=None,
        clustercols=None,
        allrows=False,
        allcolumns=False,
    )
