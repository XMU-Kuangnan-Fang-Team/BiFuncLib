import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from BiFuncLib.bcheatmap import bcheatmap, ma_palette


@pytest.fixture(scope="session", autouse=True)
def no_show():
    with patch.object(plt, "show"):
        yield


@pytest.fixture
def X():
    np.random.seed(42)
    return np.random.randn(30, 20)


def make_res(n_row, n_col, number, row_idx=None, col_idx=None, as_df=True):
    if number == 0:

        class ZeroRes:
            Number = 0

        return ZeroRes()
    if row_idx is None:
        row_idx = [[i, i + 1] for i in range(number)]
    if col_idx is None:
        col_idx = [[i, i + 1] for i in range(number)]
    Row = np.zeros((n_row, number), bool)
    Col = np.zeros((n_col, number), bool)
    for k, rlist in enumerate(row_idx):
        Row[rlist, k] = True
    for k, clist in enumerate(col_idx):
        Col[clist, k] = True

    class MockRes:
        Number = number
        RowxNumber = pd.DataFrame(Row) if as_df else Row
        NumberxCol = pd.DataFrame(Col) if as_df else Col

    return MockRes()


@pytest.mark.parametrize("number", [0, 1, 2, 5])
@pytest.mark.parametrize("as_df", [True, False])
@pytest.mark.parametrize("axisR,axisC", [(True, True), (False, False)])
@pytest.mark.parametrize("allrows,allcolumns", [(True, False), (False, True)])
def test_parameterized(X, number, as_df, axisR, axisC, allrows, allcolumns):
    if number == 0:
        res = make_res(30, 20, 0)
        with pytest.raises(AttributeError):
            bcheatmap(X, res)
        return
    res = make_res(30, 20, number, as_df=as_df)
    try:
        bcheatmap(
            X,
            res,
            axisR=axisR,
            axisC=axisC,
            allrows=allrows,
            allcolumns=allcolumns,
        )
    except Exception as e:
        pytest.fail(f"number={number}, as_df={as_df} failed: {e}")


def test_empty_cluster(X):
    res = make_res(30, 20, 1, row_idx=[[]], col_idx=[[]])
    try:
        bcheatmap(X, res)
    except Exception as e:
        pytest.fail(f"empty cluster failed: {e}")


def test_color_shortage(X):
    """clustercols 长度 < Number"""
    res = make_res(30, 20, 3)
    with pytest.raises(IndexError):
        bcheatmap(X, res, clustercols=["red"])


def test_df_index_labels(X):
    df = pd.DataFrame(
        X,
        index=[f"gene{i}" for i in range(30)],
        columns=[f"sample{i}" for i in range(20)],
    )
    res = make_res(30, 20, 2, as_df=True)
    try:
        bcheatmap(df, res, axisR=True, axisC=True)
    except Exception as e:
        pytest.fail(f"DataFrame labels failed: {e}")


def test_ma_palette_edge():
    cmap = ma_palette(low="k", high="w", k=2)
    assert cmap.N == 2
    cmap = ma_palette(low="k", mid=None, high="w", k=100)
    assert cmap.N == 100
