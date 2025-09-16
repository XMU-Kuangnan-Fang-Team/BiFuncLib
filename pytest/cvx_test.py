import math
import random
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
from BiFuncLib.simulation_data import cvx_sim_data
from BiFuncLib.cvx_main_func import gkn_weights
from BiFuncLib.cvx_biclus import cvx_biclus_valid, cvx_biclus_missing


@pytest.fixture(scope="session")
def data():
    X = cvx_sim_data().copy()
    X -= np.mean(X)
    X /= np.linalg.norm(X, "fro")
    return X


@pytest.fixture(scope="session")
def weights(data):
    return gkn_weights(data, phi=0.5, k_row=5, k_col=5)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(42)
    np.random.seed(42)


def test_cvx_biclus_missing(data, weights):
    n, p = data.shape[1], data.shape[0]
    m_row, m_col = weights["E_row"].shape[0], weights["E_col"].shape[0]
    Lambda_row = np.random.randn(n, m_row)
    Lambda_col = np.random.randn(p, m_col)
    Theta = random.sample(range(1, n * p + 1), math.floor(0.1 * n * p))
    res = cvx_biclus_missing(
        data,
        weights["E_row"],
        weights["E_col"],
        weights["w_row"],
        weights["w_col"],
        gam=200,
        Lambda_row=Lambda_row,
        Lambda_col=Lambda_col,
        Theta=Theta,
    )
    assert res is not None


def test_cvx_biclus_valid_example2(data, weights):
    gammaSeq = 10 ** np.linspace(0, 3, 5)
    res = cvx_biclus_valid(
        data,
        weights["E_row"],
        weights["E_col"],
        weights["w_row"],
        weights["w_col"],
        gammaSeq,
        plot_error=False,
    )
    assert res is not None


def test_cvx_biclus_valid_example3(data, weights):
    gammaSeq = 10 ** np.linspace(0, 1, 7)
    res = cvx_biclus_valid(
        data,
        weights["E_row"],
        weights["E_col"],
        weights["w_row"],
        weights["w_col"],
        gammaSeq,
        smooth=True,
    )
    assert res is not None
