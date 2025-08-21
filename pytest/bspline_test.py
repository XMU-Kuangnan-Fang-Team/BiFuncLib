import numpy as np
import pytest
from BiFuncLib.BsplineFunc import BsplineFunc
from GENetLib.fda_func import create_bspline_basis, create_fourier_basis


@pytest.fixture
def sample_data():
    n = 50
    argvals = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * argvals) + 0.1 * np.random.randn(n)
    basis = create_bspline_basis(rangeval=[0, 1], nbasis=10, norder=4)
    return argvals, y, basis

def test_create_bsplinefunc(sample_data):
    argvals, y, basis = sample_data
    bsp = BsplineFunc(basisobj=basis, Lfdobj=2)
    assert bsp.basisobj is not None
    assert bsp.Lfdobj == 2

def test_penalty_matrix_shape(sample_data):
    argvals, y, basis = sample_data
    bsp = BsplineFunc(basisobj=basis, Lfdobj=2)
    bsp2 = BsplineFunc(basisobj=create_fourier_basis(), Lfdobj=2)
    P = bsp.penalty_matrix(btype='spline')
    P2 = bsp2.penalty_matrix(btype='fourier')
    nbasis = basis['nbasis']
    assert P.shape == (nbasis, nbasis)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) != 0)
    assert P2 is not None

def test_smooth_basis_returns_dict(sample_data):
    argvals, y, basis = sample_data
    bsp = BsplineFunc(basisobj=basis, Lfdobj=2)
    res = bsp.smooth_basis(argvals, y.reshape(-1, 1))
    assert isinstance(res, dict)
    for key in ['fd', 'df', 'gcv', 'beta', 'SSE', 'penmat', 'y2cMap', 'argvals', 'y']:
        assert key in res

def test_smooth_basis_fd_shape(sample_data):
    argvals, y, basis = sample_data
    bsp = BsplineFunc(basisobj=basis, Lfdobj=2)
    res = bsp.smooth_basis(argvals, y.reshape(-1, 1))
    nbasis = basis['nbasis']
    assert res['fd']['coefs'].shape == (nbasis, 1)

def test_smooth_basis_multivariate(sample_data):
    argvals, y, basis = sample_data
    y3d = np.stack([y, y*2], axis=-1)[:, None, :]
    bsp = BsplineFunc(basisobj=basis, Lfdobj=1)
    res = bsp.smooth_basis(argvals, y3d)
    nbasis = basis['nbasis']
    assert res['fd']['coefs'].shape == (nbasis, 1, 2)
    assert res['gcv'] is not None
