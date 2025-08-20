import pytest
import numpy as np
from GENetLib.fda_func import create_fourier_basis
from BiFuncLib.fem_bifunc import fem_bifunc
from BiFuncLib.simulation_data import fem_sim_data
from BiFuncLib.BsplineFunc import BsplineFunc
from GENetLib.fda_func import basis_fd

# Test default parameters for basis_fd
def test_basis_fd_defaults():
    result = basis_fd()
    assert result['btype'] == 'bspline'
    assert result['rangeval'] == [0, 1]
    assert result['nbasis'] == 2

# Test create_fourier_basis
def test_create_fourier_basis():
    basis = create_fourier_basis((0, 181), nbasis=25)
    assert basis['btype'] == 'fourier'
    assert basis['rangeval'] == [0, 181]
    assert basis['nbasis'] == 25

# Test fem_sim_data
def test_fem_sim_data():
    fem_simdata = fem_sim_data()
    assert isinstance(fem_simdata, dict)
    assert 'data' in fem_simdata

# Test BsplineFunc
def test_bspline_func():
    basis = create_fourier_basis((0, 181), nbasis=25)
    time_grid = np.arange(1, 182).tolist()
    fem_simdata = fem_sim_data()
    fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
    assert isinstance(fdobj, dict)
    assert 'coefs' in fdobj

# Test fem_bifunc
def test_fem_bifunc():
    basis = create_fourier_basis((0, 181), nbasis=25)
    time_grid = np.arange(1, 182).tolist()
    fem_simdata = fem_sim_data()
    fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
    res = fem_bifunc(fdobj, K=[5, 6], model=['AkjBk', 'DkBk', 'DB'], crit='icl',
                     init='hclust', lambda_=0.01, disp=True)
    assert isinstance(res, dict)
    assert 'K' in res
    assert 'P' in res

# Test fem_bifunc with user initialization
def test_fem_bifunc_with_init():
    basis = create_fourier_basis((0, 181), nbasis=25)
    time_grid = np.arange(1, 182).tolist()
    fem_simdata = fem_sim_data()
    fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
    res = fem_bifunc(fdobj, K=[5, 6], model=['AkjBk', 'DkBk', 'DB'], crit='icl',
                     init='hclust', lambda_=0.01, disp=True)
    res2 = fem_bifunc(fdobj, K=[res['K']], model=['AkjBk', 'DkBk'], init='user', Tinit=res['P'], 
                      lambda_=0.01, disp=True, graph=False)
    assert isinstance(res2, dict)
    assert 'K' in res2
    assert 'P' in res2
