import pytest
import numpy as np
from GENetLib.fda_func import create_fourier_basis
from BiFuncLib.fem_bifunc import fem_bifunc
from BiFuncLib.simulation_data import fem_sim_data
from BiFuncLib.BsplineFunc import BsplineFunc
from GENetLib.fda_func import basis_fd

# Test fem_sim_data
def test_fem_sim_data():
    fem_simdata = fem_sim_data()
    assert isinstance(fem_simdata, dict)
    assert 'data' in fem_simdata

# Test fem_bifunc
def test_fem_bifunc():
    basis = create_fourier_basis((0, 181), nbasis=25)
    time_grid = np.arange(1, 182).tolist()
    fem_simdata = fem_sim_data()
    fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
    res1 = fem_bifunc(fdobj, K=[5, 6], model=['AkjBk', 'DkBk', 'DB'], crit='aic',
                     init='kmeans', lambda_=0.01, disp=True)
    res2 = fem_bifunc(fdobj, K=[5, 6], model=['DkB', 'DBk', 'AkjB'], crit='bic',
                 init='random', lambda_=0.01, disp=True)
    assert isinstance(res1, dict)
    assert isinstance(res2, dict)

# Test fem_bifunc with user initialization
def test_fem_bifunc_with_init():
    basis = create_fourier_basis((0, 181), nbasis=25)
    time_grid = np.arange(1, 182).tolist()
    fem_simdata = fem_sim_data()
    fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
    res = fem_bifunc(fdobj, K=[5, 6], model=['AjBk', 'AjB', 'AB'], crit='icl',
                     init='hclust', lambda_=0.01, disp=True)
    res2 = fem_bifunc(fdobj, K=[res['K']], model=['AkBk', 'AkB', 'ABk'], init='user', Tinit=res['P'], 
                      lambda_=0.01, disp=True, graph=False)
    assert isinstance(res2, dict)
    assert 'K' in res2
    assert 'P' in res2

