<!--  
<div align="center">
<img src="image/overall_fig.png" alt="overall_fig.png" width="1000">
</div> -->

[![pypi](https://img.shields.io/pypi/v/BiFuncLib?logo=Pypi)](https://pypi.org/project/BiFuncLib)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-lightblue.svg)
[![Build](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/actions/workflows/CI.yml/badge.svg)](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/actions/workflows/CI.yml/badge.svg)
[![codecov](https://codecov.io/github/XMU-Kuangnan-Fang-Team/BiFuncLib/graph/badge.svg?token=9J9QMN7L9Z)](https://codecov.io/github/XMU-Kuangnan-Fang-Team/BiFuncLib)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bifunclib/badge/?version=latest)](https://bifunclib.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## `BiFuncLib`:  A Python library for biclustering with functional data
``BiFuncLib`` is a Python package that aggregates multiple biclustering methods.

Our package provides biclustering methods for both functional and scalar data (mainly for functional data). The functional-data approaches are further divided into biclustering and local clustering variants. A small set of scalar-data biclustering algorithms has also been included to ensure the package’s extensibility.

For functional data, in standard biclustering, each sample contains multiple functions, and the algorithm jointly clusters both samples and these functions across the entire dataset. In contrast, local clustering assumes only one function per sample; it segments that single function into local pieces and then performs biclustering on the resulting sub-functions.

## Methods
This package is capable of handling a variety of biclustering methods, by category:
### Functional data (biclustering)
- FunFEM (EM algorithm)
- FunLBM (Latent Block Model)
- FunCC (Cheng and Church)
- FunPF (Penalized Fusion)

### Functional data (local clustering)
- FunSparse (Sparse clustering)
- FunSAS (Sparse And Smooth)
- FunLocal (Local clustering)

### Scalar data
- Bimax (divide-and-conquer algorithm)
- SSVD (Sparse Singular Value Decomposition)
- CVX (ConVeX biclustering)

BiFuncLib unifies these approaches into a comprehensive and easily extensible biclustering toolkit. The framework is shown below.

<div align="center">
<img src="image/framework.png" alt="framework" width="600">
</div>

We provide a web-based documentation which introduces the meaning of function parameters, the usage of functions, detailed information about methods, and gives examples for each. The web page is available at
[documentations](https://bifunclib.readthedocs.io/en/latest/).
This package has been uploaded to PyPI with previous versions, and the web page is available at
[PyPI package](https://pypi.org/project/bifunclib/). Users can also check [releases](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/releases) to get historical versions.

## Installation
It is recommended to use ``pip`` for installation:
```c
pip install BiFuncLib
```

If you want to install a specific version, you can use:
```c
pip install BiFuncLib==x.y.z
```

To get further information about installation and independencies, please move to [installation instructions](https://bifunclib.readthedocs.io/en/latest/installation.html).

## Quick Start
Below are individual usage examples for each method, including computational results and plots (where applicable).

### FunFEM
``fem_bifunc`` performs FunFEM algorithm which allows to cluster functional data by modeling the curves within a common and discriminative functional subspace.
```Python
import numpy as np
from BiFuncLib.fem_bifunc import fem_bifunc
from BiFuncLib.simulation_data import fem_sim_data
from BiFuncLib.BsplineFunc import BsplineFunc
from GENetLib.fda_func import create_fourier_basis
basis = create_fourier_basis((0, 181), nbasis=25)
time_grid = np.arange(1, 182).tolist()
fem_simdata = fem_sim_data()
fdobj = BsplineFunc(basis).smooth_basis(time_grid, np.array(fem_simdata['data'].T))['fd']
fem_res = fem_bifunc(fdobj, K=[6], model=['AkjBk'], init='kmeans', lambda_=0, disp=True)
FDPlot(fem_res).fem_fdplot(fem_simdata, fdobj)
```
### FunLBM
`lbm_bifunc` performs FunLBM algorithm which is a model-based co-clustering method for large-scale functional data that simultaneously clusters rows and columns by fitting a latent block model where each block is represented by a low-dimensional functional subspace.
```python
from BiFuncLib.simulation_data import lbm_sim_data
from BiFuncLib.lbm_bifunc import lbm_bifunc
from BiFuncLib.lbm_main_func import ari
lbm_simdata = lbm_sim_data(n = 100, p = 100, t = 30, seed = 1)
data = lbm_simdata['data']
lbm_res = lbm_bifunc(data, K=4, L=3)
FDPlot(lbm_res).lbm_fdplot('proportions')
print(ari(lbm_res['col_clust'],lbm_simdata['col_clust']))
print(ari(lbm_res['row_clust'],lbm_simdata['row_clust']))
```

### FunCC
`cc_bifunc` performs FunCC, a non-parametric, non-exhaustive functional bi-clustering algorithm that extends the Cheng–Church framework; it simultaneously identifies row–column subsets of curves by minimizing an H-score and optionally aligns them with domain shifts, all without distributional assumptions. 

`cc_bifunc_cv` provides a function for finding the best tunning delta.
```python
import numpy as np
from BiFuncLib.simulation_data import cc_sim_data
from BiFuncLib.cc_bifunc import cc_bifunc, cc_bifunc_cv
delta_list = np.linspace(0.1, 20, num = 21)
fun_mat = cc_sim_data()
cc_result_cv = cc_bifunc_cv(fun_mat, delta_list = delta_list, alpha = 1, beta = 0, const_alpha = True)
cc_result = cc_bifunc(fun_mat, delta = 10, alpha = 1, beta = 0, const_alpha = True, shift_alignment = False)
FDPlot(cc_result).cc_fdplot(fun_mat, only_mean = True, aligned = False, warping = False)
```

### FunPF
`pf_bifunc` performs FunPF, which is a penalized-fusion biclustering approach for functional data that combines a smoothness penalty for curve estimation with a fusion penalty on coefficient differences, enabling simultaneous and consistent identification of row and column clusters without assuming any generative model.
```python
from BiFuncLib.pf_bifunc import pf_bifunc
from BiFuncLib.simulation_data import pf_sim_data
pf_simdata = pf_sim_data(n = 60, T = 10, nknots = 3, order = 3, seed = 123)['data']
pf_result = pf_bifunc(pf_simdata, nknots = 3, order = 3, gamma1 = 0.023, gamma2 = 3, 
                      theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)
FDPlot(pf_result).pf_fdplot()
```

### FunSparse
`sparse_bifunc` stands for FunSparse, an algorithm performs functional data grouping while simultaneously selecting the most discriminative curve regions via a hard-thresholded weighting function, yielding a concise and interpretable partition.
```python
import numpy as np
from BiFuncLib.simulation_data import sparse_sim_data
from BiFuncLib.sparse_bifunc import sparse_bifunc
K = 2
paramC = 0.7
n = 100
x = np.linspace(0, 1, 1000)
sparse_simdata = sparse_sim_data(n, x, paramC)['data']
part_vera = sparse_sim_data(n, x, paramC)['cluster']
sparse_res = sparse_bifunc(sparse_simdata, x, K, true_clus = part_vera)
FDPlot(sparse_res).sparse_fdplot(x, sparse_simdata)
```

### FunSAS
`sas_bifunc` performs FunSAS algorithm. FunSAS simultaneously clusters functional curves and pinpoints, via adaptive pairwise-fusion penalized likelihood, the specific time intervals where each pair of cluster mean curves actually differ, thereby delivering both accurate grouping and a sparse, interpretable map of informative regions.

`sas_bifunc_cv` provides a function for finding the best tunning gamma.
```python
import numpy as np
from BiFuncLib.simulation_data import sas_sim_data
from BiFuncLib.sas_bifunc import sas_bifunc, sas_bifunc_cv
sas_simdata_0 = sas_sim_data(0, n_i = 20, var_e = 1, var_b = 0.25)
sas_result = sas_bifunc(X = sas_simdata_0['X'], grid = sas_simdata_0['grid'],
                        lambda_s = 1e-6, lambda_l = 10, G = 2, maxit = 5, q = 10)
lambda_s_seq = 10 ** np.arange(-4, -2, dtype=float)
lambda_l_seq = 10 ** np.arange(-1, 1, dtype=float)
G_seq = [2, 3]
sas_cv_result = sas_bifunc_cv(X = sas_simdata_0['X'], grid = sas_simdata_0['grid'],
                              lambda_l_seq = lambda_l_seq, lambda_s_seq = lambda_s_seq,
                              G_seq = G_seq, maxit = 20, K_fold = 2, q = 10)
FDPlot(sas_result).sas_fdplot()
FDPlot(sas_cv_result).sas_cvplot()
```

### FunLocal
`local_bifunc` performs FunLocal, which simultaneously identifies data-driven subintervals and performs clustering within each, allowing functions to exhibit distinct grouping structures across different regions of their domain.
```python
from BiFuncLib.local_bifunc import local_bifunc
from BiFuncLib.simulation_data import local_sim_data
local_simdata = local_sim_data(n = 100, T = 100, sigma = 0.75, seed = 1)
opt_res = local_bifunc(local_simdata['data'], local_simdata['location'],
                       1.02e-5, 2, 0.3, opt=False)
FDPlot(opt_res).local_individuals_fdplot()
FDPlot(opt_res).local_center_fdplot()
```

### Bimax
`bimax_biclus` performs Bimax algorithm, which exhaustively finds all inclusion-maximal submatrices of 1s in a binarized data matrix without overlap.
```python
from BiFuncLib.bimax_biclus import bimax_biclus
from BiFuncLib.simulation_data import bimax_sim_data
bimax_simdata = bimax_sim_data()
bimax_res = bimax_biclus(bimax_simdata, minr=4,minc=4,number=10)
bcheatmap(bimax_simdata,bimax_res)
```

### SSVD
`s4vd_biclus` performs SSVD algorithm, which simultaneously sparsifies the left and right singular vectors of an SVD layer to extract coherent row–column blocks (biclusters).

Here, `ssvd_biclus` is the original algorithm, whereas `s4vd_biclus` is its cross-validated, hyper-parameter-tuned version; in practice, `s4vd_biclus` is recommended for modeling. The `jaccardmat` function is provided to measure similarity.

```python
from BiFuncLib.simulation_data import ssvd_sim_data
from BiFuncLib.bcheatmap import bcheatmap
from BiFuncLib.ssvd_main_func import jaccardmat
from BiFuncLib.ssvd_biclus import s4vd_biclus, ssvd_biclus
ssvd_simdata = ssvd_sim_data()
data = ssvd_simdata['data']
res_sim = ssvd_simdata['res']
s4vd_res = s4vd_biclus(data, pcerv=0.5, pceru=0.5, pointwise=False, nbiclust=1)
print(jaccardmat(res_sim, s4vd_res, 'row'))
print(jaccardmat(res_sim, s4vd_res, 'column'))
bcheatmap(data, s4vd_res)
res2_ssvd = ssvd_biclus(data,K=1)
print(jaccardmat(res_sim, res2_ssvd))
bcheatmap(data, res2_ssvd)
```

### CVX
`cvx_biclus_valid` performs COBRA algorithm with validation, which is a convex biclusteRing algorithm designed to identify checkerboard patterns in data matrices by solving a convex optimization problem, offering stability, uniqueness, and interpretability in biclustering.

`cvx_biclus_missing` handles data with missing values.

```python
import numpy as np
import random
import math
from BiFuncLib.simulation_data import cvx_sim_data
from BiFuncLib.cvx_main_func import gkn_weights
from BiFuncLib.cvx_biclus import cvx_biclus_valid, cvx_biclus_missing
cvx_simdata = cvx_sim_data()
X = cvx_simdata.copy()
X = X - np.mean(np.mean(X))
X = X / np.linalg.norm(X, 'fro')
data = X
phi = 0.5
k = 5
# Example 1
wts = gkn_weights(X, phi=phi, k_row=k, k_col=k)
E_row = wts["E_row"]
E_col = wts["E_col"]
gam = 200
m_row = E_row.shape[0]
m_col = E_col.shape[0]
n = X.shape[1]
p = X.shape[0]
Lambda_row = np.random.randn(n, m_row)
Lambda_col = np.random.randn(p, m_col)
Theta = random.sample(range(1, n*p+1), math.floor(0.1 * n * p))
wts = gkn_weights(data, phi=phi, k_row=k, k_col=k)
w_row = wts["w_row"]
w_col = wts["w_col"]
E_row = wts["E_row"]
E_col = wts["E_col"]
cvx_res1 = cvx_biclus_missing(data, E_row, E_col, w_row, w_col, gam, Lambda_row, Lambda_col, Theta)
# Example 2
nGamma = 5
gammaSeq = 10 ** np.linspace(0, 3, nGamma)
wts = gkn_weights(data, phi=phi, k_row=k, k_col=k)
w_row = wts["w_row"]
w_col = wts["w_col"]
E_row = wts["E_row"]
E_col = wts["E_col"]
cvx_res2 = cvx_biclus_valid(data, E_row, E_col, w_row, w_col, gammaSeq, plot_error = False)
```

For more information about the functions and methods, please check [main functions](https://bifunclib.readthedocs.io/en/latest/methods%20and%20main%20functions/methods%20and%20main%20functions.html).

## Reference
The main referenced papers of these methods are:
### FunFEM
Bouveyron C, Côme E, Jacques J. The discriminative functional mixture model for the analysis of bike sharing systems[J]. Preprint HAL, 2014 (01024186).
### FunLBM
Bouveyron C, Bozzi L, Jacques J, et al. The functional latent block model for the co-clustering of electricity consumption curves[J]. Journal of the Royal Statistical Society Series C: Applied Statistics, 2018, 67(4): 897-915.
### FunCC
Galvani M, Torti A, Menafoglio A, et al. FunCC: A new bi-clustering algorithm for functional data with misalignment[J]. Computational Statistics & Data Analysis, 2021, 160: 107219.
### FunPF
Fang K, Chen Y, Ma S, et al. Biclustering analysis of functionals via penalized fusion[J]. Journal of multivariate analysis, 2022, 189: 104874.
### FunSparse
Floriello D, Vitelli V. Sparse clustering of functional data[J]. Journal of Multivariate Analysis, 2017, 154: 1-18.
### FunSAS
Centofanti F, Lepore A, Palumbo B. Sparse and smooth functional data clustering[J]. Statistical Papers, 2024, 65(2): 795-825.
### FunLocal
Chen Y, Zhang Q, Ma S. Local clustering for functional data[J]. Journal of Computational and Graphical Statistics, 2025: 1-16.
### Bimax
Prelić A, Bleuler S, Zimmermann P, et al. A systematic comparison and evaluation of biclustering methods for gene expression data[J]. Bioinformatics, 2006, 22(9): 1122-1129.
### SSVD
Lee M, Shen H, Huang J Z, et al. Biclustering via sparse singular value decomposition[J]. Biometrics, 2010, 66(4): 1087-1095.
### CVX 
Chi E C, Allen G I, Baraniuk R G. Convex biclustering[J]. Biometrics, 2017, 73(1): 10-19.

Other referenced papers can be obtained in [references](https://biffunclib.readthedocs.io/en/latest/references.html).

## License
BiFuncLib is licensed under the MIT License. See [LICENSE](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/blob/main/LICENSE) for details.

## Feedback
- Welcome to submit [issues](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/issues) or [pull requests](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/pulls).
- Send an email to Barry57@163.com to contact us.
- Thanks for all the supports! 👏

