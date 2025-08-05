<div align="center">
<img src="image/overall_fig.png" alt="overall_fig.png" width="1000">
</div>

[![pypi](https://img.shields.io/pypi/v/GENetLib?logo=Pypi)](https://pypi.org/project/GENetLib)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-lightblue.svg)
[![Build](https://github.com/XMU-Kuangnan-Fang-Team/GENetLib/actions/workflows/CI.yml/badge.svg)](https://github.com/XMU-Kuangnan-Fang-Team/GENetLib/actions/workflows/CI.yml/badge.svg)
[![codecov](https://codecov.io/github/XMU-Kuangnan-Fang-Team/GENetLib/graph/badge.svg?token=9J9QMN7L9Z)](https://codecov.io/github/XMU-Kuangnan-Fang-Team/BiFuncLib)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/genetlib/badge/?version=latest)](https://genetlib.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## `BiFuncLib`:  A Python library for biclustering with functional data
``BiFuncLib`` is a Python package that aggregates multiple biclustering methods.

Our package provides biclustering methods for both functional and scalar data (mainly for functional data). The functional-data approaches are further divided into biclustering and local clustering variants. A small set of scalar-data biclustering algorithms has also been included to ensure the package’s extensibility.

For functional data, in standard biclustering, each sample contains multiple functions, and the algorithm jointly clusters both samples and these functions across the entire dataset. In contrast, local clustering assumes only one function per sample; it segments that single function into local pieces and then performs biclustering on the resulting sub-functions.

Thus, this package is capable of handling a variety of biclustering methods, by category:
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
[documentations](https://open-box.readthedocs.io/en/latest/).
This package has been uploaded to PyPI with previous versions, and the web page is available at
[PyPI package](https://pypi.org/project/genetlib/). Users can also check [releases](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/releases) to get historical versions.

## Features
``GENetLib`` has the following features:
- **Comprehensiveness**: Supports a variety of input and output formats, enabling the construction of comprehensive neural network models for G-E interaction analysis.
- **Flexibility**: Offers a multitude of parameters allowing users to build models flexibly according to their specific needs.
- **Functional data compatibility**: Implements methods for functional data analysis (FDA) in Python, facilitating the processing of functional data with Python.
- **Scalability**: New methods for G-E interaction analysis via deep learning can be easily integrated into the system.

## Installation
It is recommended to use ``pip`` for installation:
```c
pip install GENetLib
```
To get further information about installation and independencies, please move to [installation instructions](https://genetlib.readthedocs.io/en/latest/installation.html).

## Quick Start
We start with the two basic functions ``scalar_ge`` and ``func_ge``.
### scalar_ge
``scalar_ge`` performs G-E interaction analysis via deep leanring when the input is scalar data.
```Python
from GENetLib.sim_data import sim_data_scalar
from GENetLib.scalar_ge import scalar_ge

# Get example data where input is scalar data and output is survival data
scalar_survival_linear = sim_data_scalar(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500,
                                         dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)

# Set up the ScalerGE model
scalar_ge_res = scalar_ge(y = scalar_survival_linear['y'], G = scalar_survival_linear['G'], E = scalar_survival_linear['E'],
                          ytype = 'Survival',num_hidden_layers = 2, nodes_hidden_layer = [1000, 100], num_epochs = 100,
                          learning_rate1 = 0.06, learning_rate2 = 0.035, lambda1 = None, lambda2 = 0.09, Lambda = 0.1,
                          threshold = 0.01, split_type = 0, ratio = [7, 3], important_feature = True, plot = True)
```
### func_ge
``func_ge`` performs G-E interaction analysis via deep leanring when the input is functional data.
```Python
from GENetLib.sim_data import sim_data_func
from GENetLib.func_ge import func_ge

# Get example data where input is densely measured functional data and output is survival data
func_continuous = sim_data_func(n = 1500, m = 30, ytype = 'Continuous', seed = 123)

# Set up the FuncGE model
func_ge_res = func_ge(y = func_continuous['y'], X = func_continuous['X'], location = func_continuous['location'],
                      Z = func_continuous['Z'], ytype = 'Continuous', btype = 'Bspline', num_hidden_layers = 2,
                      nodes_hidden_layer = [100,10], num_epochs = 50, learning_rate1 = 0.02, learning_rate2 = 0.035,
                      nbasis1 = 5, params1 = 4, lambda1 = None, lambda2 = 0.01, Lambda = 0.01, Bsplines = 5,
                      norder1 = 4, split_type = 1, ratio = [3, 1, 1], plot_res = True)
```
For more information about the functions and methods, please check [main functions](https://genetlib.readthedocs.io/en/latest/main%20functions/main%20functions.html#).

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

Other referenced papers can be obtained in [references](https://genetlib.readthedocs.io/en/latest/references.html).

## License
BiFuncLib is licensed under the MIT License. See [LICENSE](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/blob/main/LICENSE) for details.

## Feedback
- Welcome to submit [issues](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/issues) or [pull requests](https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/pulls).
- Send an email to Barry57@163.com to contact us.
- Thanks for all the supports! 👏

