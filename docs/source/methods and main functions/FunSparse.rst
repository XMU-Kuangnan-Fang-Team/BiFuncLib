FunSparse
=========================

.. _funsparse-label:

Method Description
------------------
Sparse Functional K-means Clustering is an advanced clustering technique designed for functional data,
such as time series or curves. It extends traditional K-means clustering by incorporating sparsity to select the most relevant features of the data for clustering, enhancing interpretability and accuracy.

- Functional Data Representation

Each observed curve is represented in a continuous form, ensuring the data are suitable for clustering.

- Sparsity Constraint

A sparsity constraint is introduced to select the most relevant parts of the domain, controlled by a parameter m that specifies the measure of the domain where the weighting function is zero.

- Optimization Problem

The clustering problem is formulated as a variational problem, maximizing the weighted between-cluster sum of squares (BCSS) subject to the sparsity constraint.

- Iterative Algorithm

An iterative algorithm is used to solve the optimization problem. The algorithm alternates between:

    1. **Weighting Function Update**: Given the current clustering, the optimal weighting function is computed using the solution to the variational problem. This step identifies the most relevant parts of the domain for clustering.
    2. **Clustering Update**: Given the weighting function, the optimal clustering is found by applying a functional K-means clustering algorithm, where the distance between functions is weighted according to the weighting function.

- Parameter Tuning

The sparsity parameter m is tuned using a permutation-based GAP statistics approach to determine the optimal level of sparsity.

- Visualization and Interpretation

The results are visualized through estimated cluster mean functions and the weighting function, highlighting the most discriminative parts of the domain.

Function
--------------
This method provides three core functions: **sparse_sim_data**, **sparse_bifunc** and **FDPlot.sparse_fdplot**.
In this section, we detail their respective usage, as well as parameters, output values and usage examples for each function. 

sparse_sim_data
~~~~~~~~~~~~~~~
**sparse_sim_data** generates simulated data according to the FunPF model, and a true clustering result.

.. code-block:: python

    sparse_sim_data(n, T, nknots, order, seed = 123)

Parameter
^^^^^^^^^^

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **n**
     - integer, the number of samples.
   * - **T**
     - integer, the number of time points per sample.
   * - **nknots**
     - integer, the number of interior knots used in the B-spline basis expansion for estimating the underlying mean functions.
   * - **order**
     - integer, the polynomial degree of the B-spline basis plus one (e.g., order 4 gives cubic splines).
   * - **seed**
     - integer, random seeds each time when data is generated. Default is 123.

Value
^^^^^^^^^
The function **pf_sim_data** outputs a dict contains simulated data matrix and the true clustering results.

- **data**: dataframe, a simulated data with different time and measurements.

- **location**: array, a sequence of time from 0 to 1.

- **feature cluster**: list, true feature clustering results.

- **sample cluster**: list, true sample clustering results.


Example
^^^^^^^^
.. code-block:: python

  from BiFuncLib.simulation_data import pf_sim_data
  pf_simdata = pf_sim_data(n = 60, T = 10, nknots = 3, order = 3, seed = 123)['data']


pf_bifunc
~~~~~~~~~~~~~
**pf_bifunc** performs model fitting.

.. code-block:: python

  pf_bifunc(data, nknots, order, gamma1, gamma2, opt = False, theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)

Parameter
^^^^^^^^^^

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **data**
     - array or list, a data array of size n x p x t or a list contains two distinct n x p x t datasets.
   * - **nknots**
     - integer, number of interior knots used in the B-spline basis expansion for estimating the underlying mean functions.
   * - **order**
     - integer, polynomial degree of the B-spline basis plus one (e.g., order 4 gives cubic splines).
   * - **gamma1**
     - numeric, smoothness penalty tuning parameter that controls the trade-off between data fidelity and functional smoothness during estimation.
   * - **gamma2**
     - numeric, fusion penalty tuning parameter that governs the strength of clustering by penalizing differences between coefficient vectors.
   * - **opt**
     - bool, if True the function selects optimal (gamma1, gamma2) via a two-step BIC procedure; otherwise user-supplied values are used. Default is False.
   * - **theta**
     - numeric (>0), ADMM augmented-Lagrangian penalty weight. Default is 1.
   * - **tau**
     - numeric (>1), MCP/SCAD regularization parameter controlling the concavity of the fusion penalty. Default is 3.
   * - **max_iter**
     - integer, maximum number of ADMM iterations before stopping. Default is 500.
   * - **eps_abs**
     - numeric (>0), absolute convergence tolerance for primal and dual residuals. Default is 1e-3.
   * - **eps_rel**
     - numeric (>0), relative convergence tolerance for primal and dual residuals. Default is 1e-3.

Value
^^^^^^^^^
The function **pf_bifunc** outputs a dict including clustering results and information of the model.
The key results are **feature_cluster** and **sample_cluster**, and we omitted the outputs that are identical to the inputs.

- **Beta**: list, estimated regression coefficients for each covariate in the model.

- **feature_cluster**: list, the clustering assignment for each feature or covariate.

- **feature_number**: integer, the total count of features or covariates considered in the analysis.

- **iter**: integer, the number of iterations the algorithm has executed.

- **Lambda1**: numeric, the Lagrange multipliers associated with the row clustering constraints.

- **Lambda2**: numeric, the Lagrange multipliers related to the column clustering constraints.

- **sample_cluster**: list, the clustering assignment for each sample or observation.

- **sample_number**: integer, the total number of samples or observations in the dataset.


Example
^^^^^^^^
.. code-block:: python

   from BiFuncLib.simulation_data import pf_sim_data
   pf_simdata = pf_sim_data(n = 60, T = 10, nknots = 3, order = 3, seed = 123)['data']
   pf_result = pf_bifunc(pf_simdata, nknots = 3, order = 3, gamma1 = 0.023, gamma2 = 3, 
                        theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)


FDPlot.pf_fdplot
~~~~~~~~~~~~~~~~~~
**FDPlot.pf_fdplot** visualizes the result generated by **pf_bifunc** function.

.. code-block:: python

    FDPlot(result).pf_fdplot()


Parameter
^^^^^^^^^^
.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **result**
     - dict, a clustering result generated by **pf_bifunc** function.

Value
^^^^^^^^^
The function has two parts of output.
One is the lattice plot of the clustering results, and the other is the reconstructed function curves.

- Lattice plot of the clustering results

.. table::
   :class: tight-table

   +----------+----------+
   | |figa|   | |figb|   |
   +----------+----------+

.. |figa|  image:: /_static/pf_lattice1.png
   :width: 250px
.. |figb|  image:: /_static/pf_lattice2.png
   :width: 250px

- Reconstructed function curves

.. table::
   :class: tight-table

   +----------+----------+----------+
   | |fig1|   | |fig2|   | |fig3|   |
   +----------+----------+----------+
   | |fig4|   | |fig5|   | |fig6|   |
   +----------+----------+----------+
   | |fig7|   | |fig8|   | |fig9|   |
   +----------+----------+----------+

.. |fig1|  image:: /_static/pf_clus1.png
   :width: 250px
.. |fig2|  image:: /_static/pf_clus2.png
   :width: 250px
.. |fig3|  image:: /_static/pf_clus3.png  
   :width: 250px
.. |fig4|  image:: /_static/pf_clus4.png
   :width: 250px
.. |fig5|  image:: /_static/pf_clus5.png
   :width: 250px
.. |fig6|  image:: /_static/pf_clus6.png
   :width: 250px
.. |fig7|  image:: /_static/pf_clus7.png
   :width: 250px
.. |fig8|  image:: /_static/pf_clus8.png
   :width: 250px
.. |fig9|  image:: /_static/pf_clus9.png
   :width: 250px

Example
^^^^^^^^
.. code-block:: python

   from BiFuncLib.pf_bifunc import pf_bifunc
   from BiFuncLib.simulation_data import pf_sim_data
   from BiFuncLib.FDPlot import FDPlot
   pf_simdata = pf_sim_data(n = 60, T = 10, nknots = 3, order = 3, seed = 123)['data']
   pf_result = pf_bifunc(pf_simdata, nknots = 3, order = 3, gamma1 = 0.023, gamma2 = 3, 
                        theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)
   FDPlot(pf_result).pf_fdplot()