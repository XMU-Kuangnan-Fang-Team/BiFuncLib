FunSAS
=========================

.. _funsas-label:

This page references the `official documentation of FunSAS <https://cran.r-project.org/web/packages/sasfunclust/sasfunclust.pdf>`_.

Method Description
------------------
The article introduces a novel model-based procedure for sparse clustering of functional data, referred to as Sparse and Smooth Functional Clustering. It is referred to as SaS-Funclust in the paper, but for the sake of naming consistency in our package, it is denoted as FunSAS.
This method is designed to classify a set of curves into homogeneous groups while also identifying the most informative parts of the domain.
Here's a concise description of the method:

- Objective function

The method aims to enhance both the accuracy and interpretability of functional data clustering by detecting informative portions of the domain.

- Model

It relies on a functional Gaussian mixture model, where parameters are estimated by maximizing a log-likelihood function penalized with a functional adaptive pairwise fusion penalty and a roughness penalty.

- Penalties

Functional Adaptive Pairwise Fusion Penalty (FAPFP): This penalty identifies noninformative portions of the domain by allowing the means of separated clusters to converge to common values.

- Smoothness Penalty

This penalty imposes a degree of smoothing on the estimated cluster means to improve interpretability.

- Algorithm

The model is estimated using an Expectation-Conditional Maximization (ECM) algorithm paired with a cross-validation procedure for parameter selection.

- Performance

Through a Monte Carlo simulation study, SaS-Funclust outperforms other methods in terms of clustering performance and interpretability.


Function
--------------
This method provides five core functions: **sas_sim_data** for simulation module, **sas_bifunc** and **sas_bifunc_cv** for modeling module,
**FDPlot.sas_fdplot** and **FDPlot.sas_cvplot** for visualization module.
Because the parameters of functions **sas_bifunc** and **sas_bifunc_cv** are similar while their outputs differ, we will explain the two functions together. 

sas_sim_data
~~~~~~~~~~~~~~~
**sas_sim_data** generates simulated data in different scenarios.

.. code-block:: python

    sas_sim_data(scenario, n_i = 50, nbasis = 30, length_tot = 50, var_e = 1, var_b = 1, seed = 123)

Parameter
^^^^^^^^^^
.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **scenario**
     - integer 0, 1 or 2, a number indicating the scenario considered, which stands for "Scenario I", "Scenario II", and "Scenario III" respectively.
   * - **n_i**
     - integer, number of curves in each cluster. Default is 50.
   * - **nbasis**
     - integer, the dimension of the set of B-spline functions. Default is 30.
   * - **length_tot**
     - integer, number of evaluation points. Default is 50.
   * - **var_e**
     - integer, variance of the measurement error. Default is 1.
   * - **var_b**
     - integer, diagonal entries of the coefficient variance matrix, which is assumed to be diagonal, with equal diagonal entries, and the same among clusters.
   * - **seed**
     - integer, random seeds each time when data is generated. Default is 123.


Value
^^^^^^^^^

The function **sas_sim_data** outputs a dict contains following arguments:

- **X**: observation matrix, where the rows correspond to argument values and columns to replications.

- **X_fd**: functional observations without measurement error.

- **mu_fd**: true cluster mean function.

- **grid**: the vector of time points where the curves are sampled.

- **clus**: true cluster membership vector.

Example
^^^^^^^^
.. code-block:: python

    from BiFuncLib.simulation_data import sas_sim_data
    sas_simdata = sas_sim_data(1, n_i = 20, var_e = 1, var_b = 0.25)


sas_bifunc and sas_bifunc_cv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**cc_bifunc** performs model fitting,
while **cc_bifunc_cv** performs K-fold cross-validation procedure to choose the number of clusters and the tuning parameters for the model.

.. code-block:: python

    sas_bifunc(X = None, timeindex = None, curve = None, grid = None, q = 30, par_LQA = None,
               lambda_l = 1e1, lambda_s = 1e1, G = 2, tol = 1e-7, maxit = 50, plot = False,
               trace = False, init = "kmeans", varcon = "diagonal", lambda_s_ini = None)

and

.. code-block:: python

    sas_bifunc_cv(X = None, timeindex = None, curve = None, grid = None, q = 30,
                  lambda_l_seq = None, lambda_s_seq = None, G_seq = None, tol = 1e-7, maxit = 50,
                  par_LQA = None, plot = False, trace = False, init = "kmeans", varcon = "diagonal",
                  lambda_s_ini = None, K_fold = 5, X_test = None, grid_test = None, m1 = 1, m2 = 0, m3 = 1)
                  

Parameter
^^^^^^^^^^

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **X**
     - array, observation matrix, where the rows correspond to argument values and columns to replications.
   * - **timeindex**
     - array or none, a vector of length :math:`\sum_{i=1}^{N} n_i`. The entries from :math:`\sum_{i=1}^{k-1}(n_i+1)` to :math:`\sum_{i=1}^{k} n_i` provide the locations on grid of curve :math:`k`. Default is None.
   * - **curve**
     - array or none, a vector of length :math:`\sum_{i=1}^{N} n_i`. The entries from :math:`\sum_{i=1}^{k-1}(n_i + 1)` to :math:`\sum_{i=1}^{k} n_i` are equal to :math:`k`. If X is a matrix, curve is ignored. Default is None.
   * - **grid**
     - array or none, the vector of time points where the curves are sampled.
   * - **q**
     - numeric, the dimension of the set of B-spline functions. Default is 30.
   * - **par_LQA**
     - 
   * - **alpha**
     - integer 1 or 0, if **alpha=1** row shift is allowed, if **alpha=0** row shift is avoided. Default is 0.
   * - **beta**
     - integer 1 or 0, if **beta=1** column shift is allowed, if **beta=0** column shift is avoided. Default is 0.
   * - **const_alpha**
     - bool, if True, row shift is contrained as constant. Default is False.
   * - **const_beta**
     - bool, if True, column shift is contrained as constant. Default is False.
   * - **shift_alignment**
     - bool, if True, the shift aligment is performed, if False no alignment is performed. Default is False.
   * - **shift_max**
     - numeric between 0 and 1, controls the maximal allowed shift at each iteration, in the alignment procedure with respect to the range of curve domains.
   * - **max_iter_align**
     - integer, maximum number of iteration in the alignment procedure.
   * - **plot**
     - bool, whether to output graphs showing how each model metric changes with iterations. Default is True.


Value
^^^^^^^^^
The function **cc_bifunc** outputs a dict including clustering results and information of the model.

- **Number**: integer, the number of clustering groups.

- **RowxNumber**: array of bool, a matrix contains row clustering results.

- **Numberxcol**: array of bool, a matrix contains column clustering results.

- **Parameter**: dict, a dict containing the parameters setting of the algorithm.

The function **cc_bifunc_cv** outputs a dataframe including each model metric changes with different delta. Users can select best parameter through the function.
If **plot=True**, then the following graphs will be displayed:

- Delta v.s. H score

.. image:: /_static/cc_htot.png
   :width: 400
   :align: center

- Delta v.s. number of not assigned

.. image:: /_static/cc_notassigned.png
   :width: 400
   :align: center

- Delta v.s. number of cluster

.. image:: /_static/cc_numclus.png
   :width: 400
   :align: center


Example
^^^^^^^^
.. code-block:: python

  from BiFuncLib.simulation_data import cc_sim_data
  from BiFuncLib.cc_bifunc import cc_bifunc, cc_bifunc_cv
  delta_list = np.linspace(0.1, 20, num = 21)
  fun_mat = cc_sim_data()
  # Find best delta
  cc_result_cv = cc_bifunc_cv(fun_mat, delta_list = delta_list, alpha = 1, beta = 0, const_alpha = True, plot = True)
  # Without shift_alignment
  cc_result_1 = cc_bifunc(fun_mat, delta = 10, alpha = 1, beta = 0, const_alpha = True, shift_alignment = False)
  # With shift_alignment
  cc_result_2 = cc_bifunc(fun_mat, delta = 10, alpha = 1, beta = 0, const_alpha = True, shift_alignment = True)


FDPlot.cc_fdplot
~~~~~~~~~~~~~~~~~~
**FDPlot.cc_fdplot** displays the clustered function curves and provides options for mean subtraction, alignment, and warping.

.. code-block:: python

    FDPlot(result).cc_fdplot(data, only_mean = False, aligned = False, warping = False)


Parameter
^^^^^^^^^^
.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **result**
     - dict, a clustering result generated by **cc_bifunc** function.
   * - **data**
     - array, same as in **cc_bifunc**.
   * - **only_mean**
     - bool, if True, only the template functions for each bi-cluster is displayed. Default is False.
   * - **aligned**
     - bool, if True, the alignemd functions are displayed. Default is False.
   * - **warping**
     - bool, if True, a figure representing the warping functions are displayed. Default is False.


Value
^^^^^^^^^
Here we illustrate the outputs of the plot function in different settings.

- cluster results

.. table::
   :class: tight-table

   +----------+----------+----------+
   | |figa|   | |figb|   | |figc|   |
   +----------+----------+----------+

.. |figa|  image:: /_static/cc_clus1.png
   :width: 250px
.. |figb|  image:: /_static/cc_clus2.png
   :width: 250px
.. |figc|  image:: /_static/cc_clus3.png
   :width: 250px


- alignemd function

.. image:: /_static/cc_aligned.png
   :width: 400
   :align: center

- warping function

.. image:: /_static/cc_warping.png
   :width: 400
   :align: center


Example
^^^^^^^^
.. code-block:: python

   import numpy as np
   from BiFuncLib.FDPlot import FDPlot
   from BiFuncLib.simulation_data import cc_sim_data
   from BiFuncLib.cc_bifunc import cc_bifunc, cc_bifunc_cv
   delta_list = np.linspace(0.1, 20, num = 21)
   fun_mat = cc_sim_data()
   # Find best delta
   cc_result_cv = cc_bifunc_cv(fun_mat, delta_list = delta_list, alpha = 1, beta = 0, const_alpha = True, plot = True)
   # Without shift_alignment
   cc_result_1 = cc_bifunc(fun_mat, delta = 10, alpha = 1, beta = 0, const_alpha = True, shift_alignment = False)
   FDPlot(cc_result_1).cc_fdplot(fun_mat, only_mean = True, aligned = False, warping = False)
   # With shift_alignment
   cc_result_2 = cc_bifunc(fun_mat, delta = 10, alpha = 1, beta = 0, const_alpha = True, shift_alignment = True)
   FDPlot(cc_result_2).cc_fdplot(fun_mat, only_mean = False, aligned = True, warping = True)


