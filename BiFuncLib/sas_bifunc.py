import numpy as np
import itertools
from GENetLib.fda_func import fd
from GENetLib.plot_gene import plot_fd
from GENetLib.fda_func import create_bspline_basis

from BiFuncLib.sas_main_func import sasfclust_init, loglik, get_msdrule, get_zero, sasfclust_Mstep, sasfclust_Estep, classify


def sas_bifunc(X = None, timeindex = None, curve = None, grid = None, q = 30, par_LQA = None,
               lambda_l = 1e1, lambda_s = 1e1, G = 2, tol = 1e-7, maxit = 50, plot = False,
               trace = False, init = "kmeans", varcon = "diagonal", lambda_s_ini = None):
    
    # Check parameters
    if par_LQA is None:
        par_LQA = {"eps_diff": 1e-6, "MAX_iter_LQA": 200, "eps_LQA": 1e-5}
    der = 2
    gamma_ada = 1
    CK = 0
    hard = False
    perc_rankpapp = None
    pert = None
    if G == 1:
        lambda_l = 0
    if X is not None:
        if X.ndim == 2:
            n_t, n_obs = X.shape
            if grid is None:
                grid = np.linspace(0, 1, n_t)
            if len(grid) != n_t:
                raise ValueError("Length grid is wrong!")
            vec_x = np.ravel(X, order='F')
            timeindex_vec = np.tile(np.arange(1, len(grid) + 1), n_obs)
            curve_vec = np.repeat(np.arange(1, n_obs + 1), len(grid))
            vec = {"x": vec_x, "timeindex": timeindex_vec, "curve": curve_vec}
        elif X.ndim == 1:
            if grid is None:
                raise ValueError("For irregularly sampled functional data grid must be provided")
            if timeindex is None:
                raise ValueError("For irregularly sampled functional timeindex grid must be provided")
            if curve is None:
                raise ValueError("For irregularly sampled functional timeindex curve must be provided")
            vec = {"x": np.array(X), "timeindex": timeindex, "curve": curve}
        else:
            raise ValueError("No data provided")
    else:
        raise ValueError("No data provided")
    data = vec

    # Initialize parameters
    initfit = sasfclust_init(data=data, pert=pert, grid=grid, q=q, G=G, der=der, 
                             gamma_ada=gamma_ada, lambda_s_ini=lambda_s_ini,
                             init=init, varcon=varcon)
    parameters = initfit["parameters"]
    vars_ = initfit["vars"]
    S = initfit["S"]
    FullS = initfit["FullS"]
    W = initfit["W"]
    AW_vec = initfit["AW_vec"]
    P_tot = initfit["P_tot"]
    basis = initfit["basis"]
    sigma_new = parameters["sigma"]
    ind = 1
    if plot:
        basis = create_bspline_basis((grid[0], grid[-1]), nbasis=q)
        mean_fd = fd(parameters["mu"].T, basis)
        plot_fd(mean_fd, ylab="Cluster Mean")
    lk_old = 0
    lk_temp = loglik(parameters=parameters, data=data, vars_=vars_, FullS=FullS, 
                     W=W, AW_vec=AW_vec, P_tot=P_tot, lambda_s=lambda_s, lambda_l=lambda_l)
    if trace:
        print(f"Iteration 0: Sigma = {sigma_new} loglk = {lk_temp[0]} ploglk = {lk_temp[1]}")
    lk_new = lk_temp[1]
    
    # Loop for updating parameters
    while abs(lk_old - lk_new) > tol and (ind <= maxit):
        parameters = sasfclust_Mstep(parameters, data, vars_, S, tol, hard, lambda_s, lambda_l, 
                                     W, AW_vec, P_tot, par_LQA, CK, perc_rankpapp, varcon=varcon)
        vars_ = sasfclust_Estep(parameters, data, vars_, S, hard)
        lk_old = lk_new
        lk_i = loglik(parameters=parameters, data=data, vars_=vars_, FullS=FullS, 
                      W=W, AW_vec=AW_vec, P_tot=P_tot, lambda_s=lambda_s, lambda_l=lambda_l)
        lk_new = -lk_i[1]
        sigma_new = parameters["sigma"]
        if trace:
            print(f"Iteration {ind}: Sigma = {sigma_new} loglk = {lk_i[0]} ploglk = {lk_i[1]}")
        if plot:
            basis = create_bspline_basis((grid[0], grid[-1]), nbasis=q)
            mean_fd = fd(parameters["mu"].T, basis)
            plot_fd(mean_fd, ylab="Cluster Mean")
        ind += 1

    # Return results
    mod = {
        "data": data,
        "parameters": parameters,
        "vars": vars_,
        "FullS": FullS,
        "grid": grid,
        "W": W,
        "AW_vec": AW_vec,
        "P_tot": P_tot,
        "lambda_s": lambda_s,
        "lambda_l": lambda_l
    }
    mean_fd = fd(parameters["mu"].T, basis)
    clus = classify(mod)
    out = {"mod": mod, "mean_fd": mean_fd, "clus": clus}
    return out


def sas_bifunc_cv(X = None, timeindex = None, curve = None, grid = None, q = 30,
                  lambda_l_seq = None, lambda_s_seq = None, G_seq = None, tol = 1e-7, maxit = 50,
                  par_LQA = None, plot = False, trace = False, init = "kmeans", varcon = "diagonal",
                  lambda_s_ini = None, K_fold = 5, X_test = None, grid_test = None,
                  m1 = 1, m2 = 0, m3 = 1):
    if lambda_l_seq is None:
        lambda_l_seq = 10 ** np.arange(-1, 3)
    if lambda_s_seq is None:
        lambda_s_seq = 10 ** np.arange(-5, -2)
    if G_seq is None:
        G_seq = [2]
    if X is None:
        raise ValueError("No data provided")
    if np.ndim(X) == 2:
        N = X.shape[1]
    elif np.ndim(X) == 1:
        if grid is None:
            raise ValueError("For irregularly sampled functional data, grid must be provided")
        if timeindex is None:
            raise ValueError("For irregularly sampled functional data, timeindex must be provided")
        if curve is None:
            raise ValueError("For irregularly sampled functional data, curve must be provided")
        if np.max(timeindex) > len(grid):
            raise ValueError("Length of grid is wrong!")
        N = len(np.unique(curve))
    else:
        raise ValueError("No data provided")
    comb_list = []
    G_seq_list = list(G_seq) if hasattr(G_seq, '__iter__') else [G_seq]
    for g, ls, ll in itertools.product(G_seq_list, lambda_s_seq, lambda_l_seq):
        comb_list.append((g, ls, ll))
    if X_test is None:
        def parr_fun(ii):
            G_i, lambda_s_i, lambda_l_i = comb_list[ii]
            ran_seq = np.random.permutation(np.arange(N))
            split_vec = np.array_split(ran_seq, K_fold)
            l_list = []
            zeros_list = []
            for lll in range(K_fold):
                train_indices = np.concatenate([split_vec[j] for j in range(K_fold) if j != lll])
                test_indices = split_vec[lll]
                grid_fold = grid
                if np.ndim(X) == 2:
                    X_fold = X[:, train_indices]
                    timeindex_fold = None
                    curve_fold = None
                elif np.ndim(X) == 1:
                    X_fold = np.concatenate([X[np.where(np.array(curve)==idx)] for idx in train_indices])
                    timeindex_fold = np.concatenate([np.array(timeindex)[np.where(np.array(curve)==idx)] for idx in train_indices])
                    curve_fold = np.concatenate([np.full(np.sum(np.array(curve)==idx), idx) for idx in train_indices])
                else:
                    raise ValueError("No data provided")
                grid_i = grid
                if np.ndim(X) == 2:
                    X_i = X[:, test_indices]
                    timeindex_i = None
                    curve_i = None
                elif np.ndim(X) == 1:
                    X_i = np.concatenate([X[np.where(np.array(curve)==idx)] for idx in test_indices])
                    timeindex_i = np.concatenate([np.array(timeindex)[np.where(np.array(curve)==idx)] for idx in test_indices])
                    curve_i = np.concatenate([np.full(np.sum(np.array(curve)==idx), idx) for idx in test_indices])
                else:
                    raise ValueError("No data provided")
                mod = sas_bifunc(X=X_fold, timeindex=timeindex_fold, curve=curve_fold, grid=grid_fold,
                                 lambda_l=lambda_l_i, lambda_s=lambda_s_i, G=G_i, maxit=maxit, q=q,
                                 init=init, varcon=varcon, tol=tol, par_LQA=par_LQA, plot=plot, trace=trace)
                l_val = loglik(parameters=mod['mod']["parameters"], X=X_i, timeindex=timeindex_i,
                               curve=curve_i, grid=grid_i, vars_=mod['mod']["vars"], FullS=mod['mod']["FullS"],
                               W=mod['mod']["W"], AW_vec=mod['mod']["AW_vec"], P_tot=mod['mod']["P_tot"])
                l_list.append(l_val)
                zeros_list.append(get_zero(mod['mod']))
            mean_l = np.mean(l_list)
            sd_l = np.std(l_list, ddof=1) / np.sqrt(K_fold)
            zeros_mean = np.mean(zeros_list)
            return {"mean": mean_l, "sd": sd_l, "zeros": zeros_mean}
    else:
        if not np.allclose(grid, grid_test):
            raise ValueError("Not equal grids between training and test set")
        def parr_fun(ii):
            G_i, lambda_s_i, lambda_l_i = comb_list[ii]
            mod = sas_bifunc(X=X, timeindex=timeindex, curve=curve, grid=grid,
                             lambda_l=lambda_l_i, lambda_s=lambda_s_i, G=G_i, maxit=maxit, q=q,
                             init=init, lambda_s_ini=lambda_s_ini, varcon=varcon, tol=tol, par_LQA=par_LQA,
                             plot=plot, trace=trace)
            l_val = loglik(parameters=mod['mod']["parameters"], X=X_test, grid=grid_test,
                           vars_=mod['mod']["vars"], FullS=mod['mod']["FullS"], W=mod['mod']["W"],
                           AW_vec=mod['mod']["AW_vec"], P_tot=mod['mod']["P_tot"],
                           lambda_s=mod['mod'].get("lambda_s"), lambda_l=mod['mod'].get("lambda_l"))
            zeros_val = get_zero(mod['mod'])
            return {"mean": l_val[0], "sd": 0, "zeros": zeros_val}
    vec_par = [parr_fun(ii) for ii in range(len(comb_list))]
    CV = np.array([d["mean"] for d in vec_par])
    CV_sd = np.array([d["sd"] for d in vec_par])
    zeros_all = np.array([d["zeros"] for d in vec_par])
    ksdrule = get_msdrule(CV, CV_sd, comb_list, m1, m2, m3)
    G_opt = ksdrule[0]
    lambda_s_opt = ksdrule[1]
    lambda_l_opt = ksdrule[2]
    out = {"G_opt": G_opt,
           "lambda_l_opt": lambda_l_opt,
           "lambda_s_opt": lambda_s_opt,
           "comb_list": np.stack(comb_list, axis=0),
           "CV": CV,
           "CV_sd": CV_sd,
           "zeros": zeros_all,
           "ms": (m1, m2, m3)}
    return out

