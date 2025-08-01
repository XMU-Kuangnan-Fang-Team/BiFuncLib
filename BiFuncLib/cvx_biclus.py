import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from BiFuncLib.cvx_main_func import gkn_weights, cobra_validate, cobra_pod, biclust_smooth


warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# Cluster with validation
def cvx_biclus_valid(data, phi, k, gamma, Lambda_row=None, Lambda_col=None, fraction=0.1,
                     max_iter=100, tol=1e-3, max_iter_inner=1000, tol_inner=1e-4,
                     smooth = False, plot_error = True, plot_clus = True):
    wts = gkn_weights(data, phi=phi, k_row=k, k_col=k)
    w_row = wts["w_row"]
    w_col = wts["w_col"]
    E_row = wts["E_row"]
    E_col = wts["E_col"]
    sol = cobra_validate(data, E_row, E_col, w_row, w_col, gamma, Lambda_row=Lambda_row,
                         Lambda_col=Lambda_col, fraction=fraction, max_iter=max_iter, tol=tol,
                         max_iter_inner=1000, tol_inner=1e-4)
    if plot_error == True:
        plt.figure()
        plt.plot(sol["validation_error"], marker="o")
        plt.xlabel("Gamma index")
        plt.ylabel("Validation Error")
        plt.title("Validation error vs Gamma")
        plt.show()
    ix_opt = int(np.argmin(sol["validation_error"]))
    print("Optimal gamma index:", ix_opt)
    if smooth == True:
        groups_row = sol["groups_row"][ix_opt]
        groups_col = sol["groups_col"][ix_opt]
        M = biclust_smooth(data, groups_row, groups_col) 
    if plot_clus == True:
        if smooth == True:
            sns.clustermap(M, cmap="Blues", yticklabels=False, xticklabels=False)
            plt.show()
        elif smooth == False:
            sns.clustermap(sol['U'][2], cmap="Blues", yticklabels=False, xticklabels=False)
            plt.show()
    return sol 
    
    
# Cluster with missing data
def cvx_biclus_missing(data, phi, k, gamma, Lambda_row, Lambda_col, Theta,
                       max_iter=100, tol=1e-3, max_iter_inner=1000, tol_inner=1e-4,
                       plot_clus = True):
    wts = gkn_weights(data, phi=phi, k_row=k, k_col=k)
    w_row = wts["w_row"]
    w_col = wts["w_col"]
    E_row = wts["E_row"]
    E_col = wts["E_col"]
    sol = cobra_pod(data, Lambda_row, Lambda_col, E_row, E_col, gamma*w_row, gamma*w_col, Theta,
                    max_iter=max_iter, tol=tol, max_iter_inner=max_iter_inner,
                    tol_inner=tol_inner)
    if plot_clus == True:
        sns.clustermap(sol['U'], cmap="Blues", yticklabels=False, xticklabels=False)
        plt.show()
    return sol

