import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BiFuncLib.cc_main_func import bigcc_fun, evaluate_mat_dist, ccscore_fun
from BiFuncLib.bimax_biclus import bimax_biclus


#  Functional Cheng and Church algorithm
def cc_bifunc(data, delta, theta = 1, template_type = 'mean', number = 100,
              alpha = 0, beta = 0, const_alpha = False, const_beta = False,
              shift_alignment = False, shift_max = 0.1, max_iter_align = 100):
    
    # Check inputs
    if len(data.shape) != 3:
        raise ValueError("Error: data should be an array of three dimensions")
    if template_type == 'medoid' and (alpha != 0 or beta != 0):
        raise ValueError("Error: Medoid template is defined only for alpha and beta equal to 0")
    if shift_max > 1 or shift_max < 0:
        raise ValueError("Error: shift_max must be in [0,1]")
    if alpha not in [0, 1] or beta not in [0, 1]:
        raise ValueError("Error: alpha and beta must be 0 or 1")
    if template_type not in ['mean', 'medoid']:
        raise ValueError(f"Error: template_type {template_type} is not defined")
    if number <= 0:
        raise ValueError("Error: number must be an integer greater than 0")
    if not isinstance(shift_alignment, bool):
        raise ValueError("Error: shift_alignment should be a logical variable")
    if max_iter_align <= 0:
        raise ValueError("Error: max.iter.align must be an integer greater than 0")
    if not isinstance(const_alpha, bool) or not isinstance(const_beta, bool):
        raise ValueError("Error: const_alpha and const_beta must be TRUE or FALSE")
    if delta < 0:
        raise ValueError("Error: delta must be a number greater than 0")

    # Initialize parameters
    n, m, p = data.shape
    parameter_input = {
        'delta': [delta],
        'theta': [theta],
        'template_type': [template_type],
        'alpha': [alpha],
        'beta': [beta],
        'const_alpha': [const_alpha],
        'const_beta': [const_beta],
        'shift_alignment': [shift_alignment],
        'shift_max': [shift_max],
        'max_iter_align': [max_iter_align]}
    if alpha == 0 and beta == 0:
        only_one = 'True'
    elif alpha == 0 and beta != 0:
        only_one = 'True_beta'
    elif alpha != 0 and beta == 0:
        only_one = 'True_alpha'
    elif alpha != 0 and beta != 0:
        only_one = 'False'
    x = np.full((n, number), False, dtype=bool)
    y = np.full((number, m), False, dtype=bool)
    xy = np.zeros((n, m), dtype=int)
    logr = np.full(n, True, dtype=bool)
    logc = np.full(m, True, dtype=bool)
    n_clust = 2
    k = 0
    cl = 1
    clus_row = None
    clus_col = None

    # Update logr and logc
    for i in range(number):
        if np.any(logr) and np.any(logc):
            submat = data[logr, :, 0][:, logc]
            rows_with_all_nan = np.all(np.isnan(submat), axis=1)
            logr[logr] = ~rows_with_all_nan
        if (only_one == 'False' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) <= 1 or np.sum(logc) <= 1)) \
           or (only_one == 'True' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) < 1 or np.sum(logc) < 1)) \
           or (only_one == 'True_alpha' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) < 1 or np.sum(logc) <= 1)) \
           or (only_one == 'True_beta' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) <= 1 or np.sum(logc) < 1)):
           break
        if np.any(logr) and np.any(logc):
            submat = data[logr, :, 0][:, logc]
            cols_with_all_nan = np.all(np.isnan(submat), axis=0)
            logc[logc] = ~cols_with_all_nan
        if (only_one == 'False' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) <= 1 or np.sum(logc) <= 1)) \
           or (only_one == 'True' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) < 1 or np.sum(logc) < 1)) \
           or (only_one == 'True_alpha' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) < 1 or np.sum(logc) <= 1)) \
           or (only_one == 'True_beta' and (np.sum(~(xy.astype(bool))) < 2 or np.sum(logr) <= 1 or np.sum(logc) < 1)):
           break
        fun_mat_temp = data[logr][:, logc, :]
        erg = bigcc_fun(fun_mat_temp, delta, theta, template_type, alpha, beta,
                        const_alpha, const_beta, shift_alignment, shift_max,
                        max_iter_align, only_one)
        n_clust -= 1
        if (np.sum(erg[0]) == 0 and n_clust == 0) or (np.sum(erg[0]) == 0 and i == 0):
            break
        elif np.sum(erg[0]) == 0 and n_clust >= 1:
            cl += 1
            logr = np.full(xy.shape[0], False, dtype=bool)
            logr[np.array(clus_row)[:, cl - 1]] = True
            logc = np.full(xy.shape[1], False, dtype=bool)
            logc[np.array(clus_col)[cl - 1, :]] = True
        else:
            k += 1
            true_rows = np.where(logr)[0]
            x[true_rows, k - 1] = erg[0]
            true_cols = np.where(logc)[0]
            y[k - 1, true_cols] = erg[1]
            xy = xy + np.outer(x[:, k - 1].astype(int), y[k - 1, :].astype(int))
            if only_one == 'False':
                res = bimax_biclus(1 - xy, minr=2, minc=2, number=100)
            elif only_one == 'True':
                res = bimax_biclus(1 - xy, minr=1, minc=1, number=100)
            elif only_one == 'True_alpha':
                res = bimax_biclus(1 - xy, minr=1, minc=2, number=100)
            elif only_one == 'True_beta':
                res = bimax_biclus(1 - xy, minr=2, minc=1, number=100)
            n_clust = res.Number
            cl = 1
            if n_clust == 0:
                break
            clus_row = res.RowxNumber
            clus_col = res.NumberxCol
            d_rows = np.sum(clus_row, axis=0)
            d_cols = np.sum(clus_col, axis=1)
            dimensioni = d_rows * d_cols
            if_oneone = np.where(dimensioni > 1)[0]
            if len(if_oneone) == 0:
                break
            if len(if_oneone) != n_clust:
                n_clust = len(if_oneone)
                clus_row = np.array(clus_row)[:, if_oneone]
                clus_col = np.array(clus_col)[if_oneone, :]
            if n_clust == 0:
                break
            if n_clust > 1:
                d = np.sum(clus_row, axis=0) * np.sum(clus_col, axis=1)
                sorted_idx = np.argsort(-d)
                clus_row = np.array(clus_row)[:, sorted_idx]
                clus_col = np.array(clus_col)[sorted_idx, :]
            logr = np.full(xy.shape[0], False, dtype = bool)
            logr[np.array(clus_row)[:, 0]] = True
            logc = np.full(xy.shape[1], False, dtype = bool)
            logc[np.array(clus_col)[0, :]] = True
    
    # Return results
    clus_row_final = x[:, :k]
    clus_col_final = y[:k, :]
    if k > 1:
        d = np.sum(clus_row_final, axis=0) * np.sum(clus_col_final, axis=1)
        sorted_idx = np.argsort(-d)
        clus_row_final = clus_row_final[:, sorted_idx]
        clus_col_final = clus_col_final[sorted_idx, :]
    result = {"NumberxCol": clus_col_final,
              "RowxNumber": clus_row_final,
              "Number": k,
              "Parameter": parameter_input}
    return result


def cc_bifunc_cv(data, delta_list, theta = 1.5, template_type = 'mean', number = 100,
                 alpha = 0, beta = 0, const_alpha = False, const_beta = False,
                 shift_alignment = False, shift_max = 0.1, max_iter_align = 100, plot = True):

    # Initialize parameters
    Htot_best = max(delta_list)
    Htot_sum_list = []
    Htot_all_mean_list = []
    num_clust_list = []
    not_assigned_list = []
    
    # Optimize delta
    for d in delta_list:
        res_fun = cc_bifunc(data, delta = d, template_type = template_type,
                            theta = theta, number = number, alpha = alpha, beta = beta,
                            const_alpha = const_alpha, const_beta = const_beta, 
                            shift_alignment = shift_alignment,
                            shift_max = shift_max, max_iter_align = max_iter_align)
        if res_fun['Number'] == 0:
            Htot_all_mean_list.append(np.nan)
            Htot_sum_list.append(np.nan)
            num_clust_list.append(0)
            not_assigned_list.append(data.shape[0] * data.shape[1])
        elif res_fun['Number'] == 1:
            row_mask = res_fun['RowxNumber']
            col_mask = res_fun['NumberxCol']
            fun_mat_cl = data[row_mask, :, :][:, col_mask, :]
            dist_mat = evaluate_mat_dist(fun_mat_cl, template_type, alpha, beta, 
                                         const_alpha, const_beta, shift_alignment, 
                                         shift_max, max_iter_align)
            H_cl = ccscore_fun(dist_mat)
            total_elements = data.shape[0] * data.shape[1]
            assigned_elements = fun_mat_cl.shape[0] * fun_mat_cl.shape[1]
            not_assigned_list.append(total_elements - assigned_elements)
            Htot_d = np.mean(H_cl)
            Htot_all_mean_list.append(Htot_d)
            Htot_sum_list.append(np.sum(H_cl))
            num_clust_list.append(1)
        elif res_fun['Number'] > 1:
            num_clust_list.append(res_fun['Number'])
            H_cl = []
            total_elements = data.shape[0] * data.shape[1]
            for cl in range(res_fun['Number']):
                row_mask = res_fun['RowxNumber'][:, cl]
                col_mask = res_fun['NumberxCol'][cl, :]
                rows_idx = np.where(row_mask)[0]
                cols_idx = np.where(col_mask)[0]
                submatrix = data[np.ix_(rows_idx, cols_idx, np.arange(data.shape[0]))]
                final_shape = (len(rows_idx), len(cols_idx), data.shape[0])
                fun_mat_cl = submatrix.reshape(final_shape)
                dist_mat = evaluate_mat_dist(fun_mat_cl, template_type, alpha, beta, 
                                             const_alpha, const_beta, shift_alignment,
                                             shift_max, max_iter_align)
                H_cl_temp = ccscore_fun(dist_mat)
                H_cl.append(H_cl_temp)
                total_elements -= fun_mat_cl.shape[0] * fun_mat_cl.shape[1]
            not_assigned_list.append(total_elements)
            Htot_d = np.mean(H_cl)
            Htot_all_mean_list.append(Htot_d)
            Htot_sum_list.append(np.sum(H_cl))
            if Htot_d < Htot_best:
                Htot_best = Htot_d
    
    # Retrun results
    h = pd.DataFrame({
        "Htot_sum": Htot_sum_list,
        "Htot_all_mean": Htot_all_mean_list,
        "num_clust": num_clust_list,
        "delta": delta_list,
        "not_assigned": not_assigned_list
    })
    
    # Plot
    if plot:
        plt.figure()
        plt.plot(h["delta"], h["Htot_sum"], marker='o')
        plt.title("Delta vs H tot")
        plt.xlabel("delta")
        plt.ylabel("Htot_sum")
        plt.show()
        
        plt.figure()
        plt.plot(h["delta"], h["num_clust"], marker='o')
        plt.title("Delta vs num clusters")
        plt.xlabel("delta")
        plt.ylabel("num_clust")
        plt.show()
        
        plt.figure()
        plt.plot(h["delta"], h["not_assigned"], marker='o')
        plt.title("Delta vs not assigned")
        plt.xlabel("delta")
        plt.ylabel("not_assigned")
        plt.show()  
    return h

