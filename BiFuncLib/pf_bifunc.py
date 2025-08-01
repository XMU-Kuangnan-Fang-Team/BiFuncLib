import numpy as np
import pandas as pd
import networkx as nx
from GENetLib.fda_func import bspline_mat
from scipy.linalg import solve

from BiFuncLib.pf_main_func import inv_uty_cal, beta_ini_cal, biclustr_admm
from BiFuncLib.AuxFunc import AuxFunc


def pf_bifunc(data, nknots, order, gamma1, gamma2, opt = False, theta = 1, tau = 3, 
              max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3):
    
    # Data process
    data.iloc[:, 0] = data.iloc[:, 0].astype('int')
    data.iloc[:, 1] = data.iloc[:, 1].astype('int')
    if min(data.iloc[:, 1]) == 1:
        data.iloc[:, 1] = data.iloc[:, 1] - 1
    n = max(data.iloc[:, 1]) + 1
    q = data.shape[1] - 2
    
    # Reform data
    reformed_data = []
    data_list = []
    def time_mapping(time):
        min_time = data['time'].min()
        max_time = data['time'].max()
        return (time - min_time) / (max_time - min_time)
    for measurement in data['measurement'].unique():
        current_measurement_df = data[data['measurement'] == measurement]
        current_measurement_df['time'] = time_mapping(current_measurement_df['time'])
        data_list.append(current_measurement_df)
        reformed_dataframe = pd.DataFrame()
        for i in range(2, current_measurement_df.shape[1]):
            temp_df = pd.DataFrame({
                'id': [i-2] * len(current_measurement_df['measurement']),
                'time': current_measurement_df['time'],
                'y': current_measurement_df[current_measurement_df.columns[i]]
            })
            reformed_dataframe = pd.concat([reformed_dataframe, temp_df], ignore_index=True).dropna()
        reformed_data.append(reformed_dataframe)
    
    # Generate second order difference matrix
    C = np.zeros((nknots + order - 2, nknots + order))
    for j in range(nknots + order - 2):
        d_j = [0] * j + [1, -2, 1] + [0] * (nknots + order - 3 - j)
        e_j = [0] * j + [1] + [0] * (nknots + order - 3 - j)
        C += np.outer(e_j, d_j)
    D_d = C.T @ C
    p = nknots + order
    
    # Genarate spline design matrix and response Y
    auxfunc_1 = AuxFunc(n = n, m = nknots, x = time_mapping(data['time'].unique()))
    spline_list = []
    for i, sample in enumerate(reformed_data):
        sublist = []
        for _, group in sample.groupby('id'):
            basis = bspline_mat(np.array(group['time']), auxfunc_1.knots_eq(), norder = order)
            sublist.append(basis)
        spline_list.append(sublist)
    Y_list = []
    for sample in reformed_data:
        Y_sublist = [np.array(sample[sample['id'] == j]['y']) for j in range(q)]
        Y_list.append(Y_sublist)
    
    # Whether to optimal parameters
    if opt == False:
        
        # ADMM algorithm
        inv_UTY_result = inv_uty_cal(spline_list, Y_list, D_d, n, q, p, gamma1, theta)
        Beta_ini = beta_ini_cal(spline_list, Y_list, D_d, n, q, p, gamma1)
        result = biclustr_admm(inv_UTY_result, data_list, Y_list, D_d, Beta_ini, n, q, p, 
                               gamma1, gamma2, theta, tau, max_iter, eps_abs, eps_rel)
        
        # Clustering
        Ad_final_sam = AuxFunc(n = n, V = result['V1']).create_adjacency() # Row clustering membership
        G_final_sam = nx.from_numpy_array(Ad_final_sam)
        cls_final_sam = list(nx.connected_components(G_final_sam))
        Ad_final_fea = AuxFunc(n = q, V = result['V2']).create_adjacency() # Column clustering membership
        G_final_fea = nx.from_numpy_array(Ad_final_fea)
        cls_final_fea = list(nx.connected_components(G_final_fea))
        result.update({'nknots':nknots,
                       'order':order,
                       'sample cluster':cls_final_sam, 
                       'feature cluster':cls_final_fea,
                       'gamma1':gamma1,
                       'gamma2':gamma2})
        return result
    
    elif opt == True:
        
        # Calculate BIC to find the best gamma1 and gamma2
        # Optimal gamma1
        bic_value_1 = np.zeros(len(gamma1))
        bic_mat = np.zeros((n, q))
        for l in range(len(gamma1)):
            Beta_ini = beta_ini_cal(spline_list, Y_list, D_d, n, q, p, gamma1[l])
            Beta_mat = np.reshape(Beta_ini, (p * q, n), order='F')
            for i in range(n):
                for j in range(q):
                    y_hat = np.dot(spline_list[i][j], Beta_mat[(j * p):(j + 1) * p, i])
                    n_1 = spline_list[i][j].shape[0]
                    df_1 = np.trace(np.dot(spline_list[i][j], solve(np.dot(spline_list[i][j].T, spline_list[i][j]) + gamma1[l] * D_d, spline_list[i][j].T)))
                    bic_mat[i, j] = np.log(np.sum((Y_list[i][j] - y_hat) ** 2) / n_1) + np.log(n_1) * df_1 / n_1
            bic_value_1[l] = np.sum(bic_mat)
        opt_gamma1 = gamma1[np.argmin(bic_value_1)]
        inv_UTY_result = inv_uty_cal(spline_list, Y_list, D_d, n, q, p, gamma1 = opt_gamma1, theta = theta)
        Beta_ini = beta_ini_cal(spline_list, Y_list, D_d, n, q, p, gamma1 = opt_gamma1)
    
        # Optimal gamma2
        bic_value_2 = np.zeros(len(gamma2))
        for l in range(len(gamma2)):
            result = biclustr_admm(inv_UTY_result, data_list, Y_list, D_d, Beta_ini, n, q, p, 
                                   gamma1 = opt_gamma1, gamma2 = gamma2[l], theta = theta, tau = tau, 
                                   max_iter = max_iter, eps_abs = eps_abs, eps_rel = eps_rel)
            
            # Clustering
            Ad_final_sam = AuxFunc(n = n, V = result['V1']).create_adjacency()
            G_final_sam = nx.from_numpy_array(Ad_final_sam)
            cls_final_sam = list(nx.connected_components(G_final_sam))
            Ad_final_fea = AuxFunc(n = q, V = result['V2']).create_adjacency()
            G_final_fea = nx.from_numpy_array(Ad_final_fea)
            cls_final_fea = list(nx.connected_components(G_final_fea))
            vstacked_data  = [np.vstack(sublist).flatten()[:, np.newaxis] for sublist in result['Beta']]
            Beta_mat = np.column_stack(vstacked_data)
            Beta_list = [[Beta_mat[(j * p):(j + 1) * p, i] for j in range(q)] for i in range(n)]
            Y_hat_list = [np.concatenate([np.dot(spline_list[i][j], Beta_list[i][j]) for j in range(q)]) for i in range(n)]
            Y_hat = np.concatenate(Y_hat_list)
            Y_real = [np.concatenate(Y_list[i]) for i in range(n)]
            Y = np.concatenate(Y_real)
            df_mat = np.zeros((n, q))
            for i in range(n):
                for j in range(q):
                    df_mat[i, j] = np.trace(np.dot(spline_list[i][j], solve(np.dot(spline_list[i][j].T, spline_list[i][j]) + opt_gamma1 * D_d, spline_list[i][j].T)))
            df = len(cls_final_sam) * len(cls_final_fea) * np.sum(df_mat)/(n * q)
            bic_value_2[l] = np.log(np.sum((Y - Y_hat) ** 2) / (n * q)) + (np.log(n * q) / (n * q)) * df
        opt_gamma2 = gamma2[np.argmin(bic_value_2)]
        result = biclustr_admm(inv_UTY_result, data_list, Y_list, D_d, Beta_ini, 
                               n, q, p, gamma1=opt_gamma1, gamma2=opt_gamma2, theta = theta, 
                               tau = tau, max_iter = max_iter, eps_abs = eps_abs, eps_rel = eps_rel)
        Ad_final_sam = AuxFunc(n = n, V = result['V1']).create_adjacency()
        G_final_sam = nx.from_numpy_array(Ad_final_sam)
        cls_final_sam = list(nx.connected_components(G_final_sam))
        Ad_final_fea = AuxFunc(n = q, V = result['V2']).create_adjacency()
        G_final_fea = nx.from_numpy_array(Ad_final_fea)
        cls_final_fea = list(nx.connected_components(G_final_fea))
        result.update({'nknots':nknots,
                       'order':order,
                       'sample cluster':cls_final_sam, 
                       'feature cluster':cls_final_fea,
                       'opt_gamma1':opt_gamma1,
                       'opt_gamma2':opt_gamma2})
        return result

    else:
        print('Please enter True or False')    

