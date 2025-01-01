import numpy as np
import pandas as pd
import networkx as nx
from AuxFunc import AuxFunc
from pf_main_func import inv_uty_cal, beta_ini_cal, biclustr_admm
from GENetLib.basis_mat import bspline_mat


def pf_bifunc(data, nknots, order, gamma1, gamma2, theta, tau, max_iter, eps_abs, eps_rel):
    
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
    
    # ADMM algorithm
    inv_UTY_result = inv_uty_cal(spline_list, Y_list, D_d, n, q, p, gamma1, theta)
    Beta_ini = beta_ini_cal(spline_list, Y_list, D_d, n, q, p, gamma1)
    result = biclustr_admm(inv_UTY_result, data_list, Y_list, D_d, Beta_ini, n, q, p, 
                           gamma1, gamma2, theta, tau, max_iter, eps_abs, eps_rel)
    
    # Clustering
    Ad_final_sam = AuxFunc(n = n, V = result['V1']).create_adjacency() # Row clustering membership
    G_final_sam = nx.from_numpy_array(Ad_final_sam)
    cls_final_sam = list(nx.connected_components(G_final_sam))
    Ad_final_fea = AuxFunc(n =  q, V = result['V2']).create_adjacency() # Column clustering membership
    G_final_fea = nx.from_numpy_array(Ad_final_fea)
    cls_final_fea = list(nx.connected_components(G_final_fea))
    result.update({'nknots':nknots,
                   'order':order,
                   'sample cluster':cls_final_sam, 
                   'feature cluster':cls_final_fea})
    return result


'''
# Test
from simulation_data import pf_sim_data
pf_simdata = pf_sim_data(n = 30, TT = 10, nknots = 3, order = 3, seed = 123)['data']
pf_result = pf_bifunc(pf_simdata, nknots = 3, order = 3, gamma1 = 0.023, gamma2 = 3, 
                      theta = 1, tau = 3, max_iter = 500, eps_abs = 1e-3, eps_rel = 1e-3)
'''


'''
# Input same data with R
import os
censored_sample_list = []
directory = "C:/Users/YYBG-WANG/Desktop/科研/4.Biclustering_python/对照数据/censored_sample"
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('censored_sample_list')]
csv_files.sort(key=lambda x: int(x.split('censored_sample_list')[1].split('.csv')[0]))
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    df = pd.read_csv(file_path)
    df['id'] =  df['id'] - 1
    censored_sample_list.append(df)
TT = 10; n = 30; q = 9; nknots = 3; order = 3
timerange = np.linspace(0, 1, num=TT)
auxfunc_1 = AuxFunc(n = n, m = nknots, x = timerange)
data_list = []
for i, sample in enumerate(censored_sample_list):
    sublist = []
    for _, group in sample.groupby('id'):
        basis = bspline_mat(np.array(group['time']), auxfunc_1.knots_eq(), norder = order)
        sublist.append(basis)
    data_list.append(sublist)
Y_list = []
for sample in censored_sample_list:
    Y_sublist = [np.array(sample[sample['id'] == j]['y']) for j in range(q)]
    Y_list.append(Y_sublist)
'''

