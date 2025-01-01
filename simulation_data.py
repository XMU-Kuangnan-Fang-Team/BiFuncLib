import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from AuxFunc import AuxFunc
from GENetLib.basis_mat import bspline_mat


# Generate simulation data for pf_bifunc
def pf_sim_data(n, T, nknots, order, seed = 123):
    np.random.seed(seed)
    q = 9
    class1 = np.arange(n/3)
    class2 = np.arange(n/3, n/3*2)
    class3 = np.arange(n/3*2, n)
    t = np.linspace(0, 1, T)
    c1_1 = np.cos(2 * np.pi * t)
    c1_2 = 1 + np.sin(2 * np.pi * t)
    c1_3 = 2 * (np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t))
    c2_1 = 1 - 2 * np.exp(-6 * t)
    c2_2 = 2 * t**2
    c2_3 = 1 + t**3
    c3_1 = -1.5 * t
    c3_2 = t + 1
    c3_3 = 2 * np.sqrt(t) + 1
    sigma = 0.6
    oridata_list = [[] for _ in range(n)]
    for i in range(n):
        if i in class1:
            oridata_list[i] = [
                c1_1 + np.random.normal(0, sigma, T),
                c1_1 + np.random.normal(0, sigma, T),
                c1_1 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T)
            ]
        elif i in class2:
            oridata_list[i] = [
                c2_1 + np.random.normal(0, sigma, T),
                c2_1 + np.random.normal(0, sigma, T),
                c2_1 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T)
            ]
        elif i in class3:
            oridata_list[i] = [
                c3_1 + np.random.normal(0, sigma, T),
                c3_1 + np.random.normal(0, sigma, T),
                c3_1 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T)
            ]
    
    # Generate a data list to contain the time information in each sample list    
    times = [[] for _ in range(n)]
    for i in range(n):
        times[i] = [t] * q
    
    # Generate a sample list to contain the all sample information in each dataframe
    sample_list = [[] for _ in range(n)]
    id_matrix = np.zeros((n, q), dtype=int)
    sample_list = []
    for i in range(n):
        for j in range(q):
            id_matrix[i, j] = len(oridata_list[i][j])
        data = {'id': np.repeat(np.arange(q), id_matrix[i, :]),
                'time': np.concatenate(times[i]),
                'y': np.concatenate(oridata_list[i])}
        df = pd.DataFrame(data)
        sample_list.append(df)
    
    # Generate a design matrix for sample basis
    timerange = np.linspace(0, 1, num = T)
    auxfunc_1 = AuxFunc(n = n, m = nknots, x = timerange)
    spline_list = []
    for i, sample in enumerate(sample_list):
        sublist = []
        for _, group in sample.groupby('id'):
            basis = bspline_mat(np.array(group['time']), auxfunc_1.knots_eq(), norder = order)
            sublist.append(basis)
        spline_list.append(sublist)
    
    # Generate response vector
    Y_list = []
    for sample in sample_list:
        Y_sublist = [np.array(sample[sample['id'] == j]['y']) for j in range(q)]
        Y_list.append(Y_sublist)
    
    # Construct missing samples under balanced data
    miss_percent = 0.3
    miss_meas = 0.2
    subgroup_sam = 3
    subgroup_fea = 3
    cluster_struc_sam = np.array([int(n/3), int(n/3), int(n/3)])
    cluster_struc_fea = np.array([3, 3, 3])
    cluster_gap_sam = np.array([0, int(n/3), int(n/3*2)])
    cluster_gap_fea = np.array([0, 3, 6])
    for i in range(subgroup_sam):
        for j in range(subgroup_fea):
            num_elem = int(cluster_struc_sam[i] * cluster_struc_fea[j])
            num_miss = int(np.ceil(num_elem * miss_percent))
            id1 = np.random.choice(num_elem, num_miss, replace=False)
            id_loac = np.zeros((num_miss, 2), dtype=int)
            mat = np.arange(num_elem).reshape(cluster_struc_sam[i], cluster_struc_fea[j])
            for l, val in enumerate(id1):
                idx = np.where(mat == val)
                id_loac[l, :] = [idx[0][0], idx[1][0]]
            for l in range(len(id1)):
                i1 = id_loac[l, 0]
                j1 = id_loac[l, 1]
                id2 = np.sort(np.random.choice(T, int((1-miss_meas)*T), replace=False))
                times[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = t[id2]
                basis_new = bspline_mat(t[id2], auxfunc_1.knots_eq(), norder = order)
                spline_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = basis_new
                Y_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = Y_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]][id2]
    
    # Generate censored sample list
    censored_sample_list = []
    merged_Y_list = []
    for Y in Y_list:
        merged_Y_list.append([item for sublist in Y for item in sublist])
    for df, merged_Y in zip(sample_list, merged_Y_list):
        df_censored = df[df['y'].isin(merged_Y)]
        censored_sample_list.append(df_censored)
    
    # Generate censored sample matrix
    censored_none_list = []
    for i in range(len(censored_sample_list)):
        result_df = pd.DataFrame()
        for j in range(q):
            df = censored_sample_list[i][censored_sample_list[i]['id'] == j]
            df = df.set_index('time').reindex(timerange).reset_index()
            df['y'] = df['y'].apply(lambda x: x if not pd.isnull(x) else None)
            df['id'] = j
            result_df = pd.concat([result_df, df], ignore_index=True)
        censored_none_list.append(result_df)
    censored_sample_matrix = pd.DataFrame()
    for i in range(len(censored_none_list)):
        pivot_df = censored_none_list[i].pivot_table(index='time', columns='id', values='y')
        pivot_df.index = range(T)   
        pivot_df.reset_index(inplace=True)
        pivot_df.rename(columns={'index': 'time'}, inplace=True)
        time_index = pivot_df.columns.get_loc('time')
        pivot_df.insert(loc=time_index + 1, column = 'measurement', value = i)
        censored_sample_matrix = pd.concat([censored_sample_matrix, pivot_df], axis = 0)

    # Order and return result
    censored_sample_matrix = censored_sample_matrix.sort_values(by='time', ascending=True)
    def sort_measurement(group):
        return group.sort_values(by='measurement', ascending=True)
    censored_sample_matrix = censored_sample_matrix.groupby('time', group_keys=False).apply(sort_measurement).reset_index(drop=True)
    return {'data': censored_sample_matrix,
            'location': t,
            'feature cluster': [set(range(0,3)), set(range(3,6)), set(range(6,9))],
            'sample cluster': [set(range(0,int(n/3))), set(range(int(n/3),int(n/3*2))), set(range(int(n/3*2),n))]}

'''
pf_simdata = pf_sim_data(n = 30, T = 10, nknots = 3, order = 3, seed = 123)
'''


def local_sim_data(n, T, sigma, seed = 123):
    np.random.seed(seed)
    Times = np.linspace(0, 1, T)
    class1 = np.arange(2*n//5)
    class2 = np.arange(2*n//5, 7*n//10)
    class3 = np.arange(7*n//10, n)
    setpoint1, setpoint2, setpoint3 = 0.2, 0.6, 1.0
    times = [Times.copy() for _ in range(n)]
    oridata_list = [None] * n
    mu = np.zeros(T)
    rho1 = 0.3
    sig = np.zeros((T, T))
    for i1 in range(T):
        for i2 in range(T):
            sig[i1, i2] = sigma**2 * rho1**abs(i1 - i2)
    
    # Generate 3 clusters
    for i in range(n):
        times[i] = Times
        if i in class1:
            c1 = np.zeros(T)
            for l in range(T):
                if Times[l] >= 0 and Times[l] <= setpoint1:
                    c1[l] = 1 + 1.5 * np.sin(np.pi * (Times[l] - 0) / (setpoint1 - 0))
                elif Times[l] <= setpoint2:
                    c1[l] = 1
                else:
                    c1[l] = 1 - 1.5 * np.sin(np.pi * (Times[l] - setpoint2) / (setpoint3 - setpoint2))
            epsilon = multivariate_normal.rvs(mean=mu, cov=sig)
            oridata_list[i] = c1 + epsilon
        elif i in class2:
            c2 = np.zeros(T)
            for l in range(T):
                if Times[l] >= 0 and Times[l] <= setpoint1:
                    c2[l] = 1 + 1.5 * np.sin(np.pi * (Times[l] - 0) / (setpoint1 - 0))
                elif Times[l] <= setpoint2:
                    c2[l] = 1 - 1.5 * np.sin(np.pi * (Times[l] - setpoint1) / (setpoint2 - setpoint1))
                else:
                    c2[l] = 1
            epsilon = multivariate_normal.rvs(mean=mu, cov=sig)
            oridata_list[i] = c2 + epsilon
        elif i in class3:
            c3 = np.zeros(T)
            for l in range(T):
                if Times[l] >= 0 and Times[l] <= setpoint1:
                    c3[l] = 1 + 1.5 * np.sin(np.pi * (Times[l] - 0) / (setpoint1 - 0))
                elif Times[l] <= setpoint2:
                    c3[l] = 1 - 1.5 * np.sin(np.pi * (Times[l] - setpoint1) / (setpoint2 - setpoint1))
                else:
                    c3[l] = 1 + 1.5 * np.sin(np.pi * (Times[l] - setpoint2) / (setpoint3 - setpoint2))
            epsilon = multivariate_normal.rvs(mean=mu, cov=sig)
            oridata_list[i] = c3 + epsilon
            
    # Return result        
    return {'data': oridata_list,
            'location': times,
            'sample cluster': [set(class1), set(class2), set(class3)]}

'''
local_simdata = local_sim_data(n = 100, T = 100, sigma = 0.5, seed = 42)
'''

