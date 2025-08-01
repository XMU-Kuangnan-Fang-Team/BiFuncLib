import numpy as np
from GENetLib.fda_func import create_bspline_basis
from GENetLib.fda_func import eval_basis
from sklearn.cluster import KMeans
from scipy.linalg import norm
from itertools import combinations
import networkx as nx
import math

from GENetLib.BsplineFunc import BsplineFunc
from BiFuncLib.AuxFunc import AuxFunc



# Calculate GCV values from a series of candidate lambda1 to select lambda1
def calculate_gcv(oridata, oritimes, lambda1_seq, rangeval=(0, 1), nknots=30, order=4, nu=2):
    
    # Create B-spline basis
    n = len(oridata)
    p = nknots + order
    basisobj = create_bspline_basis(rangeval = rangeval, nbasis = p, norder = order)
    Z_lst = []
    for i in range(n):
        Z_lst.append(eval_basis(oritimes[i], basisobj))

    # Calculate penaty matrix D
    D = BsplineFunc(basisobj = basisobj, Lfdobj = nu).penalty_matrix()
    nlambda1 = len(lambda1_seq)
    gcv_value = np.zeros(nlambda1)
    for l in range(nlambda1):
        term1 = np.zeros(n)
        term2 = np.zeros(n)
        error = np.zeros(n)
        for i in range(n):
            ni = len(oridata[i])
            Hi = Z_lst[i] @ np.linalg.solve(Z_lst[i].T @ Z_lst[i] + lambda1_seq[l] * D, Z_lst[i].T)
            term1[i] = np.mean(np.square(oridata[i] - Hi @ oridata[i]))
            term2[i] = np.mean(np.square(np.diag(np.eye(ni) - Hi)))
            error[i] = term1[i] / term2[i]
        gcv_value[l] = np.sum(error)
    lambda1_op = lambda1_seq[np.argmin(gcv_value)]
    return lambda1_op


# Generate the initial smoothed B-spline coefficients and the centered coefficients
def generate_inicoef(oridata, oritimes, rangeval=(0, 1), nknots=30, order=4, nu=2, lambda1=1e-5, K0=6, rep_num=100):
    
    # Initialization
    n = len(oridata)
    p = nknots + order
    basisobj = create_bspline_basis(rangeval=rangeval, nbasis=p, norder=order)
    Z_lst = [eval_basis(oritimes[i], basisobj) for i in range(n)]
    D = BsplineFunc(basisobj = basisobj, Lfdobj = nu).penalty_matrix()
    Beta_ini = np.zeros((p, n))
    for i in range(n):
        Z_i = Z_lst[i]
        Beta_ini[:, i] = (np.linalg.solve(Z_i.T @ Z_i + lambda1 * D, (Z_i.T @ oridata[i]).T)).flatten()

    # KMeans
    error = np.zeros(rep_num)
    center_lst = [None] * rep_num
    cluster_lst = [None] * rep_num
    for l in range(rep_num):
        ini_cluster = KMeans(n_clusters=K0, n_init = 25).fit(Beta_ini.T)
        center_lst[l] = ini_cluster.cluster_centers_.T
        cluster_lst[l] = ini_cluster.labels_
        obj_value = np.zeros(n)
        for i in range(n):
            cluster_idx = cluster_lst[l][i]
            center_i = center_lst[l][:, cluster_idx]
            obj_value[i] = np.sum(np.square(Z_lst[i] @ center_i - oridata[i]))
        error[l] = np.mean(obj_value)

    # Select by KMeans results
    best_idx = np.argmin(error)
    Alpha_ini = center_lst[best_idx]
    return {"Beta_ini": Beta_ini, "Alpha_ini": Alpha_ini}
  
  
# Calculate varsigma1
def cal_varsigma1(Z_lst, D, lambda1):
    n = len(Z_lst)
    Lip_vec = np.zeros(n)
    for i in range(n):
        term = Z_lst[i].T @ Z_lst[i]
        eigen_values = np.linalg.eigvals(term)
        Lip_vec[i] = np.max(eigen_values)
    Lip = np.max(Lip_vec) + lambda1 * np.max(np.linalg.eigvals(D))
    varsigma1 = 1 / Lip
    if np.isclose(varsigma1.imag, 0):
        varsigma1 = varsigma1.real
    return varsigma1


# Calculate varsigma2
def cal_varsigma2(Omega, kappa):
    term = Omega.T @ Omega
    eigen_values = np.linalg.eigvals(term)
    Lip = kappa * np.max(eigen_values)
    varsigma2 = 1 / Lip
    if np.isclose(varsigma2.imag, 0):
        varsigma2 = varsigma2.real
    return varsigma2


# The proximal (average) mapping of mcp function
def proximal_MCP(x, varsigma, tau=3, lambda_=1):
    p = len(x)
    x_norm = np.linalg.norm(x)
    if x_norm <= lambda_ * varsigma:
        z = np.zeros(p)
    elif x_norm <= lambda_ * tau:
        z = max(0, x_norm - lambda_ * varsigma) / (1 - varsigma / tau) * (x / x_norm)
    else:
        z = x
    return z


# The proximal average mapping of mcp function
def proximal_average_MCP(x, l, r, varsigma, tau=3, lamda=1):
    p = len(x)
    xl_norm = np.linalg.norm(x[l:l + r + 1])
    z = np.zeros(p)
    for j in range(p):
        if (l <= j <= l + r):
            if xl_norm <= lamda * varsigma:
                z[j] = 0
            elif xl_norm <= lamda * tau:
                z[j] = (tau / (tau - varsigma)) * max(1 - varsigma * lamda / xl_norm, 0) * x[j]
            else:
                z[j] = x[j]
        else:
            z[j] = x[j]
    return z


# Function to assign center label to each individual
def assign_center_label(u, Alpha):
    K0 = Alpha.shape[1]
    dis_vec = np.zeros(K0)
    for k in range(K0):
        dis_vec[k] = np.sum(np.square(u - Alpha[:, k]))
    return np.argmin(dis_vec)


# Function to update Beta
def update_Beta(ZTY_lst, ZTZ_lst, Beta, Alpha, D, tau, varsigma1, lambda1, lambda2, iter_outer, max_iter_inner):
    p, n = Beta.shape
    U_middle = np.zeros((p, n))
    V_middle = np.zeros((p, n))
    Beta_old = np.zeros((p, n))
    Beta_new = np.zeros((p, n))
    for i in range(n):
        eta_old = 1
        U_middle[:, i] = Beta[:, i]
        Beta_old[:, i] = U_middle[:, i]
        iter_inner = 1
        res_inner = 1
        eps_inner = 0.01 / iter_outer
        while res_inner > eps_inner and iter_inner <= max_iter_inner:
            V_middle[:, i] = U_middle[:, i] - varsigma1 * (ZTZ_lst[i].dot(U_middle[:, i]) - ZTY_lst[i] + lambda1 * D.dot(U_middle[:, i]))
            center_index_opt = assign_center_label(V_middle[:, i], Alpha)
            Beta_new[:, i] = proximal_MCP(V_middle[:, i] - Alpha[:, center_index_opt], varsigma1, tau, lambda2) + Alpha[:, center_index_opt]
            eta_new = (1 + np.sqrt(1 + 4 * eta_old**2)) / 2
            U_middle[:, i] = Beta_new[:, i] + (eta_old - 1) / eta_new * (Beta_new[:, i] - Beta_old[:, i])
            res_inner = np.sqrt(np.sum((Beta_new[:, i] - Beta_old[:, i])**2))
            eta_old = eta_new
            iter_inner += 1
            Beta_old[:, i] = Beta_new[:, i]
    return Beta_new


# Function to update Alpha
def update_Alpha(Beta, Alpha, Psi, Gamma, Omega, tau, kappa, varsigma2, lambda2, iter_outer, max_iter_inner):
    p, n = Beta.shape
    K0 = Alpha.shape[1]
    Alpha_old = Alpha.copy()
    Alpha_new = Alpha.copy()
    u_middle = Alpha.T.flatten()
    eta_old = 1
    iter_inner = 1
    res_inner = 1
    eps_inner = 0.01 / iter_outer
    OmegaTpsi = Omega.T @ Psi.T.flatten()
    OmegaTgamma = Omega.T @ Gamma.T.flatten() / kappa
    OmegaTOmega = Omega.T @ Omega
    while res_inner > eps_inner and iter_inner <= max_iter_inner:
        v_middle = u_middle - varsigma2 * kappa / n * (OmegaTOmega @ u_middle - OmegaTpsi + OmegaTgamma)
        V_middle = v_middle.reshape(p, K0, order='F')
        index_mat = np.zeros((n, K0))
        for i in range(n):
            dis = np.zeros(K0)
            for k in range(K0):
                dis[k] = np.sqrt(np.sum((Beta[:, i] - V_middle[:, k])**2))
            index_mat[i, np.argmin(dis)] = 1
        for k in range(K0):
            if np.sum(index_mat[:, k] == 1) == 0:
                Alpha_new[:, k] = V_middle[:, k]
            else:
                Alpha_new[:, k] = np.zeros(p)
                for i in range(n):
                    if index_mat[i, k] == 1:
                        Alpha_new[:, k] += proximal_MCP(V_middle[:, k] - Beta[:, i], varsigma2, tau, lambda2) + Beta[:, i]
                Alpha_new[:, k] /= np.sum(index_mat[:, k] == 1)
        eta_new = (1 + np.sqrt(1 + 4 * eta_old ** 2)) / 2
        u_middle = Alpha_new.T.flatten() + (eta_old - 1) / eta_new * (Alpha_new - Alpha_old).T.flatten()
        res_inner = norm(Alpha_new - Alpha_old, ord='fro')
        eta_old = eta_new
        iter_inner += 1
        Alpha_old = Alpha_new.copy()
    return Alpha_new


# Function to update Psi
def update_Psi(Psi, Alpha, Gamma, index, n, L, r, tau, varsigma3, kappa, lambda3, iter_outer, max_iter_inner):
    p, S = Psi.shape
    U_middle = np.zeros((p, S))
    V_middle = np.zeros((p, S))
    Psi_old_ = np.zeros((p, S))
    Psi_new = np.zeros((p, S))
    for s in range(S):
        eta_old = 1
        U_middle[:, s] = Psi[:, s]
        Psi_old_[:, s] = U_middle[:, s]
        iter_inner = 1
        res_inner = 1
        eps_inner = 0.01 / iter_outer
        while res_inner > eps_inner and iter_inner <= max_iter_inner:
            V_middle[:, s] = U_middle[:, s] - varsigma3 * kappa * (U_middle[:, s] - (Alpha[:, index[s, 0]] - Alpha[:, index[s, 1]] + Gamma[:, s] / kappa)) / (n * L)
            Psi_new[:, s] = np.zeros(p)
            for l in range(L):
                Psi_new[:, s] += proximal_average_MCP(V_middle[:, s], l, r, varsigma3, tau, lambda3)
            Psi_new[:, s] /= L
            for l in range(L):
                Psi_new[l:(l + r + 1), s] = proximal_MCP(Psi_new[l:(l + r + 1), s], varsigma3, tau, lambda3)
            eta_new = (1 + np.sqrt(1 + 4 * eta_old**2)) / 2
            U_middle[:, s] = Psi_new[:, s] + (eta_old - 1) / eta_new * (Psi_new[:, s] - Psi_old_[:, s])
            res_inner = np.sqrt(np.sum((Psi_new[:, s] - Psi_old_[:, s])**2))
            Psi_old_[:, s] = Psi_new[:, s]
            eta_old = eta_new
            iter_inner += 1
    return Psi_new


# Function to update Gamma
def update_Gamma(Gamma_old, Alpha, Psi, index, kappa):
    p, S = Gamma_old.shape
    Gamma = np.zeros((p, S))
    for s in range(S):
        i, j = index[s, 0], index[s, 1]
        Gamma[:, s] = Gamma_old[:, s] + kappa * (Alpha[:, i] - Alpha[:, j] - Psi[:, s])
    return Gamma


# Main proximal average ADMM algorithm 
def local_admm(oridata, oritimes, lambda1, lambda2, lambda3, rangeval = (0, 1), nknots = 30, order = 4,
               nu = 2, tau = 3, K0 = 6, rep_num = 100, kappa = 1, eps_outer = 0.0001, max_iter = 100):
    
    # Initialization
    index = np.array(list(combinations(np.arange(K0), 2)))
    S = index.shape[0]
    n = len(oridata)
    p = nknots + order
    L = nknots + 1
    r = order - 1
    basisobj = create_bspline_basis(rangeval = rangeval, nbasis = p, norder = order)
    Z_lst = [None] * n
    Y_lst = [None] * n
    for i in range(n):
        Z_lst[i] = eval_basis(oritimes[i], basisobj)
        Y_lst[i] = oridata[i]
    D = BsplineFunc(basisobj = basisobj, Lfdobj = nu).penalty_matrix()
    coef_ini = generate_inicoef(oridata = oridata, oritimes = oritimes, rangeval = rangeval, nknots = nknots,
                                order = order, nu = nu, lambda1 = lambda1, K0 = K0, rep_num = rep_num)
    Beta_old = coef_ini['Beta_ini']
    Alpha_old = coef_ini['Alpha_ini']
    ZTY_lst = [None] * n
    ZTZ_lst = [None] * n
    for i in range(n):
        ZTY_lst[i] = Z_lst[i].T @ Y_lst[i]
        ZTZ_lst[i] = Z_lst[i].T @ Z_lst[i]
    Psi_old = np.zeros((p, S))
    Omega_lst = [None] * S
    for s in range(S):
        i = index[s, 0]
        j = index[s, 1]
        Psi_old[:, s] = Alpha_old[:, i] - Alpha_old[:, j]
        e = np.zeros(K0)
        e[i] = 1
        e[j] = -1
        Omega_lst[s] = np.kron(e.T, np.eye(p))
    Omega = np.vstack(Omega_lst)
    Gamma_old = np.zeros((p, S))
    varsigma1 = cal_varsigma1(Z_lst, D, lambda1)
    varsigma2 = cal_varsigma2(Omega, kappa)
    varsigma3 = 1/kappa
    if tau <= varsigma1 or tau <= varsigma2 or tau <= varsigma3:
        print("error stepsize")
    res_outer = 1
    iters = 1
    
    # Perform loop iteration
    while res_outer > eps_outer and iters <= max_iter:
        Beta_new = update_Beta(ZTY_lst, ZTZ_lst, Beta_old, Alpha_old, D, tau, varsigma1, lambda1, lambda2, iters, max_iter_inner=5)
        Alpha_new = update_Alpha(Beta_new, Alpha_old, Psi_old, Gamma_old, Omega, tau, kappa, varsigma2, lambda2, iters, max_iter_inner=5)
        Psi_new = update_Psi(Psi_old, Alpha_new, Gamma_old, index, n, L, r, tau, varsigma3, kappa, lambda3, iters, max_iter_inner=5)
        Gamma_new = update_Gamma(Gamma_old, Alpha_new, Psi_new, index, kappa)
        res_outer_vec = np.zeros(S)
        for s in range(S):
            i = index[s, 0]
            j = index[s, 1]
            res_outer_vec[s] = np.sum(np.square(Alpha_new[:, i] - Alpha_new[:, j] - Psi_new[:, s]))
        res_outer = np.sqrt(np.sum(res_outer_vec))
        res_outer = norm(Beta_new - Beta_old, 'fro') / norm(Beta_old, 'fro')
        
        # Function to generate membership of clustering
        def convert_to_class_list(clusters):
            element_to_class = {}
            for class_id, cluster in enumerate(clusters):
                for element in cluster:
                    element_to_class[element] = class_id
            result = [element_to_class[i] for i in sorted(element_to_class.keys())]
            return result
        
        for l in range(L):
            Ad_final = AuxFunc(n = K0, V = Psi_new[l:(l + r + 1), :]).create_adjacency(plot = False)
            if np.sum(Ad_final) != 0:
                G_final = nx.from_numpy_array(Ad_final)
                cls_final = list(nx.connected_components(G_final))
                for g in range(len(cls_final)):
                    cols = [x == g for x in convert_to_class_list(cls_final)]
                    sub_matrix = Alpha_new[l:(l + r + 1), cols]
                    row_means = np.mean(sub_matrix, axis=1)
                    Alpha_new[l:(l + r + 1), cols] = row_means[:, np.newaxis] 
        Beta_old = Beta_new.copy()
        Alpha_old = Alpha_new.copy()
        Psi_old = Psi_new.copy()
        Gamma_old = Gamma_new.copy()
        iters += 1
    for i in range(n):
        center_index_opt = assign_center_label(Beta_new[:, i], Alpha_new)
        Beta_new[:, i] = proximal_MCP(Beta_new[:, i] - Alpha_new[:, center_index_opt], varsigma1, tau, lambda2) + Alpha_new[:, center_index_opt]
    group = np.zeros(n, dtype=int)
    for i in range(n):
        dis_vec = np.zeros(K0)
        for k in range(K0):
            dis_vec[k] = np.sum(np.square(Beta_new[:, i] - Alpha_new[:, k]))
        group[i] = np.argmin(dis_vec)
    Beta_trans = np.zeros((p, n))
    for g in np.unique(group):
        Beta_trans[:, group == g] = Alpha_new[:, g].reshape(-1, 1)
    for j in range(p):
        beta_row = Beta_trans[j, :]
        n = len(beta_row)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                dist_mat[i, k] = np.abs(beta_row[i] - beta_row[k])
        adj_vec = dist_mat[np.triu_indices_from(dist_mat, k=1)]
        adj_vec[np.abs(adj_vec) <= 0.2] = 0
        adj_mat = adj_vec.reshape(1, -1)
        Ad_final = AuxFunc(n = n, V = adj_mat).create_adjacency(plot = False)
        G_final = nx.from_numpy_array(Ad_final)
        cls_final = list(nx.connected_components(G_final))
        for k in range(len(cls_final)):
            cols = [x == k for x in convert_to_class_list(cls_final)]
            Beta_trans[j, cols] = np.mean(Beta_trans[j, cols])
    cls_num = len(np.unique(group))
    Alpha_trans = np.zeros((p, cls_num))
    unique_values, indices = np.unique(group, return_index=True)
    unique_values = unique_values[np.argsort(indices)]
    for k in range(cls_num):
        Alpha_trans[:, k] = np.mean(Beta_trans[:, group == unique_values[k]], axis=1)
    cls_mem = np.zeros(n, dtype=int)
    for i in range(n):
        dis_vec = np.zeros(cls_num)
        for k in range(cls_num):
            dis_vec[k] = np.sum(np.square(Beta_trans[:, i] - Alpha_trans[:, k]))
        cls_mem[i] = np.argmin(dis_vec)

    # Return results
    return {
        'Beta': Beta_trans,
        'Beta_ini': coef_ini['Beta_ini'],
        'centers': Alpha_trans,
        'basisobj': basisobj,
        'cls_mem': cls_mem,
        'cls_num': cls_num}


# Calculate BIC values and generate the corresponding clustering results
def calculate_bic(oridata, oritimes, lambda1, lambda2_seq, lambda3_seq, rangeval = (0, 1), 
                 nknots = 30, order = 4, nu = 2, tau = 3, K0 = 6, rep_num = 100,
                 kappa=1, eps_outer=0.0001, max_iter=100):
    
    n = len(oridata)
    p = nknots + order
    N = 0
    basisobj = create_bspline_basis(rangeval=rangeval, nbasis=p, norder=order)
    Z_lst = [None] * n
    for i in range(n):
        Z_lst[i] = eval_basis(oritimes[i], basisobj)
        N += Z_lst[i].shape[0]
    D = BsplineFunc(basisobj = basisobj, Lfdobj = nu).penalty_matrix()
    nlambda2 = len(lambda2_seq)
    nlambda3 = len(lambda3_seq)
    funlc_mat_lst = [[None for _ in range(nlambda3)] for _ in range(nlambda2)]
    z = 0
    total_num = nlambda2 * nlambda3
    for l1 in range(nlambda2):
        for l2 in range(nlambda3):
            error = np.zeros(n)
            funlc_result = local_admm(oridata=oridata, oritimes=oritimes, rangeval=rangeval, nknots=nknots, order=order,
                                      nu=nu, tau=tau, K0=K0, rep_num=rep_num, lambda1=lambda1, lambda2=lambda2_seq[l1],
                                      lambda3=lambda3_seq[l2], kappa=kappa, eps_outer=eps_outer, max_iter=max_iter)
            cls_no = funlc_result['cls_num']
            Beta = funlc_result['Beta']
            error_vec = np.zeros(n)
            df_vec = np.zeros(n)
            for i in range(n):
                error_vec[i] = np.sum(np.square(oridata[i] - np.dot(Z_lst[i], Beta[:, i])))
                Hi = np.dot(Z_lst[i], np.linalg.solve(np.dot(Z_lst[i].T, Z_lst[i]) + lambda1 * D, Z_lst[i].T))
                df_vec[i] = np.sum(np.diag(Hi))
            error = math.log(np.sum(error_vec) / N)
            df = np.sum(df_vec) * cls_no / n
            bn = math.log(math.log(n * p))
            bic_val = error + bn * math.log(N) * df / N
            funlc_mat_lst[l1][l2] = {'funlc_result': funlc_result, 'bic_val': bic_val}
            z += 1
            progress = 100 * z / total_num
            if progress % 10 == 0:
                print(f"Finishing: {progress:.1f}%")
    return funlc_mat_lst

