import numpy as np
from math import exp, sqrt
from scipy.sparse import csc_matrix, coo_matrix
import networkx as nx
import random
import math


def kernel_weights(X, phi):
    p, n = X.shape
    num_weights = n * (n - 1) // 2
    w = np.empty(num_weights, dtype=float)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = X.iloc[:, i] - X.iloc[:, j]
            sos = np.dot(diff, diff)
            w[k] = exp(-phi * sos)
            k += 1
    return w


def spmm(M, X):
    m = M["Nrow"]
    n = M["Ncol"]
    p = X.shape[1]
    Y = np.zeros((m, p), dtype=float)
    for i in range(p):
        for j in range(n):
            for k in range(M["column_ptr"][j], M["column_ptr"][j+1]):
                row_ind = M["row_indices"][k]
                Y[row_ind, i] += M["values"][k] * X[j, i]
    return Y


def spmtm(M, X):
    n = M["Ncol"]
    p = X.shape[1]
    Y = np.zeros((n, p), dtype=float)
    for i in range(p):
        for j in range(n):
            for k in range(M["column_ptr"][j], M["column_ptr"][j+1]):
                row_ind = M["row_indices"][k]
                Y[j, i] += M["values"][k] * X[row_ind, i]
    return Y


def spmmt(M, X):
    m = M["Nrow"]
    n = M["Ncol"]
    p = X.shape[0]
    Y = np.zeros((m, p), dtype=float)
    for i in range(p):
        for j in range(n):
            for k in range(M["column_ptr"][j], M["column_ptr"][j+1]):
                row_ind = M["row_indices"][k]
                Y[row_ind, i] += M["values"][k] * X[i, j]
    return Y


def convex_cluster_dual(XT, UT):
    dual = 0.5 * (np.sum(XT**2) - np.sum(UT**2))
    return dual


def convex_cluster_primal(XT, UT, VT, Phi, w):
    primal = 0.5 * np.sum((XT - UT)**2)
    VT[:] = spmm(Phi, UT)
    penalty = np.sum(w * np.sqrt(np.sum(VT**2, axis=1)))
    return primal + penalty


def convex_bicluster_primal(XT, UT, VT_row, VT_col, Phi_row, Phi_col, w_row, w_col):
    primal = 0.5 * np.sum((XT - UT) ** 2)
    VT_row[:] = spmmt(Phi_row, UT)
    VT_col[:] = spmm(Phi_col, UT)
    penalty_row = np.sum(w_row * np.sqrt(np.sum(VT_row ** 2, axis=1)))
    penalty_col = np.sum(w_col * np.sqrt(np.sum(VT_col ** 2, axis=1)))
    return primal + penalty_row + penalty_col


def prox_L2(X, tau):
    m, n = X.shape
    Y = np.empty_like(X)
    for i in range(m):
        norm_val = np.linalg.norm(X[i, :])
        if norm_val == 0:
            Y[i, :] = X[i, :]
        else:
            factor = max(0, 1 - (tau[i] / norm_val))
            Y[i, :] = factor * X[i, :]
    return Y


def proj_L2(X, tau):
    m, n = X.shape
    Y = np.empty_like(X)
    for i in range(m):
        norm_val = np.linalg.norm(X[i, :])
        if norm_val > tau[i]:
            factor = tau[i] / norm_val
        else:
            factor = 1.0
        Y[i, :] = factor * X[i, :]
    return Y


def update_UT(XT, LambdaT, Phi):
    UT = spmtm(Phi, LambdaT)
    UT = XT - UT
    return UT


def grad_LambdaT(UT, Phi):
    gLambdaT = spmm(Phi, UT)
    return -gLambdaT


def update_LambdaT2(LambdaT, gLambdaT, nu, w):
    LambdaT_temp = LambdaT - nu * gLambdaT
    new_LambdaT = proj_L2(LambdaT_temp, w)
    return new_LambdaT


def update_LambdaT(LambdaT, UT, Phi, nu, w):
    LambdaT_temp = spmm(Phi, UT)
    LambdaT_temp = LambdaT + nu * LambdaT_temp
    new_LambdaT = proj_L2(LambdaT_temp, w)
    return new_LambdaT


def update_VT_row(U, LambdaT, Phi, w, nu):
    tau = w / nu
    VT_temp = spmm(Phi, U) - (1/nu) * LambdaT
    VT_row = prox_L2(VT_temp, tau)
    return VT_row


def update_VT_col(UT, LambdaT, Phi, w, nu):
    tau = w / nu
    VT_temp = spmm(Phi, UT) - (1/nu) * LambdaT
    VT_col = prox_L2(VT_temp, tau)
    return VT_col


def tri2vec(i, j, n):
    return n * (i - 1) - (i * (i - 1)) // 2 + j - i


def vec2tri(k, n):
    k = np.asarray(k)
    i = np.ceil(0.5 * (2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * k))).astype(int)
    j = k - n * (i - 1) + (i * (i - 1)) // 2 + i
    return np.column_stack((i, j))


def gkn_weights(X, phi=0.5, k_row=5, k_col=5):
    p, n = X.shape
    w_row = kernel_weights(X.T, phi / n)
    w_col = kernel_weights(X, phi / p)
    w_row = knn_weights(w_row, k_row, p)
    w_col = knn_weights(w_col, k_col, n)
    w_row = w_row / w_row.sum()
    w_col = w_col / w_col.sum()
    w_row = w_row / np.sqrt(n)
    w_col = w_col / np.sqrt(p)
    E_row = create_edge_incidence(w_row, p)
    E_col = create_edge_incidence(w_col, n)
    A_row = weights_graph(w_row, p)
    A_col = weights_graph(w_col, n)
    nRowComp = len(find_clusters(A_row)["size"])
    nColComp = len(find_clusters(A_col)["size"])
    return {"w_row": w_row.data, "w_col": w_col.data, "E_row": E_row, "E_col": E_col,
            "nRowComp": nRowComp, "nColComp": nColComp}


def knn_weights(w, k, n):
    keep = set()
    i = 1
    neighbors = tri2vec(i, np.arange(i + 1, n + 1), n) - 1
    sorted_idx = np.argsort(-w[neighbors])
    knn = neighbors[sorted_idx[:k]]
    keep.update(knn.tolist())
    for i in range(2, n):
        group_A = tri2vec(i, np.arange(i + 1, n + 1), n) - 1
        group_B = tri2vec(np.arange(1, i), i, n) - 1
        neighbors = np.concatenate((group_A, group_B))
        sorted_idx = np.argsort(-w[neighbors])
        knn = neighbors[sorted_idx[:k]]
        keep.update(knn.tolist())
    i = n
    neighbors = tri2vec(np.arange(1, i), i, n) - 1
    sorted_idx = np.argsort(-w[neighbors])
    knn = neighbors[sorted_idx[:k]]
    keep.update(knn.tolist())
    new_w = np.copy(w)
    mask = np.ones_like(new_w, dtype=bool)
    indices = np.arange(new_w.size)
    mask[np.isin(indices, list(keep))] = False
    new_w[mask] = 0
    return csc_matrix(new_w.reshape(-1, 1))


def create_edge_incidence(w, n):
    P = vec2tri(w.indices + 1, n)
    nEdges = P.shape[0]
    row = np.concatenate((np.arange(nEdges), np.arange(nEdges)))
    col = np.concatenate((P[:, 0] - 1, P[:, 1] - 1))
    data = np.concatenate((np.ones(nEdges), -np.ones(nEdges)))
    E = coo_matrix((data, (row, col)), shape=(nEdges, n)).tocsc()
    return E


def create_adjacency(V, Phi):
    differences = np.linalg.norm(V, axis=0)
    connected_ix = np.where(differences == 0)[0]
    n = Phi.shape[1]
    if len(connected_ix) > 0:
        ix = []
        jx = []
        Phi_dense = Phi.toarray()
        for r in connected_ix:
            i_val = np.where(Phi_dense[r, :] == 1)[0]
            j_val = np.where(Phi_dense[r, :] == -1)[0]
            if i_val.size > 0 and j_val.size > 0:
                ix.append(i_val[0])
                jx.append(j_val[0])
        if len(ix) > 0:
            data = np.ones(len(ix))
            A = coo_matrix((data, (ix, jx)), shape=(n, n)).tocsc()
        else:
            A = csc_matrix((n, n))
    else:
        A = csc_matrix((n, n))
    return A


def find_clusters(A):
    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    clusters = list(nx.connected_components(G))
    n = A.shape[0]
    cluster = np.zeros(n, dtype=int)
    for idx, comp in enumerate(clusters):
        for node in comp:
            cluster[node] = idx + 1
    sizes = np.array([len(comp) for comp in clusters], dtype=int)
    return {"cluster": cluster, "size": sizes}


def weights_graph(w, n):
    P = vec2tri(w.indices + 1, n)
    nEdges = P.shape[0]
    row = P[:, 0] - 1
    col = P[:, 1] - 1
    data = np.ones(nEdges)
    A = coo_matrix((data, (row, col)), shape=(n, n)).tocsc()
    return A


def update_majorization(MT, UT, Theta, nMissing):
    MT = np.asfortranarray(MT)
    UT = np.asfortranarray(UT)
    for idx in Theta[:nMissing]:
        MT.ravel(order='F')[idx] = UT.ravel(order='F')[idx]
    return MT


def get_subgroup_means_full(X, clusters_row, clusters_col):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    num_clusters_row = len(clusters_row["size"])
    num_clusters_col = len(clusters_col["size"])
    M = np.full((num_clusters_row, num_clusters_col), np.nan)
    for i in range(num_clusters_row):
        ix_row = np.where(np.array(clusters_row["cluster"]) == (i + 1))[0]
        for j in range(num_clusters_col):
            ix_col = np.where(np.array(clusters_col["cluster"]) == (j + 1))[0]
            if ix_row.size == 0 or ix_col.size == 0:
                M[i, j] = np.nan
            else:
                submat = X[np.ix_(ix_row, ix_col)]
                if submat.size == 0:
                    M[i, j] = np.nan
                else:
                    M[i, j] = np.nanmean(submat)
    return M


def get_subgroup_means(X, Theta, clusters_row, clusters_col):
    Y = X.copy()
    Y.to_numpy().ravel(order='F')[Theta] = np.nan
    return get_subgroup_means_full(Y, clusters_row, clusters_col)


def get_validation(p, n, fraction=0.1, seed=123):
    random.seed(seed)
    total = n * p
    num = int(round(fraction * total))
    ix = random.sample(range(total), num)
    ix1 = np.array(ix) + 1
    rows = np.ceil(ix1 / n).astype(int)
    cols = ix1 - n*(rows - 1)
    ThetaM = np.column_stack((rows, cols))
    ThetaV = ix1
    return {"ThetaM": ThetaM, "ThetaV": ThetaV}


def convex_cluster_fasta(XT, UT, VT, LambdaT, LambdaT_temp, LambdaT_old,
                         dLambdaT, gLambdaT, gLambdaT_old, Phi, w, nu, max_iter, tol):
    m = LambdaT.shape[0]
    p = LambdaT.shape[1]
    M = 10
    UT[:] = update_UT(XT, LambdaT, Phi)
    dual = convex_cluster_dual(XT, UT)
    dual_hist = [dual]
    gLambdaT[:] = grad_LambdaT(UT, Phi)
    primal_hist = []
    for its in range(1, max_iter):
        LambdaT_old[:] = LambdaT.copy()
        LambdaT[:] = update_LambdaT2(LambdaT, gLambdaT, nu, w)
        UT[:] = update_UT(XT, LambdaT, Phi)
        dual_temp = convex_cluster_dual(XT, UT)
        dual_local_max = -dual_hist[-1]
        idx_min = max(its - M, 0)
        for j in range(its - 1, idx_min - 1, -1):
            dual_local_max = max(dual_local_max, -dual_hist[j])
        lhs = -dual_temp - 1e-12
        rhs = dual_local_max
        dLambdaT.fill(0)
        for i in range(m):
            for j in range(p):
                delta = LambdaT[i, j] - LambdaT_old[i, j]
                rhs += delta * (gLambdaT[i, j] + 0.5 * delta / nu)
                dLambdaT[i, j] = delta
        backtrack_count = 0
        while lhs > rhs and backtrack_count < 20:
            nu = 0.5 * nu
            backtrack_count += 1
            LambdaT[:] = LambdaT_old.copy()
            LambdaT[:] = update_LambdaT2(LambdaT, gLambdaT, nu, w)
            UT[:] = update_UT(XT, LambdaT, Phi)
            dual_temp = convex_cluster_dual(XT, UT)
            lhs = -dual_temp - 1e-12
            rhs = dual_local_max
            for i in range(m):
                for j in range(p):
                    delta = LambdaT[i, j] - LambdaT_old[i, j]
                    rhs += delta * (gLambdaT[i, j] + 0.5 * delta / nu)
                    dLambdaT[i, j] = delta
        dual_hist.append(dual_temp)
        fp = convex_cluster_primal(XT, UT, VT, Phi, w)
        primal_hist.append(fp)
        gLambdaT_old[:] = gLambdaT.copy()
        gLambdaT[:] = grad_LambdaT(UT, Phi)
        dLambdaSq = np.sum(dLambdaT**2)
        dLambdaDotdGrad = np.sum(dLambdaT * (gLambdaT - gLambdaT_old))
        dGradSq = np.sum((gLambdaT - gLambdaT_old)**2)
        nu_s = dLambdaSq / dLambdaDotdGrad if dLambdaDotdGrad != 0 else nu
        nu_m = dLambdaDotdGrad / dGradSq if dGradSq != 0 else nu
        if 2.0 * nu_m > nu_s:
            nu = nu_m
        else:
            nu = nu_s - 0.5 * nu_m
        if fp - dual_temp < tol:
            break
    return nu, its, primal_hist, dual_hist


def convex_bicluster_dlpa(XT, LambdaT_row, LambdaT_temp_row, LambdaT_old_row, dLambdaT_row,
                          gLambdaT_row, gLambdaT_old_row, LambdaT_col, LambdaT_temp_col,
                          LambdaT_old_col, dLambdaT_col, gLambdaT_col, gLambdaT_old_col,
                          VT_row, VT_col, UT, YT, PT, QT, Phi_row, Phi_col,
                          w_row, w_col, nu_row, nu_col, max_iter, tol):
    n = Phi_col["Ncol"]
    p = Phi_row["Ncol"]
    UP = np.zeros((p, n))
    YQ = np.zeros((n, p))
    PT[:] = 0
    QT[:] = 0
    max_iter_row = 1000
    max_iter_col = 1000
    primal_row_hist = []
    dual_row_hist = []
    primal_col_hist = []
    dual_col_hist = []
    UT[:] = XT.copy()
    for its in range(max_iter):
        UP = (UT.T + PT.T)
        nu_row, iter_row, primal_row_local, dual_row_local = convex_cluster_fasta(
            UP, YT, VT_row, LambdaT_row, LambdaT_temp_row, LambdaT_old_row,
            dLambdaT_row, gLambdaT_row, gLambdaT_old_row, Phi_row, w_row, nu_row,
            max_iter_row, tol)
        VT_row[:] = update_VT_row(YT, LambdaT_row, Phi_row, w_row, nu_row)
        for i in range(n):
            for j in range(p):
                PT[i, j] += UT[i, j] - YT[j, i]
                YQ[i, j] = YT[j, i] + QT[j, i]
        nu_col, iter_col, primal_col_local, dual_col_local = convex_cluster_fasta(
            YQ, UT, VT_col, LambdaT_col, LambdaT_temp_col, LambdaT_old_col,
            dLambdaT_col, gLambdaT_col, gLambdaT_old_col,
            Phi_col, w_col, nu_col, max_iter_col, tol)
        VT_col[:] = update_VT_col(UT, LambdaT_col, Phi_col, w_col, nu_col)
        for i in range(p):
            for j in range(n):
                QT[i, j] += YT[i, j] - UT[j, i]
        diff = sqrt(np.sum((UT - YT.T)**2))
        if iter_row < len(primal_row_local):
            primal_row_hist.append(primal_row_local[iter_row])
            dual_row_hist.append(dual_row_local[iter_row])
        if iter_col < len(primal_col_local):
            primal_col_hist.append(primal_col_local[iter_col])
            dual_col_hist.append(dual_col_local[iter_col])
        if diff < tol * n * p:
            break
    its = its + 1
    return {"UT": UT,
            "YT": YT,
            "LambdaT_row": LambdaT_row,
            "VT_row": VT_row,
            "LambdaT_col": LambdaT_col,
            "VT_col": VT_col,
            "nu_row": nu_row,
            "nu_col": nu_col,
            "primal_row": primal_row_hist,
            "dual_row": dual_row_hist,
            "primal_col": primal_col_hist,
            "dual_col": dual_col_hist,
            "iter": its}


def convex_bicluster_impute(MT, UT, LambdaT_row, LambdaT_col, VT_row, VT_col,
                            Phi_row, Phi_col, Theta, nMissing, w_row, w_col,
                            nu_row, nu_col, max_iter, tol, max_iter_inner, tol_inner):
    m_row = Phi_row["Nrow"]
    m_col = Phi_col["Nrow"]
    n = Phi_col["Ncol"]
    p = Phi_row["Ncol"]
    YT    = np.zeros((p, n))
    PT    = np.zeros((n, p))
    QT    = np.zeros((p, n))
    VT_temp_row = np.zeros((m_row, n))
    VT_temp_col = np.zeros((m_col, p))
    LambdaT_temp_row = np.zeros((m_row, n))
    LambdaT_old_row  = np.zeros((m_row, n))
    dLambdaT_row     = np.zeros((m_row, n))
    gLambdaT_row     = np.zeros((m_row, n))
    gLambdaT_old_row = np.zeros((m_row, n))
    LambdaT_temp_col = np.zeros((m_col, p))
    LambdaT_old_col  = np.zeros((m_col, p))
    dLambdaT_col     = np.zeros((m_col, p))
    gLambdaT_col     = np.zeros((m_col, p))
    gLambdaT_old_col = np.zeros((m_col, p))
    MT = update_majorization(MT, UT, Theta, nMissing)
    mm_loss_last = convex_bicluster_primal(MT, UT, VT_temp_row, VT_temp_col, Phi_row, Phi_col, w_row, w_col)
    mm_loss_history = [mm_loss_last]
    for its in range(1, max_iter):
        res = convex_bicluster_dlpa(
            MT,
            LambdaT_row, LambdaT_temp_row, LambdaT_old_row, dLambdaT_row,
            gLambdaT_row, gLambdaT_old_row,
            LambdaT_col, LambdaT_temp_col, LambdaT_old_col, dLambdaT_col,
            gLambdaT_col, gLambdaT_old_col,
            VT_row, VT_col,
            UT, YT, PT, QT,
            Phi_row, Phi_col,
            w_row, w_col,
            nu_row, nu_col,
            max_iter_inner, tol_inner
        )
        MT= update_majorization(MT, UT, Theta, nMissing)
        mm_loss_temp = convex_bicluster_primal(MT, UT, VT_temp_row, VT_temp_col, Phi_row, Phi_col, w_row, w_col)
        mm_loss_history.append(mm_loss_temp)
        if mm_loss_last >= mm_loss_temp and (mm_loss_last - mm_loss_temp) < tol * (1.0 + mm_loss_last):
            break
        mm_loss_last = mm_loss_temp
    iter_count = its + 1
    return (iter_count, 
            mm_loss_history, 
            UT, 
            res['LambdaT_row'], 
            res['VT_row'], 
            res['LambdaT_col'], 
            res['VT_col'], 
            res['nu_row'], 
            res['nu_col'])


def test_convex_bicluster_impute(mt, ut, lambdat_row, lambdat_col, vt_row, vt_col,
                                 column_ptr_row, row_indices_row, values_row,
                                 column_ptr_col, row_indices_col, values_col,
                                 m_row, m_col, n, p, Theta, nMissing,
                                 w_row, w_col, nu_row, nu_col, max_iter, tol,
                                 max_iter_inner, tol_inner):
    MT = np.array(mt, dtype=float).reshape((n, p))
    UT = np.array(ut, dtype=float).reshape((n, p))
    LambdaT_row = np.array(lambdat_row, dtype=float).reshape((m_row, n))
    LambdaT_col = np.array(lambdat_col, dtype=float).reshape((m_col, p))
    VT_row = np.array(vt_row, dtype=float).reshape((m_row, n))
    VT_col = np.array(vt_col, dtype=float).reshape((m_col, p))
    Phi_row = {
        "Nrow": m_row,
        "Ncol": p,
        "column_ptr": np.array(column_ptr_row, dtype=np.int32),
        "row_indices": np.array(row_indices_row, dtype=np.int32),
        "values": np.array(values_row, dtype=float)
    }
    Phi_col = {
        "Nrow": m_col,
        "Ncol": n,
        "column_ptr": np.array(column_ptr_col, dtype=np.int32),
        "row_indices": np.array(row_indices_col, dtype=np.int32),
        "values": np.array(values_col, dtype=float)
    }
    w_row = np.array(w_row, dtype=float)
    w_col = np.array(w_col, dtype=float)
    nu_row = float(nu_row)
    nu_col = float(nu_col)
    max_iter = int(max_iter)
    tol = float(tol)
    max_iter_inner = int(max_iter_inner)
    tol_inner = float(tol_inner)
    return convex_bicluster_impute(MT, UT, LambdaT_row, LambdaT_col, VT_row, VT_col,
                                   Phi_row, Phi_col, Theta, nMissing, w_row, w_col,
                                   nu_row, nu_col, max_iter, tol, max_iter_inner, tol_inner)


def cobra_pod(X, Lambda_row, Lambda_col, E_row, E_col, w_row, w_col, Theta,
              max_iter=100, tol=1e-3, max_iter_inner=1000, tol_inner=1e-4):
    m_row = E_row.shape[0]
    m_col = E_col.shape[0]
    n = X.shape[1]
    p = X.shape[0]
    XT = np.ascontiguousarray(X.T, dtype=np.float64)
    UT = np.ascontiguousarray(X.T, dtype=np.float64)
    LambdaT_row = np.ascontiguousarray(np.array(Lambda_row).T, dtype=np.float64)
    LambdaT_col = np.ascontiguousarray(np.array(Lambda_col).T, dtype=np.float64)
    VT_row = np.zeros((m_row, n), dtype=np.float64)
    VT_col = np.zeros((m_col, p), dtype=np.float64)
    Theta = np.array(Theta, dtype=int) - 1
    nMissing = len(Theta)
    column_ptr_row = E_row.indptr.astype(np.int32)
    values_row = E_row.data.astype(np.float64)
    row_indices_row = E_row.indices.astype(np.int32)
    column_ptr_col = E_col.indptr.astype(np.int32)
    values_col = E_col.data.astype(np.float64)
    row_indices_col = E_col.indices.astype(np.int32)
    w_row = np.array(w_row, dtype=np.float64)
    w_col = np.array(w_col, dtype=np.float64)
    nu_row = 1.0 / X.shape[0]
    nu_col = 1.0 / X.shape[1]
    max_iter = int(max_iter)
    tol = float(tol)
    max_iter_inner = int(max_iter_inner)
    tol_inner = float(tol_inner)
    sol = test_convex_bicluster_impute(
        mt=XT, ut=UT,
        lambdat_row=LambdaT_row.ravel(),
        lambdat_col=LambdaT_col.ravel(),
        vt_row=VT_row.ravel(), vt_col=VT_col.ravel(),
        column_ptr_row=column_ptr_row, row_indices_row=row_indices_row, values_row=values_row,
        column_ptr_col=column_ptr_col, row_indices_col=row_indices_col, values_col=values_col,
        m_row=m_row, m_col=m_col, n=n, p=p,
        Theta=Theta, nMissing=nMissing,
        w_row=w_row, w_col=w_col,
        nu_row=nu_row, nu_col=nu_col,
        max_iter=max_iter, tol=tol,
        max_iter_inner=max_iter_inner, tol_inner=tol_inner
    )  
    iter_count, mm_loss_history, UT, LambdaT_row, VT_row, LambdaT_col, VT_col, nu_row, nu_col = sol
    return {
        "U": UT.T,
        "Lambda_row": LambdaT_row.T,
        "V_row": VT_row.T,
        "Lambda_col": LambdaT_col.T,
        "V_col": VT_col.T,
        "nu_row": nu_row,
        "nu_col": nu_col,
        "mm_loss": mm_loss_history,
        "iter": iter_count
    }


def cobra_validate(X, E_row, E_col, w_row, w_col, gamma_seq, Lambda_row=None, Lambda_col=None,
                   fraction=0.1, max_iter=100, tol=1e-3, max_iter_inner=1000, tol_inner=1e-4):
    n_samples = X.shape[1]
    p_features = X.shape[0]
    ThetaOut = get_validation(p_features, n_samples, fraction)
    ThetaM = ThetaOut["ThetaM"]
    ThetaV = ThetaOut["ThetaV"]
    nGamma = len(gamma_seq)
    UHx = [None] * nGamma
    VrHx = [None] * nGamma
    VcHx = [None] * nGamma
    validation_error = np.zeros(nGamma)
    groups_row = [None] * nGamma
    groups_col = [None] * nGamma
    XT = X.T
    if Lambda_row is None:
        Lambda_row = np.zeros((n_samples, E_row.shape[0]))
    if Lambda_col is None:
        Lambda_col = np.zeros((p_features, E_col.shape[0]))
    for ig, gam in enumerate(gamma_seq):
        sol = cobra_pod(X, Lambda_row, Lambda_col, E_row, E_col,
                        (gam * np.array(w_row)), (gam * np.array(w_col)), ThetaV,
                        max_iter=max_iter, tol=tol, max_iter_inner=max_iter_inner,
                        tol_inner=tol_inner)
        UHx[ig] = sol["U"]
        VrHx[ig] = sol["V_row"]
        VcHx[ig] = sol["V_col"]
        clusters_row = find_clusters(create_adjacency(sol["V_row"], E_row))
        clusters_col = find_clusters(create_adjacency(sol["V_col"], E_col))
        groups_row[ig] = clusters_row
        groups_col[ig] = clusters_col
        linear_idx = (ThetaM[:, 0] - 1) + p_features * (ThetaM[:, 1] - 1)
        MM = get_subgroup_means(X, linear_idx, clusters_row, clusters_col)
        MM = np.nan_to_num(MM)
        ixi = clusters_row['cluster'][ThetaM[:, 0] - 1]
        ixj = clusters_col['cluster'][ThetaM[:, 1] - 1]
        errors = np.zeros(ThetaM.shape[0])
        for i in range(ThetaM.shape[0]):
            errors[i] = MM[ixi[i]-1, ixj[i]-1] - XT.to_numpy().ravel(order='F')[ThetaV[i]-1]
        validation_error[ig] = math.sqrt(np.sum(errors**2))
        print(f"***** Completed gamma = {ig+1} *****")
    return {"U": UHx, "V_row": VrHx, "V_col": VcHx,
            "ThetaM": ThetaM, "ThetaV": ThetaV,
            "groups_row": groups_row, "groups_col": groups_col,
            "validation_error": validation_error}


def biclust_smooth(X, clusters_row, clusters_col):
    p, n = X.shape
    Y = np.full((p, n), np.nan)
    M = get_subgroup_means_full(X, clusters_row, clusters_col)
    num_clusters_row = len(clusters_row['size'])
    num_clusters_col = len(clusters_col['size'])
    for i in range(num_clusters_row):
        ix_row = np.where(clusters_row['cluster'] == (i + 1))[0]
        for j in range(num_clusters_col):
            ix_col = np.where(clusters_col['cluster'] == (j + 1))[0]
            Y[np.ix_(ix_row, ix_col)] = M[i, j]
    return Y


