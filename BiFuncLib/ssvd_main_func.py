import numpy as np
import pandas as pd
import math
from copy import deepcopy
from scipy.sparse.linalg import svds
import gc

from BiFuncLib.BiclustResult import BiclustResult


def thresh(z, delta, thredtype = 1, a = 3.7):
    z = np.asarray(z)
    if thredtype == 1:
        return np.sign(z) * ((np.abs(z) >= delta).astype(float)) * (np.abs(z) - delta)
    elif thredtype == 2:
        return z * ((np.abs(z) > delta).astype(float))
    elif thredtype == 3:
        term1 = np.sign(z) * ((np.abs(z) >= delta).astype(float)) * (np.abs(z) - delta) * ((np.abs(z) <= 2 * delta).astype(float))
        term2 = (((a - 1) * z - np.sign(z) * a * delta) / (a - 2)) * (((2 * delta < np.abs(z)).astype(float)) * ((np.abs(z) <= a * delta).astype(float)))
        term3 = z * ((np.abs(z) > a * delta).astype(float))
        return term1 + term2 + term3


def initial_svd(X, nu = 1, nv = 1):
    n, d = X.shape
    if min(n, d) < 500:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return U[:, :nu], Vt[:nv, :].T
    else:
        u, s, vt = svds(X, k=1, which='LM')
        return u, vt.T


def jaccardmat(res1, res2, mode=None):
    def jaccard_sets(A, B):
        intersection = A & B
        denom = len(A) + len(B) - len(intersection)
        return len(intersection) / denom if denom != 0 else 0
    if mode == 'row':
        if res1.Number > 0 and res2.Number > 0:
            mat = np.zeros((res1.Number, res2.Number))
            for i in range(res1.Number):
                A = set(np.nonzero(res1.RowxNumber[:, i])[0])
                for j in range(res2.Number):
                    B = set(np.nonzero(res2.RowxNumber[:, j])[0])
                    mat[i, j] = jaccard_sets(A, B)
            rownames = [f"RowC{i+1}" for i in range(res1.Number)]
            colnames = [f"RowC{j+1}" for j in range(res2.Number)]
            return pd.DataFrame(mat, index=rownames, columns=colnames)
        else:
            return pd.DataFrame([[0]])
    elif mode == 'column':
        nclus1 = res1.NumberxCol.shape[0]
        nclus2 = res2.NumberxCol.shape[0]
        if nclus1 > 0 and nclus2 > 0:
            mat = np.zeros((nclus1, nclus2))
            for i in range(nclus1):
                A = set(np.nonzero(res1.NumberxCol[i, :])[0])
                for j in range(nclus2):
                    B = set(np.nonzero(res2.NumberxCol[j, :])[0])
                    mat[i, j] = jaccard_sets(A, B)
            rownames = [f"ColC{i+1}" for i in range(nclus1)]
            colnames = [f"ColC{j+1}" for j in range(nclus2)]
            return pd.DataFrame(mat, index=rownames, columns=colnames)
        else:
            return pd.DataFrame([[0]])
    else:
        if res1.Number > 0 and res2.Number > 0:
            mat = np.zeros((res1.Number, res2.Number))
            for i in range(res1.Number):
                for j in range(res2.Number):
                    product1 = np.outer(res1.RowxNumber[:, i], res1.NumberxCol[i, :])
                    product2 = np.outer(res2.RowxNumber[:, j], res2.NumberxCol[j, :])
                    flat1 = np.ravel(product1 > 0, order='F')
                    flat2 = np.ravel(product2 > 0, order='F')
                    A = set(np.nonzero(flat1)[0])
                    B = set(np.nonzero(flat2)[0])
                    mat[i, j] = jaccard_sets(A, B)
            rownames = [f"BC{i+1}" for i in range(res1.Number)]
            colnames = [f"BC{j+1}" for j in range(res2.Number)]
            return pd.DataFrame(mat, index=rownames, columns=colnames)
        else:
            return pd.DataFrame([[0]])


def ssvd(X, threu=1, threv=1, gamu=0, gamv=0, u0=None, v0=None, merr=1e-4, niter=100):
    X = deepcopy(X)
    n, d = X.shape
    if u0 is None or v0 is None:
        u0, v0 = initial_svd(X, nu=1, nv=1)
        u0 = np.squeeze(u0)
        v0 = np.squeeze(v0)
    stop = False
    ud = 1.0
    vd = 1.0
    iteration = 0
    SST = np.sum(X**2)
    while ud > merr or vd > merr:
        iteration += 1
        print("iter:", iteration)
        print("v:", np.count_nonzero(v0))
        print("u:", np.count_nonzero(u0))
        z = np.dot(X.T, u0)
        winv = np.abs(z) ** gamv
        sigsq = np.abs(SST - np.sum(z**2)) / (n * d - d)
        tv = np.sort(np.concatenate(([0], np.abs(z * winv))))
        rv = np.sum(tv > 0)
        Bv = np.full(d + 1, np.inf)
        ind_v = np.where(winv != 0)[0]
        for i in range(1, int(rv) + 1):
            lvc = tv[-i]
            delta = lvc / winv[ind_v]
            z_subset = z[ind_v]
            temp2 = np.sign(z_subset) * ((np.abs(z_subset) >= delta).astype(float)) * (np.abs(z_subset) - delta)
            vc = np.zeros_like(z)
            vc[ind_v] = temp2
            Bv[i] = np.sum((z - vc) ** 2) / sigsq + i * math.log(n * d)
        candidate_Bv = Bv[1:int(rv) + 1]
        Iv = np.argmin(candidate_Bv) + 1
        lv = tv[-Iv]
        v_temp = thresh(z[ind_v], thredtype=threv, delta=lv / winv[ind_v])
        v1 = np.zeros_like(z)
        v1[ind_v] = v_temp
        norm_v1 = np.linalg.norm(v1)
        if norm_v1 != 0:
            v1 = v1 / norm_v1
        print("v1:", np.count_nonzero(v1))
        z = np.dot(X, v1)
        winu = np.abs(z) ** gamu
        sigsq = np.abs(SST - np.sum(z**2)) / (n * d - n)
        tu = np.sort(np.concatenate(([0], np.abs(z * winu))))
        ru = np.sum(tu > 0)
        Bu = np.full(n + 1, np.inf)
        ind_u = np.where(winu != 0)[0]
        for i in range(1, int(ru) + 1):
            luc = tu[-i]
            delta = luc / winu[ind_u]
            z_subset = z[ind_u]
            temp2 = np.sign(z_subset) * ((np.abs(z_subset) >= delta).astype(float)) * (np.abs(z_subset) - delta)
            uc = np.zeros_like(z)
            uc[ind_u] = temp2
            Bu[i] = np.sum((z - uc) ** 2) / sigsq + i * math.log(n * d)
        candidate_Bu = Bu[1:int(ru) + 1]
        Iu = np.argmin(candidate_Bu) + 1
        lu = tu[-Iu]
        u_temp = thresh(z[ind_u], delta=lu / winu[ind_u])
        u1 = np.zeros_like(z)
        u1[ind_u] = u_temp
        norm_u1 = np.linalg.norm(u1)
        if norm_u1 != 0:
            u1 = u1 / norm_u1
        ud = np.linalg.norm(u0 - u1)
        vd = np.linalg.norm(v0 - v1)
        if iteration > niter:
            print("Fail to converge! Increase the niter!")
            stop = True
            break
        u0 = u1
        v0 = v1
    return {"u": u1, "v": v1, "iter": iteration, "stop": stop}


def ssvd_bc(X, K=10, threu=1, threv=1, gamu=0, gamv=0, merr=1e-4, niter=100):
    X = deepcopy(X)
    res = []
    n, d = X.shape
    RowxNumber = np.zeros((n, K), dtype=bool)
    NumberxCol = np.zeros((K, d), dtype=bool)
    current_X = deepcopy(X)
    actual_K = K
    for k in range(K):
        current_res = ssvd(current_X, threu=threu, threv=threv,
                           gamu=gamu, gamv=gamv, merr=merr, niter=niter)
        res.append(current_res)
        if current_res.get("stop", False):
            actual_K = k
            break
        RowxNumber[:, k] = (current_res["u"] != 0)
        NumberxCol[k, :] = (current_res["v"] != 0)
        d_val = float(np.dot(current_res["u"].T, np.dot(current_X, current_res["v"])))
        current_res["d"] = d_val
        current_X = current_X - d_val * np.outer(current_res["u"], current_res["v"])
    RowxNumber = RowxNumber[:, :actual_K]
    NumberxCol = NumberxCol[:actual_K, :]
    params = {"K": actual_K, "threu": threu, "threv": threv,
              "gamu": gamu, "gamv": gamv, "merr": merr, "niter": niter}
    Number = actual_K
    info = {"res": res}
    return BiclustResult(params, RowxNumber, NumberxCol.T, Number, info)


def adalasso_nc(X, b, lam, steps, size, gamm=0):
    n = len(b)
    m = int(n * size)
    subsets = np.column_stack([np.random.choice(n, size=m, replace=False) for _ in range(steps)])
    results = np.array([adalassosteps_nc(i, subsets, X, b, lam, gamm) for i in range(steps)]).T
    return results


def adalassosteps_nc(index, subsets, X, b, lam, gamm):
    subset = subsets[:, index]
    ols = X[:, subset] @ b[subset]
    delta = lam / (np.abs(ols) ** gamm)
    ols_thresholded = np.sign(ols) * np.where(np.abs(ols) >= delta, np.abs(ols) - delta, 0)
    ols_thresholded = np.nan_to_num(ols_thresholded)
    return ols_thresholded


def adalasso(X, b, lam, steps, size, gamm=0):
    n = len(b)
    m = int(n * size)
    subsets = np.column_stack([np.random.choice(n, size=m, replace=False) for _ in range(steps)])
    results = np.array([adalassosteps(i, subsets, X, b, lam, gamm) for i in range(steps)]).T
    return results


def adalassosteps(index, subsets, X, b, lam, gamm):
    subset = subsets[:, index]
    ols = X[:, subset] @ b[subset]
    delta = lam / (np.abs(ols) ** gamm)
    ols_thresholded = np.sign(ols) * np.where(np.abs(ols) >= delta, np.abs(ols) - delta, 0)
    ols_thresholded = np.nan_to_num(ols_thresholded)
    mostof = np.sign(np.sum(np.sign(ols_thresholded)))
    if mostof == 0:
        mostof = 1
    ols_thresholded = np.where(np.sign(ols_thresholded) == mostof, ols_thresholded, 0)
    ols_thresholded = np.nan_to_num(ols_thresholded)
    return ols_thresholded


def update_v(X, u0, pcer, n_ini, ss_thr, steps, size, gamm, cols_nc=False, savepath=False):
    n_ini = X.shape[1]
    n = n_ini
    err = pcer * n_ini
    ols = np.dot(X.T, u0)
    stop_flag = False
    lambdas = np.sort(np.concatenate((np.abs(ols), np.array([0]))))[::-1]
    if savepath:
        selprobpath = np.zeros((len(ols), len(lambdas)))
    qs = np.zeros(len(lambdas))
    thrall = np.zeros(len(lambdas))
    ls_index = len(lambdas) - 1
    if cols_nc:
        for l in range(len(lambdas)):
            temp = adalasso_nc(X.T, u0, lambdas[l], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            if savepath:
                selprobpath[:, l] = sp
            thrall[l] = ((qs[l]**2 / (err * n_ini)) + 1) / 2
            if thrall[l] >= ss_thr[0]:
                ls_index = l
                break
    else:
        for l in range(len(lambdas)):
            temp = adalasso(X.T, u0, lambdas[l], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            if savepath:
                selprobpath[:, l] = sp
            thrall[l] = ((qs[l]**2 / (err * n_ini)) + 1) / 2
            if thrall[l] >= ss_thr[0]:
                ls_index = l
                break
    thr = thrall[ls_index]
    if thr > ss_thr[1]:
        while pcer <= 0.5:
            pcer += 0.01
            thrall = ((qs**2 / ((pcer * n_ini) * n_ini)) + 1) / 2
            thr = thrall[ls_index]
            if thr < ss_thr[1]:
                break
    stable = np.where(sp >= thr)[0]
    if stable.size == 0:
        stop_flag = True
    vc = np.zeros(n)
    delta = lambdas[ls_index] / (np.abs(ols) ** gamm)
    vc = np.sign(ols) * ((np.abs(ols) >= delta).astype(float)) * (np.abs(ols) - delta)
    if savepath:
        return {"vc": vc, "sp": sp, "stop": stop_flag, "qs": qs,
                "thr": thr, "l": ls_index, "delta": delta, "selprobpath": selprobpath}
    else:
        return {"vc": vc, "sp": sp, "stop": stop_flag, "qs": qs,
                "thr": thr, "l": ls_index, "delta": delta}


def update_u(X, v0, pcer, p_ini, ss_thr, steps, size, gamm, rows_nc=False, savepath=False, start=False):
    p_ini = X.shape[0]
    p = p_ini
    err = pcer * p_ini
    ols = np.dot(X, v0)
    stop_flag = False
    lambdas = np.sort(np.concatenate((np.abs(ols), np.array([0]))))[::-1]
    if savepath:
        selprobpath = np.zeros((len(ols), len(lambdas)))
    qs = np.zeros(len(lambdas))
    thrall = np.zeros(len(lambdas))
    ls_index = len(lambdas) - 1
    if rows_nc:
        for l in range(len(lambdas)):
            temp = adalasso_nc(X, v0, lambdas[l], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            if savepath:
                selprobpath[:, l] = sp
            thrall[l] = ((qs[l]**2 / (err * p_ini)) + 1) / 2
            if thrall[l] >= ss_thr[0]:
                ls_index = l
                break
    else:
        for l in range(len(lambdas)):
            temp = adalasso(X, v0, lambdas[l], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            if savepath:
                selprobpath[:, l] = sp
            thrall[l] = ((qs[l]**2 / (err * p_ini)) + 1) / 2
            if thrall[l] >= ss_thr[0]:
                ls_index = l
                break
    thr = thrall[ls_index]
    if thr > ss_thr[1]:
        while pcer <= 0.5:
            pcer += 0.01
            thrall = ((qs**2 / ((pcer * p_ini) * p_ini)) + 1) / 2
            thr = thrall[ls_index]
            if thr < ss_thr[1]:
                break
    stable = np.where(sp >= thr)[0]
    if stable.size == 0:
        stop_flag = True
    uc = np.zeros(p)
    delta = lambdas[ls_index] / (np.abs(ols) ** gamm)
    uc = np.sign(ols) * ((np.abs(ols) >= delta).astype(float)) * (np.abs(ols) - delta)
    if savepath:
        return {"uc": uc, "sp": sp, "stop": stop_flag, "qs": qs,
                "thr": thr, "l": ls_index, "delta": delta, "selprobpath": selprobpath}
    else:
        return {"uc": uc, "sp": sp, "stop": stop_flag, "qs": qs,
                "thr": thr, "l": ls_index, "delta": delta}


def update_v_pw(X, u0, pcer, n_ini, ss_thr, steps, size, gamm, cols_nc=False, l_val=None, start=False):
    n_ini = X.shape[1]
    n = n_ini
    err = pcer * n_ini
    ols = np.dot(X.T, u0)
    stop_flag = False
    lambdas = np.sort(np.concatenate((np.abs(ols), np.array([0]))))[::-1]
    qs = np.zeros(len(lambdas))
    thrall = np.zeros(len(lambdas))
    ls_index = len(lambdas) - 1
    if l_val is None:
        median_val = np.quantile(lambdas, 0.5, interpolation='linear')
        indices = np.where(np.isclose(lambdas, median_val))[0]
        if len(indices) > 0:
            l_val = indices[0]
        else:
            l_val = 0
    l_idx = l_val
    l_min = 0
    l_max = len(lambdas) - 1
    if cols_nc:
        for g in range(len(lambdas)):
            temp = adalasso_nc(X.T, u0, lambdas[l_idx], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l_idx] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            thrall[l_idx] = ((qs[l_idx]**2 / (err * n_ini)) + 1) / 2
            if ss_thr[0] <= thrall[l_idx] <= ss_thr[1]:
                ls_index = l_idx
                break
            if thrall[l_idx] < ss_thr[0]:
                l_min = l_idx
                if l_idx == len(lambdas) - 1:
                    break
                if thrall[l_idx+1] > ss_thr[1]:
                    ls_index = l_idx + 1
                    if thrall[l_idx+1] > 1:
                        ls_index = l_idx
                    temp = adalasso_nc(X.T, u0, lambdas[ls_index], steps, size, gamm)
                    t_bool = (temp != 0)
                    qs[ls_index] = np.mean(np.sum(t_bool, axis=0))
                    sp = np.mean(t_bool, axis=1)
                    thrall[ls_index] = ((qs[ls_index]**2 / (err * n_ini)) + 1) / 2
                    break
                l_idx = min(len(lambdas) - 1, l_max, l_idx + int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx -= 1
                    if l_idx == 0:
                        break
            if thrall[l_idx] > ss_thr[1]:
                l_max = l_idx
                if l_idx == 0:
                    break
                if l_idx - 1 >= 0 and thrall[l_idx-1] < ss_thr[0] and thrall[l_idx-1] != 0:
                    ls_index = l_idx
                    break
                l_idx = max(0, l_min, l_idx - int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx += 1
                    if l_idx == len(lambdas):
                        break
    else:
        for g in range(len(lambdas)):
            temp = adalasso(X.T, u0, lambdas[l_idx], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l_idx] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            thrall[l_idx] = ((qs[l_idx]**2 / (err * n_ini)) + 1) / 2
            if ss_thr[0] <= thrall[l_idx] <= ss_thr[1]:
                ls_index = l_idx
                break
            if thrall[l_idx] < ss_thr[0]:
                l_min = l_idx
                if l_idx == len(lambdas) - 1:
                    break
                if thrall[l_idx+1] > ss_thr[1]:
                    ls_index = l_idx + 1
                    if thrall[l_idx+1] > 1:
                        ls_index = l_idx
                    temp = adalasso(X.T, u0, lambdas[ls_index], steps, size, gamm)
                    t_bool = (temp != 0)
                    qs[ls_index] = np.mean(np.sum(t_bool, axis=0))
                    sp = np.mean(t_bool, axis=1)
                    thrall[ls_index] = ((qs[ls_index]**2 / (err * n_ini)) + 1) / 2
                    break
                l_idx = min(len(lambdas) - 1, l_max, l_idx + int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx -= 1
                    if l_idx == 0:
                        break
            if thrall[l_idx] > ss_thr[1]:
                l_max = l_idx
                if l_idx == 0:
                    break
                if l_idx - 1 >= 0 and thrall[l_idx-1] < ss_thr[0] and thrall[l_idx-1] != 0:
                    ls_index = l_idx
                    break
                l_idx = max(0, l_min, l_idx - int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx += 1
                    if l_idx == len(lambdas):
                        break
    thr = ((qs[ls_index]**2 / ((pcer * n_ini) * n_ini)) + 1) / 2
    stable = np.where(sp >= thr)[0]
    if stable.size == 0:
        stop_flag = True
    vc = np.zeros(n)
    delta = lambdas[l_idx] / (np.abs(ols) ** gamm)
    vc = np.sign(ols) * ((np.abs(ols) >= delta).astype(float)) * (np.abs(ols) - delta)
    return {"vc": vc, "sp": sp, "stop": stop_flag, "qs": qs, "thr": thr, "l": ls_index, "delta": delta}


def update_u_pw(X, v0, pcer, p_ini, ss_thr, steps, size, gamm, rows_nc=False, l_val=None, start=False):
    p_ini = X.shape[0]
    p = p_ini
    err = pcer * p_ini
    ols = np.dot(X, v0)
    stop_flag = False
    lambdas = np.sort(np.concatenate((np.abs(ols), np.array([0]))))[::-1]
    qs = np.zeros(len(lambdas))
    thrall = np.zeros(len(lambdas))
    ls_index = len(lambdas) - 1
    if l_val is None:
        median_val = np.quantile(lambdas, 0.5, interpolation='linear')
        indices = np.where(np.isclose(lambdas, median_val))[0]
        if len(indices) > 0:
            l_val = indices[0]
        else:
            l_val = 0
    l_idx = l_val
    l_min = 0
    l_max = len(lambdas) - 1
    if rows_nc:
        for g in range(len(lambdas)):
            temp = adalasso_nc(X, v0, lambdas[l_idx], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l_idx] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            thrall[l_idx] = ((qs[l_idx]**2 / (err * p_ini)) + 1) / 2
            if ss_thr[0] <= thrall[l_idx] <= ss_thr[1]:
                ls_index = l_idx
                break
            if thrall[l_idx] < ss_thr[0]:
                l_min = l_idx
                if l_idx == len(lambdas) - 1:
                    break
                if thrall[l_idx+1] > ss_thr[1]:
                    ls_index = l_idx + 1
                    if thrall[l_idx+1] > 1:
                        ls_index = l_idx
                    temp = adalasso_nc(X, v0, lambdas[ls_index], steps, size, gamm)
                    t_bool = (temp != 0)
                    qs[ls_index] = np.mean(np.sum(t_bool, axis=0))
                    sp = np.mean(t_bool, axis=1)
                    thrall[ls_index] = ((qs[ls_index]**2 / (err * p_ini)) + 1) / 2
                    break
                l_idx = min(len(lambdas) - 1, l_max, l_idx + int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx -= 1
                    if l_idx == 0:
                        break
            if thrall[l_idx] > ss_thr[1]:
                l_max = l_idx
                if l_idx == 0:
                    break
                if l_idx - 1 >= 0 and thrall[l_idx-1] < ss_thr[0] and thrall[l_idx-1] != 0:
                    ls_index = l_idx
                    break
                l_idx = max(0, l_min, l_idx - int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx += 1
                    if l_idx == len(lambdas):
                        break
    else:
        for g in range(len(lambdas)):
            temp = adalasso(X, v0, lambdas[l_idx], steps, size, gamm)
            t_bool = (temp != 0)
            qs[l_idx] = np.mean(np.sum(t_bool, axis=0))
            sp = np.mean(t_bool, axis=1)
            thrall[l_idx] = ((qs[l_idx]**2 / (err * p_ini)) + 1) / 2
            if ss_thr[0] <= thrall[l_idx] <= ss_thr[1]:
                ls_index = l_idx
                break
            if thrall[l_idx] < ss_thr[0]:
                l_min = l_idx
                if l_idx == len(lambdas) - 1:
                    break
                if thrall[l_idx+1] > ss_thr[1]:
                    ls_index = l_idx + 1
                    if thrall[l_idx+1] > 1:
                        ls_index = l_idx
                    temp = adalasso(X, v0, lambdas[ls_index], steps, size, gamm)
                    t_bool = (temp != 0)
                    qs[ls_index] = np.mean(np.sum(t_bool, axis=0))
                    sp = np.mean(t_bool, axis=1)
                    thrall[ls_index] = ((qs[ls_index]**2 / (err * p_ini)) + 1) / 2
                    break
                l_idx = min(len(lambdas) - 1, l_max, l_idx + int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx -= 1
                    if l_idx == 0:
                        break
            if thrall[l_idx] > ss_thr[1]:
                l_max = l_idx
                if l_idx == 0:
                    break
                if l_idx - 1 >= 0 and thrall[l_idx-1] < ss_thr[0] and thrall[l_idx-1] != 0:
                    ls_index = l_idx
                    break
                l_idx = max(0, l_min, l_idx - int(math.ceil(len(lambdas) / (g+1))))
                while thrall[l_idx] != 0:
                    l_idx += 1
                    if l_idx == len(lambdas):
                        break
    thr = ((qs[ls_index]**2 / ((pcer * p_ini) * p_ini)) + 1) / 2
    stable = np.where(sp >= thr)[0]
    if stable.size == 0:
        stop_flag = True
    uc = np.zeros(p)
    delta = lambdas[l_idx] / (np.abs(ols) ** gamm)
    uc = np.sign(ols) * ((np.abs(ols) >= delta).astype(float)) * (np.abs(ols) - delta)
    return {"uc": uc, "sp": sp, "stop": stop_flag, "qs": qs, "thr": thr, "l": ls_index, "delta": delta}


def s4vd(X, steps=100, pcerv=0.1, pceru=0.1, ss_thr=(0.6, 0.65), size=0.5, gamm=0,
         iters=100, nbiclust=10, merr=1e-3, cols_nc=True, rows_nc=True,
         row_overlap=True, col_overlap=True, row_min=1, col_min=1, pointwise=True,
         start_iter=3, savepath=False):
    X = deepcopy(X)
    startX = deepcopy(X)
    p_ini = X.shape[0]
    n_ini = X.shape[1]
    rowsin = np.ones(p_ini, dtype=bool)
    colsin = np.ones(n_ini, dtype=bool)
    stop = False
    info = []
    Rows = []
    Cols = []
    vc_dict = {}
    uc_dict = {}
    for k in range(nbiclust):
        gc.collect()
        print("Bicluster", k+1)
        rows = np.zeros(startX.shape[0], dtype=bool)
        cols = np.zeros(startX.shape[1], dtype=bool)
        if X.shape[0] == 0 or X.shape[1] == 0:
            number = k
            stop = True
            break
        U, s, Vt = np.linalg.svd(X, full_matrices=True)
        v0 = Vt[0, :]
        u0 = U[:, 0]
        d0 = s
        vc_dict = {}
        uc_dict = {}
        if (len(u0) * size) <= 2 or (len(v0) * size) <= 2:
            print("submatrix too small for resampling")
            number = k
            stop = True
            break
        if pointwise:
            for i in range(iters):
                uc_dict = update_u_pw(X, v0, pceru, p_ini, ss_thr, steps, size, gamm, rows_nc, l_val=uc_dict.get("l"))
                norm_u = np.linalg.norm(uc_dict["uc"])
                u1 = uc_dict["uc"] / norm_u if norm_u != 0 else uc_dict["uc"]
                u1 = np.nan_to_num(u1)
                vc_dict = update_v_pw(X, u1, pcerv, n_ini, ss_thr, steps, size, gamm, cols_nc, l_val=vc_dict.get("l"))
                norm_v = np.linalg.norm(vc_dict["vc"])
                v1 = vc_dict["vc"] / norm_v if norm_v != 0 else vc_dict["vc"]
                v1 = np.nan_to_num(v1)
                if uc_dict["stop"] and i > start_iter:
                    print("rows not stable")
                    stop = True
                    break
                if vc_dict["stop"] and i > start_iter:
                    print("columns not stable")
                    stop = True
                    break
                ud = np.linalg.norm(u0 - u1)
                vd = np.linalg.norm(v0 - v1)
                print(".", end="", flush=True)
                v0 = v1
                u0 = u1
                if min(ud, vd) < merr and i > start_iter:
                    break
        else:
            for i in range(iters):
                uc_dict = update_u(X, v0, pceru, p_ini, ss_thr, steps, size, gamm, rows_nc, savepath)
                norm_u = np.linalg.norm(uc_dict["uc"])
                u1 = uc_dict["uc"] / norm_u if norm_u != 0 else uc_dict["uc"]
                u1 = np.nan_to_num(u1)
                vc_dict = update_v(X, u1, pcerv, n_ini, ss_thr, steps, size, gamm, cols_nc, savepath)
                norm_v = np.linalg.norm(vc_dict["vc"])
                v1 = vc_dict["vc"] / norm_v if norm_v != 0 else vc_dict["vc"]
                v1 = np.nan_to_num(v1)
                if uc_dict["stop"] and i > start_iter:
                    print("rows not stable")
                    stop = True
                    break
                if vc_dict["stop"] and i > start_iter:
                    print("columns not stable")
                    stop = True
                    break
                ud = np.linalg.norm(u0 - u1)
                vd = np.linalg.norm(v0 - v1)
                print(".", end="", flush=True)
                v0 = v1
                u0 = u1
                if min(ud, vd) < merr and i > start_iter:
                    break
        stableu = (uc_dict["sp"] >= uc_dict["thr"])
        stablev = (vc_dict["sp"] >= vc_dict["thr"])
        d0 = float(np.dot(u0.T, np.dot(X, v0)))
        u0[~stableu] = 0
        v0[~stablev] = 0
        rows[rowsin] = (u0 != 0)
        cols[colsin] = (v0 != 0)
        Rows.append(rows)
        Cols.append(cols)
        if stop:
            number = k
            break
        if i == iters - 1:
            number = k
            stop = True
            print("Fail to converge! Increase the number of iterations!")
            gc.collect()
            break
        if not row_overlap:
            rowsin[rows] = False
            X = startX[np.ix_(rowsin, colsin)]
            info.append((vc_dict, uc_dict, (u0, v0, d0)))
        elif not col_overlap:
            colsin[cols] = False
            X = startX[np.ix_(rowsin, colsin)]
            info.append((vc_dict, uc_dict, (u0, v0, d0)))
        elif row_overlap and col_overlap:
            X_sub = X[np.ix_(rows, cols)]
            if X_sub.size > 0:
                tempU, tempS, tempVt = np.linalg.svd(X_sub, full_matrices=False)
                correction = tempS[0] * np.outer(tempU[:, 0], tempVt[0, :])
                X_sub = X_sub - correction
                X[np.ix_(rows, cols)] = X_sub
            info.append((vc_dict, uc_dict, (u0, v0, d0)))
        print("")
    if not stop:
        number = nbiclust
    params = {
        "steps": steps, "pcerv": pcerv, "pceru": pceru, "iter": iters, "ss_thr": ss_thr, "size": size, "gamm": gamm,
        "row_overlap": row_overlap, "col_overlap": col_overlap, "rows_nc": rows_nc, "cols_nc": cols_nc,
        "nbiclust": nbiclust, "merr": merr, "row_min": row_min, "col_min": col_min, "pointwise": pointwise,
        "start_iter": start_iter, "savepath": savepath
    }
    if len(Rows) > 0:
        RowxNumber = np.column_stack(Rows)
        NumberxCol = np.column_stack(Cols)
        RowxNumber = RowxNumber[:, :number]
        NumberxCol = NumberxCol[:, :number]
    else:
        RowxNumber = np.array([[]])
        NumberxCol = np.array([[]])
    info.append(params)
    Number = number
    return BiclustResult(params, RowxNumber, NumberxCol, Number, info)


