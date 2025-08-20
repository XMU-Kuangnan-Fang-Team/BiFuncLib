import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd, pinv
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
from GENetLib.fda_func import inprod
from scipy.cluster.hierarchy import linkage, cut_tree


def criteria(loglik, T, prms, n):
    K = prms['K']
    p = prms['p']
    model = prms['model']
    comp = 0
    if model == 'DkBk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + np.square(K)*(K-1)//2 + K
    elif model == 'DkB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + np.square(K)*(K-1)//2 + 1
    elif model == 'DBk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + K*(K-1)//2 + K
    elif model == 'DB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + K*(K-1)//2 + 1
    elif model == 'AkjBk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + np.square(K)
    elif model == 'AkjB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + K*(K-1) + 1
    elif model == 'AkBk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + 2*K
    elif model == 'AkB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + K + 1
    elif model == 'AjBk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + (K-1) + K
    elif model == 'AjB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + (K-1) + 1
    elif model == 'ABk':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + K + 1
    elif model == 'AB':
        comp = (K-1) + K*(K-1) + (K-1)*(p - K/2) + 2
    aic = loglik - comp 
    bic = loglik - 0.5 * comp * np.log(n)
    T[T < 1e-6] = 1e-6
    icl = loglik - 0.5 * comp * np.log(n) - np.sum(T * np.log(T))
    return {'aic': aic, 'bic': bic, 'icl': icl, 'nbprm': comp}


def estep(prms, fd, U):
    Y = np.asarray(fd['coefs'].T)
    n = Y.shape[0]
    p = Y.shape[1]
    K = prms['K']
    prop = prms['prop']
    D = prms['D']
    d = K - 1
    QQ = np.zeros((n, K))
    T = np.zeros((n, K))
    for k in range(K):
        bk = D[k, p-1, p-1]
        mY = prms['my'][k, :]
        YY = Y - np.tile(mY, (n, 1))
        projYY = YY @ U @ U.T
        if d == 1:
            for i in range(n):
                QQ[i, k] = (1 / D[k, 0, 0] * np.sum(np.square(projYY[i, :])) +
                            1 / bk * np.sum(np.square(YY[i, :] - projYY[i, :])) +
                            (p - d) * np.log(bk) + np.log(D[k, 0, 0]) -
                            2 * np.log(prop[k]) + p * np.log(2 * np.pi))
        else:
            sY = U @ inv(D[k, :d, :d]) @ U.T
            for i in range(n):
                QQ[i, k] = (projYY[i, :] @ sY @ projYY[i, :].T +
                            1 / bk * np.sum(np.square(YY[i, :] - projYY[i, :])) +
                            (p - d) * np.log(bk) + np.log(np.linalg.det(D[k, :d, :d])) -
                            2 * np.log(prop[k]) + p * np.log(2 * np.pi))
    A = -0.5 * QQ
    loglik = np.sum(np.log(np.sum(np.exp(A - np.max(A, axis=1, keepdims=True)), axis=1)) + np.max(A, axis=1))
    for k in range(K):
        T[:, k] = 1 / np.sum(np.exp(0.5 * (QQ[:, k].reshape(-1, 1) - QQ)), axis=1)
    return {'T': T, 'loglik': loglik}


def fstep(fd, T, lambda_):
    if np.min(np.sum(T, axis=0)) <= 1:
        raise ValueError("One cluster is almost empty!")
    G = np.asarray(fd['coefs'].T)
    d = T.shape[1] - 1
    basisobj = fd['basis']
    W = inprod(basisobj, basisobj)    
    Ttilde = T / np.sqrt(T.sum(axis=0))
    U = svd(pinv(G.T @ G @ W) @ (G.T @ Ttilde @ Ttilde.T @ G @ W), full_matrices=False)[0][:, :d]
    if lambda_ > 0:
        X = G @ U
        Utilde = U.copy()
        for i in range(d):
            x_predict = X[:, i].reshape(-1, 1)
            enet = ElasticNet(alpha=lambda_, l1_ratio=0.5, fit_intercept=False)
            enet.fit(x_predict, G)
            coef = enet.coef_
            Utilde[:, i] = (coef / np.sqrt(np.sum(np.square(coef)))).ravel()
        U = svd(Utilde, full_matrices=False)[0]
    return U


def fem_main_func(fd, K, model='AkjBk', init='kmeans', lambda_=0, Tinit=None,
                  maxit=50, eps=1e-8, graph=False):
    Y = np.asarray(fd['coefs'].T)
    n = Y.shape[0]
    Lobs = np.full(maxit + 1, -np.inf)
    if init == 'user':
        T = Tinit
    elif init == 'kmeans':
        ind = KMeans(n_clusters=K, init = 'random').fit(Y).labels_
        T = np.zeros((n, K))
        T[np.arange(n), ind] = 1
        T.sum(axis=0)
    elif init == 'random':
        T = np.random.multinomial(1, [1/K]*K, size=n)
    elif init == 'hclust':
        Z = linkage(Y, method='ward')
        ind = cut_tree(Z, n_clusters=K).flatten()
        T = np.zeros((n, K))
        T[np.arange(n), ind] = 1
    V = fstep(fd, T, lambda_)
    prms = mstep(fd, V, T, model = model)
    res_estep = estep(prms, fd, V)
    T = res_estep['T']
    Lobs[0] = res_estep['loglik']
    Linf_new = Lobs[0]
    for i in range(maxit):
        V = fstep(fd, T, lambda_)
        prms = mstep(fd, V, T, model=model)
        res_estep = estep(prms, fd, V)
        T = res_estep['T']
        Lobs[i+1] = res_estep['loglik']
        if i >= 1:
            acc = (Lobs[i+1] - Lobs[i]) / (Lobs[i] - Lobs[i-1])
            Linf_old = Linf_new
            Linf_new = Lobs[i] + 1 / (1 - acc) * (Lobs[i+1] - Lobs[i])
            if Linf_old is not None and (abs(Linf_new - Linf_old) < eps or np.isnan(Linf_new)):
                break
    if graph:
        plt.figure(figsize=(5, 3))
        plt.plot(Lobs[:i+2], 'r.')
        plt.tick_params(axis='both', labelsize=8)
        plt.xlabel('Iterations')
        plt.ylabel('Log-likelihood (observed)')
        plt.show()
    cls_ = np.argmax(T, axis=1)
    crit = criteria(Lobs[i+1], T, prms, n)
    W = inprod(fd['basis'], fd['basis'])
    U = W.T @ V
    return {
        'model': model,
        'K': K,
        'cls': cls_,
        'P': T,
        'prms': prms,
        'U': U,
        'aic': crit['aic'],
        'bic': crit['bic'],
        'icl': crit['icl'],
        'loglik': Lobs[1:i+2],
        'll': Lobs[i+1],
        'nbprm': crit['nbprm']
    }


def mstep(fd, U, T, model):
    Y = fd['coefs'].T
    n = Y.shape[0]
    p = Y.shape[1]
    K = T.shape[1]
    d = K - 1
    mu = np.zeros((K, d))
    m = np.zeros((K, p))
    prop = np.zeros(K)
    D = np.zeros((K, p, p))
    W = inprod(fd['basis'], fd['basis'])
    U = W.T @ U
    X = Y @ U
    for k in range(K):
        nk = np.sum(T[:, k])
        prop[k] = nk / n
        mu[k, :] = np.sum(T[:, k].reshape(-1, 1) * np.ones((n,d)) * np.array(X), axis=0) / nk
        m[k, :] = np.sum(T[:, k].reshape(-1, 1) * np.ones((n,p)) * np.array(Y), axis=0) / nk
        YY = Y - np.tile(m[k, :], (n, 1))
        Ck = (T[:, k].reshape(-1, 1) * np.array(YY)).T @ np.array(YY) / (nk - 1)
        C = np.cov(Y, rowvar=False)
        if model == 'DkBk':
            D[k, :d, :d] = (Ck @ U).T @ U
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'DkB':
            D[k, :d, :d] = (Ck @ U).T @ U
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'DBk':
            D[k, :d, :d] = (C @ U).T @ U
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'DB':
            D[k, :d, :d] = (C @ U).T @ U
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AkjBk':
            if d == 1:
                D[k, 0, 0] = np.diag((Ck @ U).T @ U)
            else:
                D[k, :d, :d] = np.diag(np.diag((Ck @ U).T @ U))
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AkjB':
            if d == 1:
                D[k, 0, 0] = np.diag((Ck @ U).T @ U)
            else:
                D[k, :d, :d] = np.diag(np.diag((Ck @ U).T @ U))
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AkBk':
            if d == 1:
                D[k, 0, 0] = np.trace((Ck @ U).T @ U) / d
            else:
                D[k, :d, :d] = np.eye(d) * (np.trace((Ck @ U).T @ U) / d)
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AkB':
            if d == 1:
                D[k, 0, 0] = np.trace((Ck @ U).T @ U) / d
            else:
                D[k, :d, :d] = np.eye(d) * (np.trace((Ck @ U).T @ U) / d)
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AjBk':
            if d == 1:
                D[k, 0, 0] = np.diag((C @ U).T @ U)
            else:
                D[k, :d, :d] = np.diag(np.diag((C @ U).T @ U))
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AjB':
            if d == 1:
                D[k, 0, 0] = np.diag((C @ U).T @ U)
            else:
                D[k, :d, :d] = np.diag(np.diag((C @ U).T @ U))
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'ABk':
            if d == 1:
                D[k, 0, 0] = np.trace((C @ U).T @ U)
            else:
                D[k, :d, :d] = np.eye(d) * (np.trace((C @ U).T @ U) / d)
            bk = (np.trace(Ck) - np.trace((Ck @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
        elif model == 'AB':
            if d == 1:
                D[k, 0, 0] = np.trace((C @ U).T @ U)
            else:
                D[k, :d, :d] = np.eye(d) * (np.trace((C @ U).T @ U) / d)
            bk = (np.trace(C) - np.trace((C @ U).T @ U)) / (p - d)
            bk = max(bk, 1e-3)
            D[k, d:, d:] = np.eye(p - d) * bk
    return {'K': K, 'p': p, 'mean': mu, 'my': m, 'prop': prop, 'D': D, 'model': model}

