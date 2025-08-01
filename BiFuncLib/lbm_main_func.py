import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import cholesky, solve, eig
import copy
from GENetLib.fda_func import create_bspline_basis, create_fourier_basis
from GENetLib.fda_func import inprod

from GENetLib.BsplineFunc import BsplineFunc
from BiFuncLib.fem_bifunc import fem_bifunc


def ari(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    xx = x[:, None] == x
    yy = y[:, None] == y
    upper = np.triu_indices(len(x), k=1)
    xx_upper = xx[upper]
    yy_upper = yy[upper]
    a = np.sum(xx_upper & yy_upper)
    b = np.sum(xx_upper & ~yy_upper)
    c = np.sum(~xx_upper & yy_upper)
    d = np.sum(~xx_upper & ~yy_upper)
    ni = b + a
    nj = c + a
    abcd = a + b + c + d
    q = (ni * nj) / abcd
    ari_value = (a - q) / ((ni + nj) / 2 - q)
    return ari_value


def dummy(Z, K):
    Z = np.asarray(Z).astype(int)
    return np.eye(K, dtype=int)[Z - 1]


def empty_class_check(W, Pw = None, display = False):
    nb = 0
    while np.min(np.sum(W, axis=0)) < 1 and nb < 10:
        ll = np.where(np.sum(W, axis=0) < 1)[0]
        for l in ll:
            if Pw is None or len(Pw) == 0:
                ind = np.random.choice(W.shape[0], 1)[0]
            else:
                ind = np.argmax(Pw[:, l])
            W[ind, :] = np.zeros(W.shape[1])
            W[ind, l] = 1
        if display:
            print('Warning: Class is becoming empty! A regularization has been done.')
        nb += 1
    return W


def mypca_fd(fdobj_, center = True):
    fdobj = copy.deepcopy(fdobj_)
    if isinstance(fdobj, list):
        mean_fd = [copy.deepcopy(fd) for fd in fdobj]
        if center:
            for i in range(len(fdobj)):
                coefmean = np.mean(fdobj[i]['coefs'], axis=1)
                fdobj[i]['coefs'] = fdobj[i]['coefs'] - coefmean[:, np.newaxis]
                mean_fd[i]['coefs'] = coefmean
        W_vars = {}
        for i in range(len(fdobj)):
            W_fdobj = inprod(fdobj[i]['basis'], fdobj[i]['basis'])
            W_vars[f'W_var{i+1}'] = W_fdobj
        prow = W_fdobj.shape[0]
        pcol = len(fdobj) * prow
        W1 = np.hstack([W_fdobj, np.zeros((prow, pcol - W_fdobj.shape[1]))])
        W_list = []
        for i in range(1, len(fdobj)):
            left_zeros = np.zeros((prow, i * W_fdobj.shape[1]))
            middle_matrix = W_vars[f'W_var{i+1}']
            right_zeros = np.zeros((prow, pcol - (i+1) * W_fdobj.shape[1]))
            W2 = np.hstack([left_zeros, middle_matrix, right_zeros])
            W_list.append(W2)
        W_tot = np.vstack([W1] + W_list)
        coef = fdobj[0]['coefs'].T
        for i in range(1, len(fdobj)):
            coef = np.hstack((coef, fdobj[i]['coefs'].T))
        mat_interm = coef @ cholesky(W_tot) / np.sqrt(fdobj[0]['coefs'].shape[1] - 1)
        cov = mat_interm.T @ mat_interm
        valeurs_propres, vecteurs_propres = eig(cov)
        valeurs_propres = valeurs_propres.real
        bj = solve(cholesky(W_tot), vecteurs_propres)
        fonctionspropres = copy.deepcopy(fdobj[0])
        fonctionspropres['coefs'] = bj
        scores = coef @ W_tot @ bj
        varprop = valeurs_propres / sum(valeurs_propres)
        return {
            'values': valeurs_propres,
            'harmonics': fonctionspropres,
            'scores': scores,
            'U': bj,
            'varprop': varprop,
            'meanfd': mean_fd,
            'Wmat': W_tot
        }
    else:
        mean_fd = copy.deepcopy(fdobj)
        if center:
            coefmean = np.mean(fdobj['coefs'], axis=1)
            fdobj['coefs'] = fdobj['coefs'] - coefmean[:, np.newaxis]
            mean_fd['coefs'] = np.array([coefmean]).T
        W = inprod(fdobj['basis'], fdobj['basis'])
        W[W < 1e-15] = 0
        coef = fdobj['coefs'].T
        mat_interm = 1 / np.sqrt(coef.shape[0] - 1) * np.dot(coef, cholesky(W))
        cov = mat_interm.T @ mat_interm
        valeurs_propres, vecteurs_propres = eig(cov)
        valeurs_propres = valeurs_propres.real
        fonctionspropres = copy.deepcopy(fdobj)
        bj = solve(cholesky(W), vecteurs_propres)
        fonctionspropres['coefs'] = bj
        scores = inprod(fdobj, fonctionspropres)
        varprop = valeurs_propres / sum(valeurs_propres)
        return {
            'values': valeurs_propres,
            'harmonics': fonctionspropres,
            'scores': scores,
            'U': bj,
            'varprop': varprop,
            'meanfd': mean_fd,
            'Wmat': W
        }


def cattell(x, thd = 0.01):
    sc = np.abs(np.diff(x))
    p = len(x)
    d = p - 1
    for j in range(p - 2):
        if np.prod(sc[(j + 1):] < thd * np.max(sc)):
            d = j + 1
            break
    return d


def estep_Z_cost(xx, alpha, beta, mu, a, b, d, Q, k, l, Wmat):
    nbasis = xx.shape[1]
    Qkl = np.dot(Wmat[k][l], Q[k][l])
    Pa = (xx - np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1)) @ Qkl @ Qkl.T
    Pb = Pa + (np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1)) - xx
    A = 1 / a[k, l] * np.sum(np.square(Pa), axis=1) + 1 / b[k, l] * np.sum(np.square(Pb), axis=1) +d[k, l] * np.log(a[k, l]) + (nbasis - d[k, l]) * np.log(b[k, l]) - 2 * np.log(beta[l])
    return A.T




def estep_W_cost(xx, alpha, beta, mu, a, b, d, Q, k, l, Wmat):
    nbasis = xx.shape[1]
    Qkl = np.dot(Wmat[k][l], Q[k][l])
    Pa = (xx - np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1)) @ Qkl @ Qkl.T
    Pb = Pa + (np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1)) - xx
    A = 1 / a[k, l] * np.sum(np.square(Pa), axis=1) +1 / b[k, l] * np.sum(np.square(Pb), axis=1) +d[k, l] * np.log(a[k, l]) + (nbasis - d[k, l]) * np.log(b[k, l]) - 2 * np.log(alpha[k])
    return A.T


def estep_lik_cost(xx, alpha, beta, mu, a, b, d, Q, k, l, Wmat):
    nbasis = xx.shape[1]
    Qkl = Wmat[k][l] @ np.array(Q[k][l])
    Pa = (xx - np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1)) @ Qkl @ Qkl.T
    Pb = Pa + (np.ones((xx.shape[0], 1)) @ mu[k, l, :].reshape(1, -1) - xx)
    A = 1 / a[k, l] * np.sum(np.square(Pa), axis=1) + 1 / b[k, l] * np.sum(np.square(Pb), axis=1) + d[k, l] * np.log(a[k, l]) + (nbasis - d[k, l]) * np.log(b[k, l]) + nbasis * np.log(2 * np.pi)
    return A.T




def estep_Z(X, alpha, beta, mu, a, b, d, Q, W, Wmat):
    K = len(alpha)
    L = len(beta)
    P = np.full((X.shape[0], K), np.nan)
    for k in range(K):
        A = np.full((X.shape[0], X.shape[1]), np.nan)
        for l in range(L):
            Xl = X[:, W[:, l] == 1, :]
            if np.sum(W[:, l]) == 1:
                A[:, W[:, l] == 1] = estep_Z_cost(Xl, alpha, beta, mu, a, b, d, Q, k, l, Wmat)
            else:
                A[:, W[:, l] == 1] = np.column_stack([estep_Z_cost(Xl[:, j, :], alpha, beta, mu, a, b, d, Q, k, l, Wmat).T for j in range(Xl.shape[1])])
        P[:, k] = np.nansum(-1 / 2 * A, axis=1)
    P = np.array([1 / np.sum(np.exp(row - row[k])) for row in P for k in range(K)]).reshape(P.shape)
    return P


def estep_W(X, alpha, beta, mu, a, b, d, Q, Z, Wmat):
    K = len(alpha)
    L = len(beta)
    P = np.full((X.shape[1], L), np.nan)
    for l in range(L):
        A = np.full((X.shape[0], X.shape[1]), np.nan)
        for k in range(K):
            Xk = X[Z[:, k] == 1, :, :]
            if np.sum(Z[:, k]) == 1:
                A[Z[:, k] == 1, :] = estep_W_cost(Xk, alpha, beta, mu, a, b, d, Q, k, l, Wmat)
            else:
                A[Z[:, k] == 1, :] = np.column_stack([estep_W_cost(Xk[:, j, :], alpha, beta, mu, a, b, d, Q, k, l, Wmat).T for j in range(Xk.shape[1])])
        P[:, l] = np.nansum(-1 / 2 * A, axis=0)
    P = np.array([1 / np.sum(np.exp(row - row[l])) for row in P for l in range(L)]).reshape(P.shape)
    return P


def compute_complete_likelihood(X, alpha, beta, mu, a, b, d, Q, Z, W, Wmat):
    K, L = len(alpha), len(beta)
    lik = np.sum(Z @ np.log(alpha)) + np.sum(W @ np.log(beta))
    for l in range(L):
        for k in range(K):
            if np.sum(W[:, l]) == 1 or np.sum(Z[:, k]) == 1:
                Xkl = X[Z[:, k] == 1][:, W[:, l] == 1]
            else:
                Xkl = np.column_stack([X[Z[:, k] == 1, :, :][:, W[:, l] == 1, :][:, :, t].flatten('F') for t in range(X.shape[2])])
            lik = lik - 0.5 * np.sum(estep_lik_cost(Xkl, alpha, beta, mu, a, b, d, Q, k, l, Wmat))
    return lik


def mode(x):
    ux = np.unique(x)
    counts = np.array([np.sum(x == u) for u in ux])
    return ux[np.argmax(counts)]


def lbm_main_func(data, K, L, maxit = 100, burn = 50, basis_name = 'fourier',
                  nbasis = 15, gibbs_it = 5, display = False, init = 'kmeans'):
    X = copy.deepcopy(data)
    if not isinstance(X, list):
        N, p, T = X.shape
    else:
        N, p, T = X[0].shape
    
    # Create basis functions
    if basis_name == 'spline':
        basis = create_bspline_basis((0, T), nbasis)
    elif basis_name == 'fourier':
        basis = create_fourier_basis((0, T), nbasis)
    else:
        raise ValueError("Unavailable basis functions!")
    nbasis = basis['nbasis']

    # Calculate functional coefficients
    if not isinstance(X, list):
        Xcol = np.column_stack([X[:, :, t].flatten('F') for t in range(X.shape[2])])
        obj = BsplineFunc(basis).smooth_basis(np.arange(1, T+1), Xcol.T)['fd']
        Coefs3 = obj['coefs'].T
        C_tot = np.full((N, p, nbasis), np.nan)
        for j in range(p):
            C_tot[:, j, :] = Coefs3[j * N:(j + 1) * N, :]
        X_tot = X.copy()
    else:
        obj = []
        fus = []
        Coefs2 = []
        for i in range(len(X)):
            X_sel = X[i]
            Xcol = np.column_stack([X_sel[:, :, t].flatten('F') for t in range(X_sel.shape[2])])
            obj_s = BsplineFunc(basis).smooth_basis(np.arange(1, T+1), Xcol.T)['fd']
            Coefs = obj_s['coefs'].T
            C = np.full((N, p, nbasis), np.nan)
            for j in range(p):
                C[:, j, :] = Coefs[j * N:(j + 1) * N, :]
            fus.append(C)
            obj.append(obj_s)
            Coefs2.append(Coefs)
        C_tot = np.concatenate(fus, axis=2)
        X_tot = np.concatenate(X, axis=2)
        Coefs3 = np.concatenate(Coefs2, axis=1)
    
    # Initialize parameter Z
    if init == "kmeans":
        if not isinstance(X, list):
            z1 = np.random.choice(np.arange(N), size=int(0.3 * N), replace=False)
            z2 = np.random.choice(np.arange(p), size=int(0.3 * p), replace=False)
            obj_s = BsplineFunc(basis).smooth_basis(np.arange(1, T+1), Xcol.T)['fd']
            Coefs3_ech = obj_s['coefs'].T
        else:
            z1 = np.random.choice(np.arange(N), size=int(0.3 * N), replace=False)
            z2 = np.random.choice(np.arange(p), size=int(0.3 * p), replace=False)
            Coefs2_ech = []
            for i in range(len(X)):
                X_sel = X[i][np.ix_(z1, z2, range(T))]
                Xcol = np.column_stack([X_sel[:, :, t].flatten('F') for t in range(X_sel.shape[2])])
                obj_s = BsplineFunc(basis).smooth_basis(np.arange(1, T+1), Xcol.T)['fd']
                Coefs = obj_s['coefs'].T
                Coefs2_ech.append(Coefs)
            Coefs3_ech = np.concatenate(Coefs2_ech, axis=1)
    alpha = np.ones(K) / K
    beta = np.ones(L) / L
    if display:
        print('Initialization: FunFEM...')
    if init == 'funFEM':
        if basis_name == 'spline':
            basisfunFEM = create_bspline_basis((0, p * T), nbasis)
        elif basis_name == 'fourier':
            basisfunFEM = create_fourier_basis((0, p * T), nbasis)
        if not isinstance(X, list):
            fd = BsplineFunc(basisfunFEM).smooth_basis(np.arange(1, p * T + 1), np.vstack(np.transpose(X, (2, 1, 0))))['fd']
            while True:
                try:
                    out_funFEM = fem_bifunc(fd, np.atleast_1d(K).tolist(), maxit=10)
                    break
                except ValueError:
                    print('Rerun')
            Z = np.apply_along_axis(lambda x: (x >= np.max(x)).astype(int), 1, out_funFEM["P"])
        else:
            fd = BsplineFunc(basisfunFEM).smooth_basis(np.arange(1, p * T + 1), np.vstack(np.transpose(X[0], (2, 1, 0))))['fd']
            while True:
                try:
                    out_funFEM = fem_bifunc(fd, np.atleast_1d(K).tolist(), maxit=10)
                    break
                except ValueError:
                    print('Rerun')
            Z = np.apply_along_axis(lambda x: (x >= np.max(x)).astype(int), 1, out_funFEM["P"])
    elif init == 'kmeans':
        if display:
            print('Initialization: kmeans...')
        data = np.array([X_tot[i, :, :].flatten('F') for i in range(N)])
        cls_ = KMeans(n_clusters=K, n_init=1, init="random").fit(data).labels_
        Z = dummy(cls_, K)
    else:
        print('Initialization: random...')
        Z = np.random.multinomial(1, alpha / np.sum(alpha), size=N)
    Z = empty_class_check(Z)
    if display:
        print('Z:', np.sum(Z, axis=0))

    # Initialize parameter W
    if init == 'funFEM':
        if basis_name == 'spline':
            basisfunFEM = create_bspline_basis((0, N * T), nbasis)
        elif basis_name == 'fourier':
            basisfunFEM = create_fourier_basis((0, N * T), nbasis)
        if not isinstance(X, list):
            fd = BsplineFunc(basisfunFEM).smooth_basis(np.arange(1, N * T + 1), np.vstack(np.transpose(X, (2, 0, 1))))['fd']
            while True:
                try:
                    out_funFEM = fem_bifunc(fd, np.atleast_1d(L).tolist(), maxit=10)
                    break
                except ValueError:
                    continue
            W = np.apply_along_axis(lambda x: (x >= np.max(x)).astype(int), 1, out_funFEM["P"])
        else:
            fd = BsplineFunc(basisfunFEM).smooth_basis(np.arange(1, N * T + 1), np.vstack(np.transpose(X[0], (2, 0, 1))))['fd']
            while True:
                try:
                    out_funFEM = fem_bifunc(fd, np.atleast_1d(L).tolist(), maxit=10)
                    break
                except ValueError:
                    print('Rerun')
            W = np.apply_along_axis(lambda x: (x >= np.max(x)).astype(int), 1, out_funFEM["P"])
    elif init == 'kmeans':
        data = np.array([X_tot[:, j, :].flatten('F') for j in range(p)])
        cls_ = KMeans(n_clusters=L, n_init=1, init="random").fit(data).labels_
        W = dummy(cls_, L)
    else:
        W = np.random.multinomial(1, beta / np.sum(beta), size=p)
    W = empty_class_check(W)
    if display:
        print('W:', np.sum(W, axis=0))

    # Initialize other parameters
    Q = [[[] for _ in range(L)] for _ in range(K)]
    Wmat = [[[] for _ in range(L)] for _ in range(K)]
    if not isinstance(X, list):
        mu = np.full((K, L, nbasis), np.nan)
    else:
        mu = np.full((K, L, len(X) * nbasis), np.nan)
    if init == "kmeans":
        base_coefs = np.mean(Coefs3_ech, axis=0)
    else:
        base_coefs = np.mean(Coefs3, axis=0)
    for l in range(L):
        mu[:, l, :] = (np.tile(base_coefs, K) + np.tile(np.random.normal(0, 0.01, K * nbasis), base_coefs.size // nbasis)).reshape(K, -1)
    a = np.full((K, L), np.nan)
    b = np.full((K, L), np.nan)
    d = np.full((K, L), np.nan)
    lik = np.full(maxit, np.nan)
    Alphas = np.full((maxit, K), np.nan)
    Betas = np.full((maxit, L), np.nan)
    Zs = np.full((N, maxit), np.nan)
    Ws = np.full((p, maxit), np.nan)
    Alphas[0, :] = alpha
    Betas[0, :] = beta
    Zs[:, 0] = np.argmax(Z, axis=1)
    Ws[:, 0] = np.argmax(W, axis=1)

    # SEM-Gibbs iteration
    for it in range(maxit):
        if display:
            print(f"Iteration {it+1}/{maxit}", end=' ')
        # M step
        for k in range(K):
            for l in range(L):
                if np.sum(W[:, l]) == 1 or np.sum(Z[:, k]) == 1:
                    x = C_tot[Z[:, k] == 1, :, :][:, W[:, l] == 1, :]
                else:
                    x = np.column_stack([C_tot[Z[:, k] == 1, :, :][:, W[:, l] == 1, :][:, :, t].flatten('F') for t in range(C_tot.shape[2])])
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                alpha[k] = np.sum(Z[:, k] == 1) / N
                alpha[alpha < 0.05] = 0.05
                beta[l] = np.sum(W[:, l] == 1) / p
                beta[beta < 0.05] = 0.05
                mu[k, l, :] = np.mean(x, axis=0)
                if not isinstance(X, list):
                    obj['coefs'] = x.T
                else:
                    for f in range(len(X)):
                        obj[f]['coefs'] = x[:, f * nbasis:(f + 1) * nbasis].T           
                dc = mypca_fd(obj)
                d[k, l] = cattell(dc['values'])
                a[k, l] = np.mean(dc['values'][:int(d[k, l])])
                b[k, l] = np.mean(dc['values'][int(d[k, l]):len(dc['values'])])
                Q[k][l] = dc['U'][:, :int(d[k, l])]
                Wmat[k][l] = dc['Wmat']
        
        # SE step
        for r in range(gibbs_it):
            # Update Z
            Pz = estep_Z(C_tot, alpha, beta, mu, a, b, d, Q, W, Wmat)
            Z = np.array([np.random.multinomial(1, row) for row in Pz])
            Z = empty_class_check(Z, Pz)
            if display:
                print("Z:", np.sum(Z, axis=0))
            # Update W
            Pw = estep_W(C_tot, alpha, beta, mu, a, b, d, Q, Z, Wmat)
            W = np.array([np.random.multinomial(1, row) for row in Pw])
            W = empty_class_check(W, Pw)
            if display:
                print("W:", np.sum(W, axis=0))
            if np.min(np.sum(W, axis=0)) < 1 or np.min(np.sum(Z, axis=0)) < 1:
                raise Exception("One class is empty, re-run the algorithm or choose an other number of clusters") 
        lik[it] = compute_complete_likelihood(C_tot, alpha, beta, mu, a, b, d, Q, Z, W, Wmat)
        Alphas[it, :] = alpha
        Betas[it, :] = beta
        Zs[:, it] = np.argmax(Z, axis=1)
        Ws[:, it] = np.argmax(W, axis=1)
        if burn > 5 and it > burn:
            if sum(abs(np.diff(lik[it:(it-6):-1]))) < 1e-6:
                burn = it - 5
                break

    # Averaging and computing MAP parameters
    alpha = np.mean(Alphas[burn:it+1, :], axis=0)
    beta = np.mean(Betas[burn:it+1, :], axis=0)
    Z = dummy(np.apply_along_axis(mode, 1, Zs[:, burn:it+1]), K)
    Z = empty_class_check(Z, Pz)
    W = dummy(np.apply_along_axis(mode, 1, Ws[:, burn:it+1]), L)
    W = empty_class_check(W, Pw)
    for k in range(K):
        for l in range(L):
            if np.sum(W[:, l]) == 1 or np.sum(Z[:, k]) == 1:
                x = C_tot[Z[:, k] == 1, :, :][:, W[:, l] == 1, :]
            else:
                x = np.column_stack([C_tot[Z[:, k] == 1, :, :][:, W[:, l] == 1, :][:, :, t].flatten('F') for t in range(C_tot.shape[2])])
            mu[k, l, :] = np.mean(x, axis=0)
            if not isinstance(X, list):
                obj['coefs'] = x.T
            else:
                for f in range(len(X)):
                    obj[f]['coefs'] = x[:, f * nbasis:(f + 1) * nbasis].T           
            dc = mypca_fd(obj)
            d[k, l] = cattell(dc['values'])
            a[k, l] = np.mean(dc['values'][:int(d[k, l])])
            b[k, l] = np.mean(dc['values'][int(d[k, l]):len(dc['values'])])
            Q[k][l] = dc['U'][:, :int(d[k, l])]
            Wmat[k][l] = dc['Wmat']
    b[b < 1e-6] = 1e-6

    # ICL-BIC criterion
    if not isinstance(X, list):
        nbasis2 = nbasis
    else:
        nbasis2 = len(X) * nbasis
    nu = K * L * (nbasis2 + 2) + np.sum(d * (nbasis2 - (d + 1) / 2))
    crit = compute_complete_likelihood(C_tot, alpha, beta, mu, a, b, d, Q, Z, W, Wmat) - ((K - 1) / 2) * np.log(N) - ((L - 1) / 2) * np.log(p) - (nu / 2) * np.log(N * p)

    # Return results
    prms = {
        'alpha': alpha,
        'beta': beta,
        'mu': mu,
        'a': a,
        'b': b,
        'd': d,
        'Q': Q
    }
    allPrms = {
        'Alphas': Alphas[:it+1, :],
        'Betas': Betas[:it+1, :],
        'Zs': Zs[:, :it+1],
        'Ws': Ws[:, :it+1]
    }
    out = {
        'basisName': basis_name,
        'nbasis': nbasis,
        'T': T,
        'K': K,
        'L': L,
        'prms': prms,
        'Z': Z,
        'W': W,
        'row_clust': np.argmax(Z, axis=1),
        'col_clust': np.argmax(W, axis=1),
        'allPrms': allPrms,
        'loglik': lik,
        'icl': crit
    }
    return out

