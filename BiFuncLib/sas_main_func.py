import numpy as np
from GENetLib.fda_func import create_bspline_basis, eval_basis, fdpar
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix

from GENetLib.BsplineFunc import BsplineFunc


def get_sigma(x, curve, time, S, piigivej, gcov, n_i, gamma, mu):
    N = piigivej.shape[0]
    K = piigivej.shape[1]
    q = gcov.shape[0]
    n_i = np.asarray(n_i, dtype=int)
    n_vec = np.empty(len(n_i) + 1, dtype=int)
    n_vec[0] = -1
    n_vec[1:] = np.cumsum(n_i) - 1
    mu = mu.T
    sigma_val = 0.0
    for i in range(N):
        start_idx = n_vec[i] + 1
        end_idx = n_vec[i+1]
        y = x.reshape(-1,1)[start_idx:end_idx+1, :]
        Si = S[start_idx:end_idx+1, :]
        for j in range(K):
            pp = gamma[i, j, :]
            residual = y.flatten() - np.dot(Si, mu[:, j]) - np.dot(Si, pp)
            dot_term = np.dot(residual, residual.T)
            gcov_slice = gcov[:, i*q:(i+1)*q]
            trace_term = np.trace(np.dot(Si, np.dot(gcov_slice, Si.T)))
            sigma_val += piigivej[i, j] * (dot_term + trace_term)
    return np.array(sigma_val)


def get_numden(x, curve, time, S, piigivej, gcov, n_i, gamma):
    N = piigivej.shape[0]
    K = piigivej.shape[1]
    q = gcov.shape[0]
    sum_num = np.zeros((K*q,))
    sum_den = np.zeros((K*q, K*q))
    rep1 = np.ones(len(n_i), dtype=int)
    n_vec = np.empty(len(n_i)+1, dtype=int)
    n_vec[0] = -1
    n_vec[1:] = np.cumsum(n_i) - rep1
    matp = np.eye(K*q)
    pii_vec = np.zeros((K*q,))
    for i in range(N):
        start_idx = n_vec[i] + 1
        end_idx = n_vec[i+1] + 1
        pro = x.reshape(-1,1)[start_idx:end_idx, :]
        y = np.tile(pro, (K, 1))
        y = y.flatten()
        Si = S[start_idx:end_idx, :]
        ni_i = n_i[i]
        Si_mat = np.zeros((ni_i * K, K*q))
        for j in range(K):
            Si_mat[j*ni_i : (j+1)*ni_i, j*q : (j+1)*q] = Si
            pij_j = piigivej[i, j]
            pii_vec[j*q : (j+1)*q] = np.full((q,), pij_j)
        pp = gamma[i, :, :]
        gammai_vec = np.ravel(pp.T, order='F')
        term1 = Si_mat.T @ y
        term2 = (Si_mat.T @ Si_mat) @ gammai_vec
        diff_term = term1 - term2
        sum_num = sum_num + pii_vec * diff_term
        matp = np.diag(pii_vec)
        sum_den = sum_den + matp @ (Si_mat.T @ Si_mat)
    return sum_num, sum_den


def get_Estep(par, data, vars_, S, hard, n_i):
    gamma = vars_['gamma']
    N = gamma.shape[0]
    K = gamma.shape[1]
    q = gamma.shape[2]
    x = data['x']
    pi_mat = par['pi']
    if pi_mat.ndim > 1:
        pi_vec = pi_mat.flatten()
    else:
        pi_vec = pi_mat
    piigivej = vars_['piigivej'].copy()
    rep1 = np.ones(len(n_i), dtype=int)
    n_vec = np.empty(len(n_i) + 1, dtype=int)
    n_vec[0] = -1
    n_vec[1:] = np.cumsum(n_i) - rep1
    Gamma = par['Gamma']
    Cgamma = np.zeros_like(Gamma)
    mu1 = par['mu']
    mu = mu1.T
    sigma = par['sigma']
    gprod = np.zeros((q, q * N))
    gcov  = np.zeros((q, q * N))
    for i in range(N):
        ni_val = int(n_i[i])
        rep2 = np.eye(ni_val)
        start_index = n_vec[i] + 1
        end_index = n_vec[i+1] + 1
        Si = S[start_index:end_index, :]
        y  = x.reshape(-1,1)[start_index:end_index, :]
        yrep = np.tile(y, (1, K))
        invar = np.diag(np.full(ni_val, 1.0/sigma))
        temp_inv = np.linalg.inv(rep2 + Si @ Gamma @ Si.T @ invar)
        Cgamma = Gamma - (Gamma @ Si.T @ temp_inv @ invar @ Si @ Gamma)
        centx = yrep - (Si @ mu)
        gamma[i, :, :] = (Cgamma @ Si.T @ invar @ centx).T
        covx = Si @ Gamma @ Si.T + np.linalg.inv(invar)
        diag2 = centx.T @ np.linalg.inv(covx) @ centx
        d = np.exp(- np.diag(diag2) / 2.0) * pi_vec
        if np.all(d == 0):
            d[0] = 1e-200
        piigivej[i, :] = d / np.sum(d)
        if hard:
            m = np.argmax(d)
            piigivej[i, :] = 0
            piigivej[i, m] = 1
        gamma_row = gamma[i, :, :]
        pirep = np.tile(piigivej[i, :], (q, 1))
        temp_elem = np.multiply(gamma_row, pirep.T)
        gprodi = (gamma_row.T) @ temp_elem + Cgamma
        gprod[:, i*q:(i+1)*q] = gprodi
        gcov[:, i*q:(i+1)*q] = Cgamma
    return gamma, piigivej, gprod, gcov


def loglik(parameters, data = None, X = None, timeindex = None, curve = None, grid = None,
           vars_ = None, FullS=None, W=None, AW_vec=None, P_tot=None, lambda_s=None, lambda_l=None):
    if data is None:
        if X is None:
            raise ValueError("No data provided")
        X = np.array(X)
        if X.ndim == 2:
            n_t, n_obs = X.shape
            if grid is None:
                grid = np.linspace(0, 1, n_t)
            vec_x = X.flatten('F')
            timeindex_vec = np.tile(np.arange(1, len(grid)+1), n_obs)
            curve_vec = np.repeat(np.arange(1, n_obs+1), len(grid))
            data = {"x": vec_x, "timeindex": timeindex_vec, "curve": curve_vec}
        elif X.ndim == 1:
            if grid is None:
                raise ValueError("For irregularly sampled functional data grid must be provided")
            if timeindex is None:
                raise ValueError("For irregularly sampled functional timeindex grid must be provided")
            if curve is None:
                raise ValueError("For irregularly sampled functional timeindex curve must be provided")
            data = {"x": np.atleast_2d(X), "timeindex": timeindex, "curve": curve}
        else:
            raise ValueError("No data provided")
    CK = 0
    perc_rankpapp2 = None
    gamma_var = vars_["gamma"]
    pi_vals = parameters["pi"]
    S = FullS[np.array(data["timeindex"]) - 1, :]
    unique_curves = np.unique(data["curve"])
    N = len(unique_curves)
    G = gamma_var.shape[1]
    q = gamma_var.shape[2]
    Gamma_mat = parameters["Gamma"]
    if perc_rankpapp2 is not None:
        print(np.linalg.det(Gamma_mat))
        print(2)
        u, s, vh = np.linalg.svd(Gamma_mat)
        cumsum_ratio = np.cumsum(s) / np.sum(s)
        p = int(np.argmax(cumsum_ratio >= perc_rankpapp2))
        s[p+1:] = 0
        Gamma_mat = u @ np.diag(s) @ u.T
    if parameters.get("mu") is None:
        lambda_zero = parameters["lambda_zero"]
        part1 = np.full((q, G), lambda_zero)
        part2 = parameters["Lambda"] @ np.array(parameters["alpha"]).T
        mu = (part1 + part2).T
    else:
        mu = parameters["mu"]
    loglk = 0.0
    for i in range(N):
        current_curve = unique_curves[i]
        idx = (data["curve"] == current_curve)
        y = np.ravel(data["x"][idx])
        Si = S[idx, :]
        ni = Si.shape[0]
        sigma_val = parameters["sigma"]
        invvar = (1/sigma_val) * np.eye(ni)
        covx = Si @ Gamma_mat @ Si.T + np.linalg.inv(invvar)
        L = np.linalg.cholesky(covx)
        covx_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
        covx_det = np.linalg.det(covx)
        temp = 0.0
        for ll in range(G):
            diff = y - Si @ parameters["mu"].T[:, ll]
            exponent = -0.5 * (diff @ (covx_inv @ diff.T))
            exp_term = np.exp(exponent)
            temp += pi_vals[ll] * ((2 * np.pi)**(-ni/2)) * (covx_det**(-0.5)) * exp_term
        if temp == 0:
            temp = 1e-20
        loglk += np.log(temp)
    if (lambda_l is not None) or (lambda_s is not None):
        mu_vec = np.ravel(mu.T, order='F')
        p_l = 0.0
        if lambda_l is not None:
            p_l = lambda_l * (AW_vec.T @ np.abs(P_tot @ mu_vec))      
        p_s = 0.0
        if lambda_s is not None:
            for ll in range(G):
                current_mu = mu[ll, :]
                p_s += current_mu.T @ W @ current_mu
            p_s = lambda_s * p_s
        pi_logs = np.log(pi_vals)
        if np.isnan(np.sum(pi_logs)):
            p_pi = CK * np.sum(pi_logs)
        else:
            p_pi = 0
        ploglk = loglk - p_l - p_s + p_pi
        out = np.round(np.array([loglk.item(), ploglk.item()]), 2)
    else:
        out = loglk
    return out


def sasfclust_init(data, pert = 0, grid = list(np.linspace(0.01, 1, 100)),
                  q = 5, G = 1, der = 0, gamma_ada = 1, lambda_s_ini = None,
                  init = 'kmeans', varcon = 'diagonal'):
   
    # Produce spline basis matrix
    S = FullS = None
    basis = create_bspline_basis(rangeval = [grid[0], grid[-1]], nbasis = q)
    FullS = eval_basis(grid, basis)
    W = BsplineFunc(basis).penalty_matrix()
    S = FullS[data['timeindex'] - 1, :]
    if G != 1:
        P = np.zeros((((G-1)**2 + (G-1)) // 2, G))
        ind = [0, G-2]
        for ii in range(G-1):
            P[ind[0]:ind[1]+1, ii] = 1
        
            if (ind[1] - ind[0] + 1) == 1:
                aa = -1
            else:
                aa = -np.eye(ind[1] - ind[0] + 1)
            if ii+1 < G:
                P[ind[0]:ind[1]+1, ii+1:G] = aa
            ind = [ind[0] + (G-1-ii), ind[1] + (G-1-ii-1)]
            ind = [min((((G-1)**2 + (G-1)) // 2, ind[0])), min((((G-1)**2 + (G-1)) // 2, ind[1]))]
        
        P_tot = np.zeros((q * ((G-1)**2 + (G-1)) // 2, G * q))
        for ii in range(G):
            P_tot[:, ii*q:(ii+1)*q] = np.kron(P[:, ii].reshape(-1, 1), np.eye(q))
        P_tot = csr_matrix(P_tot)
    else:
        P = P_tot = np.eye(q)
    
    # Weight approximation L1 penalty
    order = q - len(basis['params'])
    breaks = basis['params']
    ext_break = np.concatenate([np.repeat(grid[0], order), breaks, np.repeat(grid[-1], order)])
    weights_vec = np.repeat((ext_break[order:] - ext_break[:-order]) / order, ((G-1)**2+(G-1))/2)
    N = len(np.unique(data['curve']))
    
    # Compute initial estimate of basis coefficients.
    if pert is not None:
        points = np.zeros((N, q))
        for i in range(N):
            Si = S[data['curve'] == np.unique(data['curve'])[i]]
            xi = data['x'][data['curve'] == np.unique(data['curve'])[i]]
            points[i, :] = np.linalg.inv(Si.T @ Si + pert * np.eye(q)) @ Si.T @ xi
    else:
        d = np.array([len(data['curve'][data['curve'] == np.unique(data['curve'])[i]]) for i in range(N)])
        e = [data['timeindex'][data['curve'] == np.unique(data['curve'])[i]] for i in range(N)]
        if len(np.unique(d)) == 1 and len([np.unique(e)]) == 1:
            basis_start = basis
            grid_i = list(np.array(grid)[data['timeindex'][data['curve'] == np.unique(data['curve'])[0]] - 1])
            X = np.array([data['x'][data['curve'] == i] for i in np.unique(data['curve'])]).T
            loglam = np.arange(-10, 6.25, 0.25)
            Gcvsave = np.zeros(len(loglam))
            for i in range(len(loglam)):
                fdPari = fdpar(fdobj = basis_start, Lfdobj=2, lambda_ = 10**loglam[i])
                Sm_i = BsplineFunc(fdPari).smooth_basis(grid_i, X)
                Gcvsave[i] = np.sum(Sm_i['gcv'])
            lambda_s_sm = 10**loglam[np.argmin(Gcvsave)] if lambda_s_ini is None else lambda_s_ini
            fdPari = fdpar(fdobj = basis_start, Lfdobj=2, lambda_ = lambda_s_sm)
            points = np.asarray(BsplineFunc(fdPari).smooth_basis(grid_i, X)['fd']['coefs'].T)
        else:
            basis_start = basis
            loglam = np.arange(-3, 1.25, 0.25)
            Gcvsave = np.zeros((N, len(loglam)))
            points = np.zeros((N, q))
            lambda_s_i_vec = np.zeros(N)
            for i in range(N):
                xi = data['x'][data['curve'] == np.unique(data['curve'])[i]]
                grid_i = list(np.array(grid)[data['timeindex'][data['curve'] == np.unique(data['curve'])[i]] - 1])
                for l in range(len(loglam)):
                    fdPari = fdpar(fdobj = basis_start, Lfdobj=2, lambda_ = 10**loglam[i])
                    Sm_i = BsplineFunc(fdPari).smooth_basis(grid_i, xi)
                    Gcvsave[i, l] = np.sum(Sm_i['gcv'])
                lambda_s_i_vec[i] = 10**loglam[np.argmin(Gcvsave[i, :])] if lambda_s_ini is None else lambda_s_ini
                fdPari = fdpar(fdobj = basis_start, Lfdobj=2, lambda_ = lambda_s_i_vec[i])
                points[i, :] = np.asarray(BsplineFunc(fdPari).smooth_basis(grid_i, xi)['fd']['coefs'])
    
    # Initialize clusters
    if G > 1:
        if init == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=G, n_init=10000)
            class_ = kmeans.fit_predict(points)
        elif init == 'model-based':
            gmm = GaussianMixture(n_components=G, covariance_type='spherical', max_iter=1000, random_state=42)
            class_ = gmm.fit_predict(points)
        elif init == 'hierarchical':
            Z = linkage(points, method='average')
            class_ = fcluster(Z, t=G, criterion='maxclust') - 1
        else:
            raise ValueError("Wrong initialization!")
    else:
        class_ = np.ones(N, dtype=int)
    piigivej = np.zeros((N, G))
    piigivej[np.arange(N), class_] = 1
    pi = np.mean(piigivej, axis=0)
    classmean = np.zeros((G, q))
    for k in range(G):
        classmean[k, :] = np.mean(points[class_ == k, :], axis=0)
    mu = mu_start = classmean
    gamma = np.zeros((N, G, q))
    gprod = None
    if G == 1:
        for i in range(N):
            gamma[i, :, :] = (points[i, :] - mu)[None, :]
            block = gamma[i, :, :] @ gamma[i, :, :].T
            gprod = block if gprod is None else np.hstack((gprod, block))
    else:
        for i in range(N):
            gamma[i, :, :] = (points[i, :] - mu)[None, :]
            block = gamma[i, :, :].T @ gamma[i, :, :]
            gprod = block if gprod is None else np.hstack((gprod, block)) 
    N = gamma.shape[0]
    ind = np.tile(np.eye(q, dtype=int), (N, 1))
    Gamma = gprod @ ind / N
    if varcon == 'diagonal':
        Gamma = np.diag(np.diag(Gamma))
    elif varcon == 'equal':
        Gamma = np.eye(q) * np.sum(np.diag(Gamma)) / q
    gcov = np.zeros((q, N * q))
    if G != 1:
        AW = np.zeros((((G - 1)**2 + (G - 1)) // 2, q))
        for i in range(q):
            product = np.dot(P, mu_start[:, i])
            AW[:, i] = 1 / (np.abs(product) ** gamma_ada)
        AW_vec = AW.flatten('F') * weights_vec
    else:
        AW_vec = np.ones(q)
    n = len(data['curve'])
    n_i = np.array([len(data['curve'][data['curve'] == np.unique(data['curve'])[i]]) for i in range(N)])
    sigma = get_sigma(data['x'], data['curve'], data['timeindex'], S, piigivej, gcov, n_i, gamma, mu) / n
    if sigma.shape == (1, 1):
        sigma = np.array(sigma)[0]
    
    # Return results    
    return {
        'S': S,
        'W': W,
        'AW_vec': AW_vec,
        'P_tot': P_tot,
        'P': P,
        'FullS': FullS,
        'parameters': {
            'mu': mu_start,
            'sigma': sigma,
            'pi': pi,
            'Gamma': Gamma
        },
        'vars': {
            'gamma': gamma,
            'piigivej': piigivej,
            'gprod': gprod,
            'gcov': gcov
        },
        'basis': basis
    }


def sasfclust_Mstep(parameters, data, vars_, S, tol, hard, lambda_s, lambda_l,
                    W, AW_vec, P_tot, par_LQA, CK, perc_rankpapp, varcon):
    G = parameters['mu'].shape[0]
    mu = parameters['mu']
    gamma = vars_['gamma']
    gcov = vars_['gcov']
    curve = data['curve']
    piigivej = vars_['piigivej']
    N = gamma.shape[0]
    n = len(curve)
    q = S.shape[1]
    if hard:
        parameters['pi'] = np.repeat(1.0 / G, G)
    else:
        parameters['pi'] = (np.apply_along_axis(np.mean, 0, piigivej) * N + CK) / (N + G * CK)
    ind = np.tile(np.eye(q, dtype=int), (N, 1))
    if perc_rankpapp is not None:
        gsvd = np.linalg.svd(vars_['gprod'] @ ind / N)
        p = np.where(np.cumsum(gsvd[1]) / np.sum(gsvd[1]) >= perc_rankpapp)[0][0]
        gsvd[1][p:] = 0
        parameters['Gamma'] = gsvd[0] @ np.diag(gsvd[1]) @ gsvd[0].T
    else:
        parameters['Gamma'] = vars_['gprod'] @ ind / N

    if varcon == "diagonal":
        parameters['Gamma'] = np.diag(np.diag(parameters['Gamma']))
    elif varcon == "equal":
        parameters['Gamma'] = np.eye(q) * np.sum(np.diag(parameters['Gamma'])) / q
    W_star = np.zeros((q * G, q * G))
    for i in range(G):
        W_star[i * q:(i + 1) * q, i * q:(i + 1) * q] = W
    n_i = np.array([np.sum(curve == unique_curve) for unique_curve in np.unique(curve)])
    numden = get_numden(data['x'], data['curve'], data['timeindex'], S, piigivej, gcov, n_i, gamma)
    VY = numden[0]
    S_den = numden[1]
    z_int = 1
    diff_inter = 100
    mu_old = mu.copy()
    while diff_inter > par_LQA['eps_LQA'] and z_int <= par_LQA['MAX_iter_LQA']:
        mu_vec = np.ravel(mu.T, order='F')
        diff_mat = np.abs(P_tot @ mu_vec)
        diff_mat[diff_mat < par_LQA['eps_diff']] = par_LQA['eps_diff']
        V_l = np.diag(AW_vec / (2 * diff_mat))
        mu_vec = np.linalg.inv(S_den * (1 / parameters['sigma']) + lambda_s * 2 * W_star + 2 * lambda_l * P_tot.T @ V_l @ P_tot) @ VY * (1 / parameters['sigma'])
        mu = mu_vec.reshape(G, q)
        diff_inter = np.sum(np.abs(mu - mu_old)) / np.sum(np.abs(mu_old))
        mu_old = mu.copy()
        z_int += 1
    sigma = get_sigma(data['x'], data['curve'], data['timeindex'], S, piigivej, gcov, n_i, gamma, mu)
    sigma = np.array(sigma) / n
    parameters['mu'] = mu
    parameters['sigma'] = sigma
    return parameters


def sasfclust_Estep(parameters, data, vars_, S, hard):
    unique_curves = np.unique(data["curve"])
    n_i = [np.sum(data["curve"] == uc) for uc in unique_curves]
    aa = get_Estep(parameters, data, vars_, S, hard, n_i)
    vars_["gamma"] = aa[0]
    vars_["piigivej"] = aa[1]
    vars_["gprod"] = aa[2]
    vars_["gcov"] = aa[3]
    return vars_


def classify(mod, data_new=None):
    parameters = mod["parameters"]
    vars_ = mod["vars"]
    if data_new is None:
        data = mod["data"]
    else:
        data = data_new
    gamma = vars_["gamma"]
    curve = data["curve"]
    pi_vec = parameters["pi"]
    if pi_vec.ndim > 1:
        pi_vec = pi_vec.flatten()
    S = mod["FullS"][data["timeindex"] - 1, : ]
    unique_curves = np.unique(curve)
    N = len(unique_curves)
    G = gamma.shape[1]
    Gamma = parameters["Gamma"]
    po_pr = np.zeros((N, G))
    for i in range(N):
        current_curve = unique_curves[i]
        indices = np.where(curve == current_curve)[0]
        y = data["x"].reshape(-1,1)[indices, :]
        time_indices = data["timeindex"][indices]
        Si = S[time_indices, :]
        ni = Si.shape[0]
        invvar = np.eye(ni) / parameters["sigma"]
        covx = Si @ Gamma @ Si.T + np.linalg.inv(invvar)
        covx_inv = np.linalg.inv(covx)
        covx_det = np.linalg.det(covx)
        temp = np.zeros(G)
        for ll in range(G):
            mu_ll = parameters["mu"].T[:, ll]
            y_vec = y.flatten()
            diff = y_vec - (Si @ mu_ll)
            exponent = -0.5 * (diff @ (covx_inv @ diff.T))
            temp[ll] = pi_vec[ll] * (2 * np.pi)**(-ni/2) * (covx_det**(-0.5)) * np.exp(exponent)
        po_pr[i, :] = temp / np.sum(temp)
    if data_new is None:
        po_pr = vars_["piigivej"]
    classes = np.argmax(po_pr, axis=1)
    return {"classes": classes, "po_pr": po_pr}


def get_msdrule(par, sds, comb_list, m1, m2, m3):
    comb_list = np.asarray(comb_list)
    par = np.asarray(par)
    sds = np.asarray(sds)
    lambda_s_g = np.unique(comb_list[:, 1])
    lambda_l_g = np.unique(comb_list[:, 2])
    
    # First stage
    kk = 0
    max_vec_nc = []
    sd_vec_nc = []
    new_comb_list = np.zeros((len(lambda_s_g) * len(lambda_l_g), 3))
    for lamb_l in lambda_l_g:
        for lamb_s in lambda_s_g:
            indexes = np.where((comb_list[:, 1] == lamb_s) & (comb_list[:, 2] == lamb_l))[0]
            par_index = par[indexes]
            sd_index = sds[indexes]
            max_idx = np.argmax(par_index)
            subset = par_index[:max_idx+1]
            threshold = par_index[max_idx] - m1 * sd_index[max_idx]
            cand = np.where(subset >= threshold)[0]
            onese = cand[0] if cand.size > 0 else max_idx
            max_vec_nc.append(par_index[onese])
            sd_vec_nc.append(sd_index[onese])
            new_comb_list[kk, :] = comb_list[indexes[onese], :]
            kk += 1
    max_vec_nc = np.array(max_vec_nc)
    sd_vec_nc = np.array(sd_vec_nc)
    
    # Second stage
    kk = 0
    max_vec_s = []
    sd_vec_s = []
    new_comb_list2 = np.zeros((len(lambda_l_g), 3))
    for lamb_l in lambda_l_g:
        indexes = np.where(new_comb_list[:, 2] == lamb_l)[0]
        par_index = max_vec_nc[indexes]
        sd_index = sd_vec_nc[indexes]
        max_idx = np.argmax(par_index)
        threshold = par_index[max_idx] - m2 * sd_index[max_idx]
        candidate_idxs = np.where(par_index >= threshold)[0]
        onese = candidate_idxs[-1] if candidate_idxs.size > 0 else max_idx
        max_vec_s.append(par_index[onese])
        sd_vec_s.append(sd_index[onese])
        new_comb_list2[kk, :] = new_comb_list[indexes[onese], :]
        kk += 1
    max_vec_s = np.array(max_vec_s)
    sd_vec_s = np.array(sd_vec_s)
    par_index_final = max_vec_s
    sd_index_final = sd_vec_s
    max_idx_final = np.argmax(par_index_final)
    range_val = abs(np.max(par_index_final) - np.min(par_index_final))
    if m3 * sd_index_final[max_idx_final] > 0.5 * range_val:
        lim = 0.5 * range_val
    else:
        lim = m3 * sd_index_final[max_idx_final]
    candidate_idxs_final = np.where(par_index_final >= par_index_final[max_idx_final] - lim)[0]
    onese_final = candidate_idxs_final[-1] if candidate_idxs_final.size > 0 else max_idx_final
    new_comb_list3 = new_comb_list2[onese_final, :].astype(float)
    num_clusters_opt = new_comb_list3[0]
    lambda_s_opt = new_comb_list3[1]
    lambda_l_opt = new_comb_list3[2]
    return (num_clusters_opt, lambda_s_opt, lambda_l_opt)


def get_zero(mod, mu_fd=None):
    mu = mod["parameters"]["mu"]
    K = mu.shape[0]
    if K != 1:
        FullS = mod["FullS"]
        nP = int((K - 1) * K // 2)
        P = np.zeros((nP, K))
        ind = [0, K - 1 - 1]
        for ii in range(K - 1):
            block_length = ind[1] - ind[0] + 1
            P[ind[0]:ind[1]+1, ii] = 1
            if block_length == 1:
                aa = -1
            else:
                aa = -np.eye(block_length)
            P[ind[0]:ind[1]+1, ii+1:K] = aa
            new_ind0 = ind[0] + (K - 1) - ii
            new_ind1 = ind[1] + (K - 1) - ii
            new_ind0 = min(nP - 1, new_ind0)
            new_ind1 = min(nP - 1, new_ind1)
            ind = [new_ind0, new_ind1]
        M = FullS @ (P @ mu).T
        fraction = np.sum(np.abs(M) < 1e-4) / M.size
        return fraction
    else:
        return np.nan

