import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn_extra.cluster import KMedoids


# Classification error rate function
def cer(P, Q):
    if len(P) != len(Q):
        raise ValueError("The two partitions must have the same length")
    cer_comp = 0
    for i in range(len(P) - 1):
        for j in range(i + 1, len(P)):
            cer_comp += abs((P[i] == P[j]) ^ (Q[i] == Q[j]))
    cer_comp /= (len(P) * (len(P) - 1)) / 2
    return cer_comp


def GetWCSS(x, Cs, ws=None):
    n_features = x.shape[1]
    wcss_perfeature = np.zeros(n_features)
    for k in np.unique(Cs):
        whichers = (Cs == k)
        if np.sum(whichers) > 1:
            subset = x[whichers, :]
            centered = subset - np.mean(subset, axis=0)
            wcss_perfeature += np.sum(centered**2, axis=0)
    total_ss = np.sum((x - np.mean(x, axis=0))**2, axis=0)
    bcss_perfeature = total_ss - wcss_perfeature
    result = {
        'wcss.perfeature': wcss_perfeature,
        'wcss': np.sum(wcss_perfeature),
        'bcss.perfeature': bcss_perfeature
    }
    if ws is not None:
        result['wcss.ws'] = np.sum(wcss_perfeature * ws)
    return result


def GetOptimalW(b, c_star):
    b_star = np.array(b, copy=True)
    b_star[b <= c_star] = 0
    norm_b_star = np.linalg.norm(b_star)
    if norm_b_star == 0:
        raise ValueError("Norm of b_star is zero; cannot compute optimal w.")
    w = b_star / norm_b_star
    return w


def GetOptimalClusters(data, K, w, method='kmea'):
    weighted_data = data * w
    if method == 'kmea':
        kmeans = KMeans(n_clusters=K)
        clusters = kmeans.fit_predict(weighted_data)
    elif method == 'pam':
        kmedoids = KMedoids(n_clusters=K, random_state=0)
        clusters = kmedoids.fit_predict(weighted_data)
    elif method == 'hier':
        Z = linkage(weighted_data, method='ward')
        clusters = fcluster(Z, t=K, criterion='maxclust')
        clusters = clusters - 1
    else:
        raise ValueError("Unknown method. Choose one of 'kmea', 'pam', or 'hier'.")    
    return clusters


def FKMSparseClustering(data, x, K, m, method='kmea', maxiter=50):
    mu = x[-1] - x[0]
    if m > mu:
        raise ValueError("m has to be less than the measure of the domain")
    
    # Initial clustering based on the chosen method
    if method == 'kmea':
        initial_clusters = KMeans(n_clusters=K).fit_predict(data)
    elif method == 'pam':
        initial_clusters = KMedoids(n_clusters=K, random_state=0).fit_predict(data)
    elif method == 'hier':
        Z = linkage(data, method='ward')
        initial_clusters = fcluster(Z, t=K, criterion='maxclust') - 1
    else:
        raise ValueError("Unknown method. Choose one of 'kmea', 'pam', or 'hier'.")
    b_old = GetWCSS(data, initial_clusters)['bcss.perfeature']
    perc = m / mu
    b_ord = np.sort(b_old)
    index = math.ceil(len(b_ord) * perc) - 1
    c_star = b_ord[index]
    niter = 1
    w = np.zeros(len(x))
    
    # Initialize k with zeros (same length as number of samples)
    k = np.zeros(len(initial_clusters), dtype=int)
    b = np.zeros(len(x))
    cluster_difference = np.sum(np.abs(initial_clusters - k))
    epsilon = 1e-6
    w_old = np.ones(len(x))
    
    # Iterative updates
    while (np.linalg.norm(w - w_old) >= epsilon and cluster_difference > 0 and niter < maxiter):
        niter += 1
        w_old = w.copy()
        k_old = k.copy()
        w = GetOptimalW(b_old, c_star)
        k = GetOptimalClusters(data, K, w, method)
        b = GetWCSS(data, k)['bcss.perfeature']
        b_old = b.copy()
        b_ord = np.sort(b_old)
        index = math.ceil(len(b_ord) * perc) - 1
        c_star = b_ord[index]
        cluster_difference = np.sum(np.abs(k_old - k))
    obj = np.sum(w * b)
    return {"w": w, "cluster": k, "obj": obj, "iteration": niter}


def FKMSparseClustering_permute(data, x, K, mbound=None, method='kmea', nperm=20, maxiter=50):
    mu = x[-1] - x[0]
    n, p = data.shape
    if mbound is not None:
        if mbound > mu:
            raise ValueError("m has to be less than the measure of the domain")
    else:
        mbound = 0.5 * mu
    num_points = max(1, int(round(0.1 * len(x))))
    qualim = np.linspace(2 * np.min(np.diff(x)), mbound, num=num_points)
    GAP = np.zeros(len(qualim))
    for i, m in enumerate(qualim):
        resTRUE = FKMSparseClustering(data, x, K, m, method=method, maxiter=maxiter)['obj']
        resPERM = []
        for _ in range(nperm):
            dataperm = data.copy()
            for j in range(p):
                perm_indices = np.random.permutation(n)
                dataperm[:, j] = data[perm_indices, j]
            res_perm = FKMSparseClustering(dataperm, x, K, m, method=method, maxiter=maxiter)['obj']
            resPERM.append(res_perm)
        GAP[i] = np.log(resTRUE) - np.mean(np.log(resPERM))
    max_gap = np.max(GAP)
    best_index = np.argmax(GAP)
    best_m = qualim[best_index]
    return {"GAP": max_gap, "m": best_m}

