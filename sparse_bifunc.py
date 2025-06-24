from sparse_main_func import FKMSparseClustering_permute, FKMSparseClustering, cer


def sparse_bifunc(data, x, K, method = 'kmea', true_clus = None):
    mscelto = FKMSparseClustering_permute(data.T, x, K, method=method)['m']
    result = FKMSparseClustering(data.T, x, K, mscelto, method)
    if true_clus is None:
        return result
    else:
        CER = cer(true_clus, result['cluster'])
    return {'result':result,
            'cer':CER}


'''
from simulation_data import sparse_sim_data
method = "kmea"
K = 2
B = 10
paramC = 2
cer_std = np.zeros(B)
cer_MY = np.zeros(B)
n = 100
x = np.linspace(0, 1, 1000)
data = sparse_sim_data(n, x, paramC)['data']
part_vera = sparse_sim_data(n, x, paramC)['cluster']
for b in range(B):
    mscelto = FKMSparseClustering_permute(data.T, x, K, method=method)['m']
    result = FKMSparseClustering(data.T, x, K, mscelto, method)
    if method == "kmea":
        result2 = KMeans(n_clusters=K).fit_predict(data.T)
    elif method == "pam":
        result2 = KMedoids(n_clusters=K, random_state=0).fit_predict(data.T)
    elif method == "hier":
        result2 = fcluster(linkage(data.T, method='ward'), t=K, criterion='maxclust') - 1
    cer_MY[b] = cer(part_vera, result['cluster'])
    cer_std[b] = cer(part_vera, result2)
'''
