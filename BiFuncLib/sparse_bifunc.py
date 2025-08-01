from BiFuncLib.sparse_main_func import FKMSparseClustering_permute, FKMSparseClustering, cer


def sparse_bifunc(data, x, K, method = 'kmea', true_clus = None):
    mscelto = FKMSparseClustering_permute(data.T, x, K, method=method)['m']
    result = FKMSparseClustering(data.T, x, K, mscelto, method)
    if true_clus is None:
        return result
    else:
        CER = cer(true_clus, result['cluster'])
    return {'result':result,
            'cer':CER}

