from BiFuncLib.sparse_main_func import (
    FKMSparseClustering_permute,
    FKMSparseClustering,
    cer,
)


def sparse_bifunc(data, x, K, method="kmea", true_clus=None):
    # Clustering for single K
    def run_single_k(k):
        # Select optimal sparsity parameter via permutation
        mscelto_res = FKMSparseClustering_permute(data.T, x, k, method=method)
        mscelto = mscelto_res["m"]
        GAP = mscelto_res['GAP']
        # Perform sparse clustering with selected parameter
        result = FKMSparseClustering(data.T, x, k, mscelto, method)
        # Return with CER if true labels provided
        if true_clus is None:
            return {"result": result, "GAP": GAP}
        else:
            CER = cer(true_clus, result["cluster"])
            return {"result": result, "cer": CER, "GAP": GAP}
    # Single K(integer)
    if isinstance(K, int):
        return run_single_k(K)
    # Select K(list) with WCSS
    elif isinstance(K, (list, tuple, range)):
        results = []
        objs = []
        for k in K:
            res = run_single_k(k)
            results.append(res)
            objs.append(res["result"]["obj"])
        best_idx = objs.index(min(objs))
        best_k = K[best_idx]
        best_result = results[best_idx]
        return {
            "best_K": best_k,
            "best_obj": max(objs),
            "best_result": best_result,
            "all_results": results,
            "all_objs": objs
        }
    else:
        raise TypeError("K must be a list or an integer")
