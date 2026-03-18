from BiFuncLib.sparse_main_func import (
    FKMSparseClustering_permute,
    FKMSparseClustering,
    cer,
)


def sparse_bifunc(data, x, K, method="kmea", true_clus=None):
    # Select optimal sparsity parameter via permutation
    mscelto = FKMSparseClustering_permute(data.T, x, K, method=method)["m"]
    # Perform sparse clustering with selected parameter
    result = FKMSparseClustering(data.T, x, K, mscelto, method)
    # Return with CER if true labels provided
    if true_clus is None:
        return result
    else:
        CER = cer(true_clus, result["cluster"])
    return {"result": result, "cer": CER}
