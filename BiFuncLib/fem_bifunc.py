import numpy as np
import warnings

from BiFuncLib.fem_main_func import fem_main_func
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Model-based clustering with the funFEM algorithm
def fem_bifunc(fd, K = np.arange(2, 7), model = ['AkjBk'], crit = 'bic', init = 'kmeans',
           Tinit = (), maxit = 50, eps = 1e-6, disp = False, lambda_ = 0, graph = False):
    resultat = [None] * (len(K) * len(model))
    bic = [None] * (len(K) * len(model))
    aic = [None] * (len(K) * len(model))
    icl = [None] * (len(K) * len(model))
    nbprm = [None] * (len(K) * len(model))
    ll = [None] * (len(K) * len(model))
    it = 0
    for k_idx in range(len(K)):
        current_K = K[k_idx]
        if disp:
            print(f'>> K = {current_K}')
        for model_idx in range(len(model)):
            current_model = model[model_idx]
            resultat[it] = fem_main_func(fd, current_K, init=init, maxit=maxit, eps=eps, Tinit=Tinit, model=current_model, lambda_=lambda_, graph=graph)
            if resultat[it] is not None:
                bic[it] = resultat[it]['bic']
                aic[it] = resultat[it]['aic']
                icl[it] = resultat[it]['icl']
                nbprm[it] = resultat[it]['nbprm']
                ll[it] = resultat[it]['ll']
                if disp:
                    if crit == 'bic':
                        print(f"{current_model}\t:\t bic = {resultat[it]['bic']}")
                    elif crit == 'aic':
                        print(f"{current_model}\t:\t aic = {resultat[it]['aic']}")
                    elif crit == 'icl':
                        print(f"{current_model}\t:\t icl = {resultat[it]['icl']}")
            it += 1
    if all(v is None for v in bic):
        raise ValueError("No reliable results to return (empty clusters in all partitions)!")
    if crit == 'bic':
        id_max = bic.index(max([v for v in bic if v is not None]))
        crit_max = resultat[id_max]['bic']
    elif crit == 'aic':
        id_max = aic.index(max([v for v in aic if v is not None]))
        crit_max = resultat[id_max]['aic']
    elif crit == 'icl':
        id_max = icl.index(max([v for v in icl if v is not None]))
        crit_max = resultat[id_max]['icl']
    res = resultat[id_max]
    if disp:
        print(f"The best model is {res['model']} with K = {res['K']} ({crit} = {crit_max})")
    res['crit'] = crit
    nm = len(model)
    res['allCriterions'] = {
        'K': K,
        'model': model,
        'bic': [bic[i::nm] for i in range(nm)],
        'aic': [aic[i::nm] for i in range(nm)],
        'icl': [icl[i::nm] for i in range(nm)],
        'nbprm': [nbprm[i::nm] for i in range(nm)],
        'll': [ll[i::nm] for i in range(nm)]
    }
    return res

