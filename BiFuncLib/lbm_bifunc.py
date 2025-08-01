import itertools
import numpy as np

from BiFuncLib.lbm_main_func import lbm_main_func


# Model-based clustering with the funLBM algorithm
def lbm_bifunc(data, K, L, maxit=50, burn=25, basis_name='fourier', nbasis=15,
               gibbs_it=3, display=False, init='funFEM'):
    if not hasattr(K, '__iter__'):
        K = [K]
    if not hasattr(L, '__iter__'):
        L = [L]
    
    # With grid search
    if (len(K) > 1 or len(L) > 1):   
        models = list(itertools.product(K, L))
        RES = []
        outNA = {"K": None, "icl": None}
        for model in models:
            try:
                res_temp = lbm_main_func(data=data, K=model[0], L=model[1], maxit=maxit, 
                                         burn=burn, basis_name=basis_name, nbasis=nbasis,
                                         gibbs_it=gibbs_it, display=False, init=init)
            except Exception:
                res_temp = outNA
            RES.append(res_temp)
        models_with_icl = []
        for i, model in enumerate(models):
            try:
                icl_value = RES[i]["icl"]
            except Exception:
                icl_value = None
            models_with_icl.append((model[0], model[1], icl_value))
        best_model_index = None
        best_icl = -np.inf
        for i, (_k, _l, icl_value) in enumerate(models_with_icl):
            if icl_value is not None and icl_value > best_icl:
                best_icl = icl_value
                best_model_index = i
        if best_model_index is None:
            raise Exception("No models converge")
        out = RES[best_model_index]
        models_with_icl.sort(key=lambda item: item[2] if item[2] is not None else -np.inf, reverse=True)
        if display:
            print("Models sorted by icl (desc):")
            for mod in models_with_icl:
                print("K =", mod[0], "L =", mod[1], "icl =", mod[2])
        out["allRes"] = RES
        out["criteria"] = models_with_icl
    
    # Without grid search
    else:
        out = lbm_main_func(data=data, K=K[0], L=L[0], maxit=maxit,
                            burn=burn, basis_name=basis_name, nbasis=nbasis,
                            gibbs_it=gibbs_it, display=display, init=init)
    if not isinstance(data, list):
        out['datatype'] = 0
    else:
        out['datatype'] = len(data) 
    return out

