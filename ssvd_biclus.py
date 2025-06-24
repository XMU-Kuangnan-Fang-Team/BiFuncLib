from ssvd_main_func import ssvd_bc, s4vd, bcheatmap, jaccardmat


def s4vd_biclus(data, plot = True, steps=100, pcerv=0.1, pceru=0.1, ss_thr=(0.6, 0.65), size=0.5,
                gamm=0, iters=100, nbiclust=10, merr=1e-3, cols_nc=True, rows_nc=True,
                row_overlap=True, col_overlap=True, row_min=1, col_min=1, pointwise=True,
                start_iter=3, savepath=False):
    res = s4vd(data, steps=steps, pcerv=pcerv, pceru=pceru, ss_thr=ss_thr, size=size,
               gamm=gamm, iters=iters, nbiclust=nbiclust, merr=merr, cols_nc=cols_nc,
               rows_nc=rows_nc, row_overlap=row_overlap, col_overlap=col_overlap, row_min=row_min,
               col_min=col_min, pointwise=pointwise, start_iter=start_iter, savepath=savepath)
    if plot:
        bcheatmap(data, res)
    return res
        
    
def ssvd_biclus(data, plot = True, K=10, threu=1, threv=1, gamu=0, gamv=0, merr=1e-4, niter=100):
    res = ssvd_bc(data, K=K, threu=threu, threv=threv, gamu=gamu, gamv=gamv, merr=merr, niter=niter)
    if plot:
        bcheatmap(data, res)
    return res


def jaccardind(res1, res2):
    return jaccardmat(res1, res2)



'''
# EG1
from simulation_data import ssvd_sim_data
lung200 = ssvd_sim_data()
res1 = s4vd_biclus(lung200, pcerv=0.5, pceru=0.01, ss_thr=(0.6, 0.65),
                   start_iter=3, size=0.632, cols_nc=True, steps=100, pointwise=True,
                   merr=1e-4, iters=100, nbiclust=10, col_overlap=False)
'''
'''
# EG2
import numpy as np
from ssvd_main_func import BiclustResult
u = np.array([10,9,8,7,6,5,4,3] + [2]*17 + [0]*75)
v = np.array([10,-10,8,-8,5,-5] + [3]*5 + [-3]*5 + [0]*34)
u = u/np.sqrt(np.sum(u**2))
v = v/np.sqrt(np.sum(v**2))
d = 50
np.random.seed(123)
X_art = (d * np.outer(u, v)) + np.random.randn(100, 50)
params = {}
info = []
RowxNumber = np.zeros((100,1), dtype=bool)
NumberxCol = np.zeros((50,1), dtype=bool)
RowxNumber[u != 0, 0] = True
NumberxCol[v != 0, 0] = True
Number = 1
res2_sim = BiclustResult(params, RowxNumber, NumberxCol, Number, info)
res2 = s4vd_biclus(X_art, pcerv=0.5, pceru=0.5, pointwise=False, nbiclust=1)
res2_pw = s4vd_biclus(X_art, pcerv=0.5, pceru=0.5, pointwise=True, nbiclust=1)
res2_ssvd = ssvd_biclus(X_art,K=1)
bcheatmap(X_art, res2)
bcheatmap(X_art, res2_pw)
bcheatmap(X_art, res2_ssvd)
print(jaccardind(res2_sim, res2))
print(jaccardind(res2_sim, res2_pw))
print(jaccardind(res2_sim, res2_ssvd))
'''

