from BiFuncLib.ssvd_main_func import ssvd_bc, s4vd


def s4vd_biclus(data, steps=100, pcerv=0.1, pceru=0.1, ss_thr=(0.6, 0.65), size=0.5,
                gamm=0, iters=100, nbiclust=10, merr=1e-3, cols_nc=True, rows_nc=True,
                row_overlap=True, col_overlap=True, row_min=1, col_min=1, pointwise=True,
                start_iter=3, savepath=False):
    res = s4vd(data, steps=steps, pcerv=pcerv, pceru=pceru, ss_thr=ss_thr, size=size,
               gamm=gamm, iters=iters, nbiclust=nbiclust, merr=merr, cols_nc=cols_nc,
               rows_nc=rows_nc, row_overlap=row_overlap, col_overlap=col_overlap, row_min=row_min,
               col_min=col_min, pointwise=pointwise, start_iter=start_iter, savepath=savepath)
    return res
        
    
def ssvd_biclus(data, K=10, threu=1, threv=1, gamu=0, gamv=0, merr=1e-4, niter=100):
    res = ssvd_bc(data, K=K, threu=threu, threv=threv, gamu=gamu, gamv=gamv, merr=merr, niter=niter)
    return res


