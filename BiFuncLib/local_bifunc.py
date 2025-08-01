import numpy as np

from BiFuncLib.local_main_func import calculate_gcv, calculate_bic, local_admm


def local_bifunc(data, times, lambda1, lambda2, lambda3, opt = False, rangeval = (0, 1), nknots = 30, order = 4,
                 nu = 2, tau = 3, K0 = 6, rep_num = 100, kappa = 1, eps_outer = 0.0001, max_iter = 100):
    
    # Data process
    oridata = [data[column].to_numpy() for column in data.columns]
    oritimes = [times] * len(oridata)
    
    # Generate result without selection
    if opt == False:
        res = local_admm(oridata, oritimes, lambda1, lambda2, lambda3, rangeval, nknots, 
                         order, nu, tau, K0, rep_num, kappa, eps_outer, max_iter)
        return res
    
    # Generate result with selection
    elif opt == True:
        lambda1_opt = calculate_gcv(oridata, oritimes, lambda1, rangeval, nknots, order, nu)
        bic_res = calculate_bic(oridata, oritimes, lambda1_opt, lambda2, lambda3, rangeval, 
                                nknots, order, nu, tau, K0, rep_num, kappa, eps_outer, max_iter)
        nlambda2 = len(lambda2)
        nlambda3 = len(lambda3)
        bic_mat = np.zeros((nlambda2, nlambda3))
        for l1 in range(nlambda2):
            for l2 in range(nlambda3):
                bic_mat[l1, l2] = bic_res[l1][l2]['bic_val']
        ids = np.unravel_index(np.argmin(bic_mat, axis=None), bic_mat.shape)
        lambda2_opt = lambda2[ids[0]]
        lambda3_opt = lambda3[ids[1]]
        res_opt = local_admm(oridata, oritimes, lambda1_opt, lambda2_opt, lambda3_opt, rangeval, 
                             nknots, order, nu, tau, K0, rep_num, kappa, eps_outer, max_iter) 
        res_opt.update({'lambda1_opt': lambda1_opt, 
                        'lambda2_opt': lambda2_opt,
                        'lambda3_opt': lambda3_opt})
        return res_opt
        
    else:
        print('Please enter True or False')

