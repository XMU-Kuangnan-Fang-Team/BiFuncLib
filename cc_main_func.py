import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform


def medoid_evaluation(fun_mat, a, b, const_a, const_b):
    n, m, p = fun_mat.shape
    fun_per_medoid = fun_mat.reshape(n * m, p)
    distance = squareform(pdist(fun_per_medoid, metric='euclidean'))
    distance = distance ** 2
    np.fill_diagonal(distance, np.nan)
    sum_dist = np.nanmean(distance, axis=0)
    rep_curve = np.nanargmin(sum_dist)
    medoid_fun = fun_per_medoid[rep_curve]
    new_fun = np.tile(medoid_fun, (n, m, 1))
    return new_fun


def medoid_evaluation_add(fun_mat, logr, logc, a, b, const_a, const_b):
    n, m, p = fun_mat.shape
    filtered_fun_mat = fun_mat[logr, :, :][:, logc, :]
    n_filtered, m_filtered, _ = filtered_fun_mat.shape
    fun_per_medoid = filtered_fun_mat.reshape(n_filtered * m_filtered, p)
    distance = squareform(pdist(fun_per_medoid, metric='euclidean'))
    distance = distance ** 2
    np.fill_diagonal(distance, np.nan)
    sum_dist = np.nanmean(distance, axis=0)
    rep_curve = np.nanargmin(sum_dist)
    medoid_fun = fun_per_medoid[rep_curve]
    new_fun = np.tile(medoid_fun, (n, m, 1))
    return new_fun


def template_evaluation(fun_mat, a, b, const_a, const_b):
    n, m, p = fun_mat.shape
    count_null = np.sum(np.isnan(fun_mat), axis=2)
    not_null = count_null < p
    fun_mean = np.nanmean(fun_mat, axis=(0, 1))
    alpha_fun = np.zeros((n, p))
    for j in range(n):
        row_data = fun_mat[j, :, :]
        alpha_fun[j] = np.nansum(row_data, axis=0) / np.sum(not_null[j, :], axis=0)
    alpha_fun = alpha_fun - fun_mean
    if const_a:
        alpha_fun = np.tile(np.nanmean(alpha_fun, axis=1, keepdims=True), (1, p))
    beta_fun = np.zeros((m, p))
    for j in range(m):
        col_data = fun_mat[:, j, :]
        beta_fun[j] = np.nansum(col_data, axis=0) / np.sum(not_null[:, j], axis=0)
    beta_fun = beta_fun - fun_mean
    if const_b:
        beta_fun = np.tile(np.nanmean(beta_fun, axis=1, keepdims=True), (1, p))
    fun_mean_mat = np.tile(fun_mean, (n, m, 1))
    beta_fun_mat = np.tile(beta_fun, (n, 1, 1))
    beta_fun_mat = np.where(np.isnan(beta_fun_mat), 0, beta_fun_mat)
    alpha_fun_mat = np.tile(alpha_fun[:, np.newaxis, :], (1, m, 1))
    alpha_fun_mat = np.where(np.isnan(alpha_fun_mat), 0, alpha_fun_mat)
    new_fun = fun_mean_mat + b * beta_fun_mat + a * alpha_fun_mat
    return new_fun


def template_evaluation_add(fun_mat, logr, logc, a, b, const_a, const_b):
    n, m, p = fun_mat.shape
    count_null = np.sum(np.isnan(fun_mat), axis=2)
    not_null = count_null < p
    filtered_fun_mat = fun_mat[logr, :, :][:, logc, :]
    fun_mean = np.nanmean(filtered_fun_mat, axis=(0, 1))
    alpha_fun = np.zeros((n, p))
    for j in range(n):
        row_data = fun_mat[j, logc, :]
        alpha_fun[j] = np.nansum(row_data, axis=0) / np.sum(not_null[j, logc], axis=0)
    alpha_fun -= fun_mean
    if const_a:
        alpha_fun = np.tile(np.nanmean(alpha_fun, axis=1, keepdims=True), (1, p))
    alpha_fun_mat = np.tile(alpha_fun[:, np.newaxis, :], (1, m, 1))
    alpha_fun_mat = np.where(np.isnan(alpha_fun_mat), 0, alpha_fun_mat)
    beta_fun = np.zeros((m, p))
    for j in range(m):
        col_data = fun_mat[logr, j, :]
        beta_fun[j] = np.nansum(col_data, axis=0) / np.sum(not_null[logr, j], axis=0)
    beta_fun -= fun_mean
    if const_b:
        beta_fun = np.tile(np.nanmean(beta_fun, axis=1, keepdims=True), (1, p))
    beta_fun_mat = np.tile(beta_fun[np.newaxis, :, :], (n, 1, 1))
    beta_fun_mat = np.where(np.isnan(beta_fun_mat), 0, beta_fun_mat)
    fun_mean_mat = np.tile(fun_mean, (n, m, 1))
    new_fun = fun_mean_mat + b * beta_fun_mat + a * alpha_fun_mat
    return new_fun


def warping_function(fun_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    def warping_shift(coeff):
        st = x_reg + coeff
        template_t = new_fun[i, j, :]
        b_mask = ~np.isnan(template_t)
        data_t = interp1d(st, fun_mat[i, j, :], fill_value="extrapolate")(x_out)
        a_mask = ~np.isnan(data_t)
        sel = a_mask & b_mask
        data_t = data_t[sel]
        template_t = new_fun[i, j, sel]
        distance = np.sum((data_t - template_t) ** 2) / np.sum(sel)
        return distance
    n, m, p = fun_mat.shape
    x = np.arange(1, p + 1)
    x_reg = x
    x_out = np.linspace(np.min(x), np.max(x), p)
    min_temp = np.ptp(x)
    lower_warp = -shift_max * min_temp
    upper_warp = shift_max * min_temp
    dist_mat = np.full((n, m), 10.0)
    align_mat = np.zeros((n, m))
    count_null = np.sum(np.isnan(fun_mat), axis=2)
    not_null = count_null < p
    fun_mat_align = fun_mat.copy()
    iter_count = 0
    conv = False
    while iter_count < max_iter and not conv:
        iter_count += 1
        dist_mat_old = dist_mat.copy()
        if template_type == 'mean':
            new_fun = template_evaluation(fun_mat_align, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation(fun_mat_align, a, b, const_a, const_b)
        if shift_alignment:
            for i in range(n):
                for j in range(m):
                    if not_null[i, j]:
                        result = minimize_scalar(warping_shift, bounds=(lower_warp, upper_warp), method='bounded')
                        dist_mat[i, j] = result.fun
                        align_mat[i, j] = result.x
                        new_x = x + align_mat[i, j]
                        fun_mat_align[i, j, :] = interp1d(new_x, fun_mat[i, j, :], fill_value="extrapolate")(x_out)
                    if not not_null[i, j]:
                        dist_mat[i, j] = np.nan
                        align_mat[i, j] = np.nan
                        fun_mat_align[i, j, :] = np.full(p, np.nan)
        if np.nansum(dist_mat_old) <= np.nansum(dist_mat):
            conv = True
            iter_count -= 1
    return dist_mat_old


def warping_function_add(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    def warping_shift(coeff):
        st = x_reg + coeff
        template_t = new_fun[i, j, :]
        b_mask = ~np.isnan(template_t)
        data_t = interp1d(st, fun_mat[i, j, :], fill_value="extrapolate")(x_out)
        a_mask = ~np.isnan(data_t)
        sel = a_mask & b_mask
        data_t = data_t[sel]
        template_t = new_fun[i, j, sel]
        distance = np.sum((data_t - template_t) ** 2) / np.sum(sel)
        return distance
    n, m, p = fun_mat.shape
    x = np.arange(1, p + 1)
    x_reg = x
    x_out = np.linspace(np.min(x), np.max(x), p)
    min_temp = np.ptp(x)  # Equivalent to diff(range(x))
    lower_warp = -shift_max * min_temp
    upper_warp = shift_max * min_temp
    dist_mat = np.full((n, m), 10.0)
    align_mat = np.zeros((n, m))
    count_null = np.sum(np.isnan(fun_mat), axis=2)
    not_null = count_null < p
    fun_mat_align = fun_mat.copy()
    iter_count = 0
    conv = False
    while iter_count < max_iter and not conv:
        iter_count += 1
        dist_mat_old = dist_mat.copy()
        if template_type == 'mean':
            new_fun = template_evaluation_add(fun_mat_align, logr, logc, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation_add(fun_mat_align, logr, logc, a, b, const_a, const_b)
        if shift_alignment:
            for i in range(n):
                for j in range(m):
                    if not_null[i, j]:
                        result = minimize_scalar(warping_shift, bounds=(lower_warp, upper_warp), method='bounded')
                        dist_mat[i, j] = result.fun
                        align_mat[i, j] = result.x
                        new_x = x + align_mat[i, j]
                        fun_mat_align[i, j, :] = interp1d(new_x, fun_mat[i, j, :], fill_value="extrapolate")(x_out)
                    if not not_null[i, j]:
                        dist_mat[i, j] = np.nan
                        align_mat[i, j] = np.nan
                        fun_mat_align[i, j, :] = np.full(p, np.nan)
        if np.nansum(dist_mat_old) <= np.nansum(dist_mat):
            conv = True
            iter_count -= 1
    return dist_mat_old


def evaluate_mat_dist(fun_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    n, m, p = fun_mat.shape
    if shift_alignment:
        mat_dist = warping_function(fun_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
    else:
        if template_type == 'mean':
            new_fun = template_evaluation(fun_mat, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation(fun_mat, a, b, const_a, const_b)
        else:
            raise ValueError("Unsupported template type")
        mat_dist = np.sum((fun_mat - new_fun) ** 2, axis = 2) / p
    return mat_dist


def evaluate_mat_dist_add(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    n, m, p = fun_mat.shape
    if shift_alignment:
        mat_dist = warping_function_add(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
    else:
        if template_type == 'mean':
            new_fun = template_evaluation_add(fun_mat, logr, logc, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation_add(fun_mat, logr, logc, a, b, const_a, const_b)
        else:
            raise ValueError("Unsupported template type")
        mat_dist = np.sum((fun_mat - new_fun) ** 2, axis=2) / p
    return mat_dist


def ccscore_fun(mat_dist):
    score_fun = np.nanmean(mat_dist)
    return score_fun


def rowscore_fun(mat_dist):
    score_fun = np.nanmean(mat_dist, axis=1)
    return score_fun


def colscore_fun(mat_dist):
    score_fun = np.nanmean(mat_dist, axis=0)
    return score_fun


def addrowscore_fun(mat_dist, logc):
    selected_cols = mat_dist[:, logc]
    score_fun = np.nanmean(selected_cols, axis=1)
    return score_fun


def addcolscore_fun(mat_dist, logr):
    selected_rows = mat_dist[logr, :]
    score_fun = np.nanmean(selected_rows, axis=0)
    return score_fun


def cc1_fun(fun_mat, logr, logc, delta, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter, only_one):
    logr = logr.copy()
    logc = logc.copy()
    sub_mat = fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])]
    dim = (np.sum(logr), np.sum(logc), fun_mat.shape[2])
    sub_mat = np.reshape(sub_mat, dim)
    dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
    score_while = ccscore_fun(dist_mat)
    while score_while > delta:
        logr[logr] = np.sum(np.isnan(fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])][:, :, 0]), axis=1) != np.sum(logc)
        logc[logc] = np.sum(np.isnan(fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])][:, :, 0]), axis=0) != np.sum(logr)
        di = rowscore_fun(dist_mat)
        dj = colscore_fun(dist_mat)
        mdi = np.argmax(di)
        mdj = np.argmax(dj)
        if di[mdi] > dj[mdj]:
            current_logr = np.where(logr)[0]
            logr[current_logr[mdi]] = False
        else:
            current_logc = np.where(logc)[0]
            logc[current_logc[mdj]] = False
        logr[logr] = np.sum(np.isnan(fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])][:, :, 0]), axis=1) != np.sum(logc)
        logc[logc] = np.sum(np.isnan(fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])][:, :, 0]), axis=0) != np.sum(logr)
        if only_one == 'False' and not (np.sum(logr) > 1 and np.sum(logc) > 1):
            break
        if only_one == 'True' and not ((np.sum(logr) >= 1 and np.sum(logc) > 1) or (np.sum(logr) > 1 and np.sum(logc) >= 1)):
            break
        if only_one == 'True_alpha' and not (np.sum(logr) >= 1 and np.sum(logc) > 1):
            break
        if only_one == 'True_beta' and not (np.sum(logr) > 1 and np.sum(logc) >= 1):
            break
        sub_mat = fun_mat[np.ix_(logr, logc, [True]*fun_mat.shape[2])]
        dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        score_while = ccscore_fun(dist_mat)
    ret = [0, f'No submatrix with score smaller than {delta} found']
    if only_one == 'False':
        if np.sum(logr) > 1 and np.sum(logc) > 1:
            ret = [logr, logc]
    elif only_one == 'True':
        if (np.sum(logr) >= 1 and np.sum(logc) > 1) or (np.sum(logr) > 1 and np.sum(logc) >= 1):
            ret = [logr, logc]
    elif only_one == 'True_alpha':
        if np.sum(logr) >= 1 and np.sum(logc) > 1:
            ret = [logr, logc]
    elif only_one == 'True_beta':
        if np.sum(logr) > 1 and np.sum(logc) >= 1:
            ret = [logr, logc]
    return ret


def cc2_fun(fun_mat, logr, logc, delta, theta, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    logr = logr.copy()
    logc = logc.copy()
    mdi = 1
    mdj = 1
    sub_mat = fun_mat[np.ix_(logr, logc, [True] * fun_mat.shape[2])]
    dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
    h = ccscore_fun(dist_mat)
    while h > delta and (np.nansum(mdi) + np.nansum(mdj)) > 0:
        mdi = 0
        mdj = 0
        if np.sum(logr) > 100:
            di = rowscore_fun(dist_mat)
            mdi = di > (theta * h)
            valid_logr = logr.copy()
            valid_logr[logr] = ~np.isnan(di)
            valid_count = np.sum(valid_logr)
            if np.sum(mdi) < (valid_count - 1):
                current_logr = np.where(logr)[0]
                logr[current_logr[mdi]] = False
                sub_mat = fun_mat[np.ix_(logr, logc, [True] * fun_mat.shape[2])]
                dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
                h = ccscore_fun(dist_mat)
            else:
                print(f"Theta {theta} too small")
                mdi = 0
        if np.sum(logc) > 100:
            dj = colscore_fun(dist_mat)
            mdj = dj > (theta * h)
            valid_logc = logc.copy()
            valid_logc[logc] = ~np.isnan(dj)
            valid_count = np.sum(valid_logc)
            if np.sum(mdj) < (valid_count - 1):
                current_logc = np.where(logc)[0]
                logc[current_logc[mdj]] = False
                sub_mat = fun_mat[np.ix_(logr, logc, [True] * fun_mat.shape[2])]
                dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
                h = ccscore_fun(dist_mat)
            else:
                print(f"Theta {theta} too small")
                mdj = 0
        h = ccscore_fun(dist_mat)
    return [logr, logc]


def cc3_fun(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    logr = logr.copy()
    logc = logc.copy()
    br = 1
    while br > 0:
        sub_mat = fun_mat[np.ix_(logr, logc, [True] * fun_mat.shape[2])]
        dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        h = ccscore_fun(dist_mat)
        br1 = np.sum(logc)
        br2 = np.sum(logr)
        dist_mat_add = evaluate_mat_dist_add(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        dj = addcolscore_fun(dist_mat_add, logr)
        mdj = dj <= h
        logc[mdj] = True
        sub_mat = fun_mat[np.ix_(logr, logc, [True] * fun_mat.shape[2])]
        dist_mat = evaluate_mat_dist(sub_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        h = ccscore_fun(dist_mat)
        dist_mat_add = evaluate_mat_dist_add(fun_mat, logr, logc, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        di = addrowscore_fun(dist_mat_add, logc)
        mdi = di <= h
        logr[mdi] = True
        br = np.sum(logc) + np.sum(logr) - br1 - br2
    return [logr, logc]


def bigcc_fun(fun_mat, delta, theta, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter, only_one):
    n, m, p = fun_mat.shape
    logr = np.ones(n, dtype=bool)
    logc = np.ones(m, dtype=bool)
    for i in range(n):
        if np.all(np.isnan(fun_mat[i, :, 0])):
            logr[i] = False
    for j in range(m):
        if np.all(np.isnan(fun_mat[:, j, 0])):
            logc[j] = False
    step1 = cc2_fun(fun_mat, logr, logc, delta, theta, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
    step2 = cc1_fun(fun_mat, step1[0], step1[1], delta, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter, only_one)
    if np.sum(step2[0]) == 0:
        print(f"No fun_matrix with score smaller than {delta} found")
        return [0]
    else:
        ret = cc3_fun(fun_mat, step2[0], step2[1], template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter)
        return ret
    

def warping_function_plot(res, fun_mat, template_type, a, b, const_a, const_b, shift_alignment, shift_max, max_iter):
    
    # Warping shift function
    def warping_shift(coeff):
        st = x_reg + coeff
        template_t = new_fun[i, j, :]
        b_mask = ~np.isnan(template_t)
        data_t = interp1d(st, fun_mat[i, j, :], bounds_error=False, fill_value="extrapolate")(x_out)
        a_mask = ~np.isnan(data_t)
        sel = a_mask & b_mask
        data_t = data_t[sel]
        template_t = template_t[sel]
        distance = np.sum((data_t - template_t) ** 2) / np.sum(sel)
        return distance

    # Define lower and upper warp
    n, m, p = fun_mat.shape
    x = np.linspace(1, p, num=p)
    x_reg = x
    x_out = np.linspace(np.min(x), np.max(x), num=p)
    min_temp = np.ptp(x)
    lower_warp = -shift_max * min_temp
    upper_warp = shift_max * min_temp
    dist_mat = np.full((n, m), 10)
    align_mat = np.zeros((n, m))
    count_null = np.apply_along_axis(lambda arr: np.sum(np.isnan(arr)), 2, fun_mat)
    not_null = count_null < p
    fun_mat_align = np.copy(fun_mat)
    iter_count = 0
    conv = False

    # Iterations for alignment until max_iter or convergence
    while iter_count < max_iter and not conv:
        iter_count += 1
        dist_mat_old = np.copy(dist_mat)
        fun_mat_align_old = np.copy(fun_mat_align)
        if template_type == 'mean':
            new_fun = template_evaluation(fun_mat_align, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation(fun_mat_align, a, b, const_a, const_b)

        if shift_alignment:
            for i in range(n):
                for j in range(m):
                    if not_null[i, j]:
                        result = minimize_scalar(warping_shift, method='bounded', bounds=(lower_warp, upper_warp))
                        dist_mat[i, j] = result.fun
                        align_mat[i, j] = result.x
                        new_x = np.linspace(1, p, num=p) + align_mat[i, j]
                        fun_mat_align[i, j, :] = interp1d(new_x, fun_mat[i, j, :], bounds_error=False, fill_value="extrapolate")(x_out)
                    else:
                        dist_mat[i, j] = np.nan
                        align_mat[i, j] = np.nan
                        fun_mat_align[i, j, :] = np.full(p, np.nan)
        if np.nansum(dist_mat_old) <= np.nansum(dist_mat):
            conv = True
            iter_count -= 1
            fun_mat_align = np.copy(fun_mat_align_old)

    # Final templates
    if template_type == 'mean':
        new_fun = template_evaluation(fun_mat_align, a, b, const_a, const_b)
    elif template_type == 'medoid':
        new_fun = medoid_evaluation(fun_mat_align, a, b, const_a, const_b)
    coeff_mat = align_mat
    x_out_matrix = np.tile(x, (n * m, 1))
    x_align = coeff_mat.flatten()
    x_out_matrix += x_align[:, np.newaxis]
    res = {
        'fun_mat_align': fun_mat_align,
        'template': new_fun,
        'x_out': x_out_matrix
    }
    return res

    # Define the warping shift function
    def warping_shift(coeff, fun_mat, new_fun, x_reg, x_out, i, j):
        st = x_reg + coeff
        template_t = new_fun[i, j, :]
        b = ~np.isnan(template_t)

        data_t = interp1d(st, fun_mat[i, j, :], bounds_error=False, fill_value="extrapolate")(x_out)
        a = ~np.isnan(data_t)
        sel = a & b
        data_t = data_t[sel]
        template_t = template_t[sel]
        distance = np.sum((data_t - template_t) ** 2) / np.sum(sel)
        return distance

    # Define lower and upper warp
    x = np.arange(1, p + 1)
    x_reg = x
    x_out = np.linspace(np.min(x), np.max(x), p)
    min_temp = np.diff(np.ptp(x))
    lower_warp = -shift_max * min_temp
    upper_warp = shift_max * min_temp
    dist_mat = np.full((n, m), np.nan)
    align_mat = np.full((n, m), np.nan)
    count_null = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), axis=2, arr=fun_mat)
    not_null = count_null < p
    fun_mat_align = fun_mat.copy()
    iter_count = 0
    conv = False
    while iter_count < max_iter and not conv:
        iter_count += 1
        dist_mat_old = dist_mat.copy()
        fun_mat_align_old = fun_mat_align.copy()

        # Calculate new templates
        if template_type == 'mean':
            new_fun = template_evaluation(fun_mat_align, a, b, const_a, const_b)
        elif template_type == 'medoid':
            new_fun = medoid_evaluation(fun_mat_align, a, b, const_a, const_b)
        if shift_alignment:
            for i in range(n):
                for j in range(m):
                    if not_null[i, j]:
                        result = minimize_scalar(
                            warping_shift,
                            bounds=(lower_warp, upper_warp),
                            args=(fun_mat, new_fun, x_reg, x_out, i, j),
                            method='bounded')
                        dist_mat[i, j] = result.fun
                        align_mat[i, j] = result.x
                        new_x = x + align_mat[i, j]
                        fun_mat_align[i, j, :] = interp1d(new_x, fun_mat[i, j, :], bounds_error=False, fill_value="extrapolate")(x_out)
                    else:
                        dist_mat[i, j] = np.nan
                        align_mat[i, j] = np.nan
                        fun_mat_align[i, j, :] = np.full(p, np.nan)
        if np.nansum(dist_mat_old) <= np.nansum(dist_mat):
            conv = True
            iter_count -= 1
            fun_mat_align = fun_mat_align_old

    # Calculate final templates
    if template_type == 'mean':
        new_fun = template_evaluation(fun_mat_align, a, b, const_a, const_b)
    elif template_type == 'medoid':
        new_fun = medoid_evaluation(fun_mat_align, a, b, const_a, const_b)
    coeff_mat = align_mat
    x_out_matrix = np.tile(x_out, (n * m, 1))
    x_align = np.tile(coeff_mat, (1, p)).flatten()
    x_out_matrix += x_align
    result = {
        'fun_mat_align': fun_mat_align,
        'template': new_fun,
        'x_out': x_out_matrix
    }
    return result

