import numpy as np
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import cg
from itertools import combinations


# Generate the initial values of Beta
def beta_ini_cal(oridata_list, Y_list, D_d, n, q, p, gamma1):
    Beta_list = []
    for i in range(n):
        Beta_i = []
        for j in range(q):
            data_ij = np.array(oridata_list[i][j])
            Y_ij = np.array(Y_list[i][j])
            Beta_ij = np.linalg.solve(
                data_ij.T @ data_ij + gamma1 * D_d, data_ij.T @ Y_ij
            )
            Beta_i.append(Beta_ij)
        Beta_list.append(np.concatenate(Beta_i))
    Beta_vec = np.concatenate(Beta_list)
    return Beta_vec


# Generate the inverse part of the solution to Beta
def inv_uty_cal(oridata_list, Y_list, D_d, n, q, p, gamma1, theta):
    UTU_list = []
    UTY_list = []
    for i in range(n):
        for j in range(q):
            UTU_list.append(np.dot(oridata_list[i][j].T, oridata_list[i][j]))
            UTY_list.append(np.dot(oridata_list[i][j].T, Y_list[i][j]))
    # Use sparse block diagonal matrix
    UTU_mat = sparse_block_diag(UTU_list, format='csr')
    UTY_vec = np.concatenate([arr.flatten() for arr in UTY_list])
    diagM = kron(eye(n * q), csr_matrix(D_d), format='csr')
    term1 = csr_matrix(n * np.eye(n) - np.ones((n, n)))
    A1TA1 = kron(term1, eye(p * q), format='csr')
    term2 = csr_matrix(q * np.eye(q) - np.ones((q, q)))
    A2TA2 = kron(eye(n), kron(term2, eye(p), format='csr'), format='csr')
    matrix = UTU_mat + gamma1 * diagM + theta * A1TA1 + theta * A2TA2
    
    return {"matrix": matrix, "UTY": UTY_vec}


# Update the corresponding terms in ADMM algorithm
def update_beta(inv, UTY, index1, index2, V1, V2, Lambda1, Lambda2, n, q, p, theta):
    # Compute A1^T * (V1 + Lambda1/theta)
    V1_tilde = V1 + Lambda1 / theta
    A1TV1 = np.zeros((p * q, n))
    A1TV1[:, 0] = np.sum(V1_tilde[:, index1[:, 0] == 0], axis=1)
    for k in range(1, n - 1):
        A1TV1[:, k] = np.sum(V1_tilde[:, index1[:, 0] == k], axis=1) - np.sum(
            V1_tilde[:, index1[:, 1] == k], axis=1
        )
    A1TV1[:, n - 1] = -np.sum(V1_tilde[:, index1[:, 1] == n - 1], axis=1)
    A1TV1_vec = A1TV1.T.flatten()
    
    # Compute A2^T * (V2 + Lambda2/theta)
    V2_tilde = V2 + Lambda2 / theta
    A2TV2_list = []
    for i in range(n):
        A2TV2_i = np.zeros((p, q))
        A2TV2_i[:, 0] = np.sum(
            V2_tilde[(i * p) : (i * p + p), index2[:, 0] == 0], axis=1
        )
        for k in range(1, q - 1):
            A2TV2_i[:, k] = np.sum(
                V2_tilde[(i * p) : (i * p + p), index2[:, 0] == k], axis=1
            ) - np.sum(
                V2_tilde[(i * p) : (i * p + p), index2[:, 1] == k], axis=1
            )
        A2TV2_i[:, q - 1] = -np.sum(
            V2_tilde[(i * p) : (i * p + p), index2[:, 1] == q - 1], axis=1
        )
        A2TV2_list.append(A2TV2_i.T.flatten())
    A2TV2_vec = np.concatenate(A2TV2_list)
    
    # Use iterative solver instead of direct inverse
    modified_rhs = UTY + theta * A1TV1_vec + theta * A2TV2_vec
    Beta, exit_code = cg(inv, modified_rhs, rtol=1e-6, maxiter=1000)
    
    return Beta


def update_v1(Beta, Lambda1, index1, p, q, n, tau, theta, gamma2):
    m1 = index1.shape[0]
    V1 = np.zeros((p * q, m1))
    U1 = np.zeros((p * q, m1))
    Beta_mat = Beta.reshape((n, p * q)).T
    for l in range(m1):
        i = index1[l, 0]
        j = index1[l, 1]
        U1[:, l] = Beta_mat[:, i] - Beta_mat[:, j] - Lambda1[:, l] / theta
        if np.linalg.norm(U1[:, l]) >= tau * gamma2:
            V1[:, l] = U1[:, l]
        else:
            V1[:, l] = (
                (tau * theta / (tau * theta - 1))
                * np.maximum(
                    0, (1 - (gamma2 / theta) / np.linalg.norm(U1[:, l]))
                )
                * U1[:, l]
            )
    return V1


def update_v2(Beta, Lambda2, index2, p, q, n, tau, theta, gamma2):
    m2 = index2.shape[0]
    V2 = np.zeros((p * n, m2))
    U2 = np.zeros((p * n, m2))
    Beta_mat = Beta.reshape((n, p * q)).T
    Beta_newmat = np.zeros((p * n, q))
    for j in range(q):
        Beta_newmat[:, j] = Beta_mat[(j * p) : (j * p + p), :].T.flatten()
    for l in range(m2):
        i = index2[l, 0]
        j = index2[l, 1]
        U2[:, l] = Beta_newmat[:, i] - Beta_newmat[:, j] - Lambda2[:, l] / theta
        if np.linalg.norm(U2[:, l]) >= np.sqrt(n / q) * tau * gamma2:
            V2[:, l] = U2[:, l]
        else:
            V2[:, l] = (
                (tau * theta / (tau * theta - 1))
                * np.maximum(
                    0,
                    (
                        1
                        - (np.sqrt(n / q) * gamma2 / theta)
                        / np.linalg.norm(U2[:, l])
                    ),
                )
                * U2[:, l]
            )
    return V2


def update_Lambda1(Beta, V1, Lambda1_old, index1, theta, p, q, n):
    Beta_mat = Beta.reshape((n, p * q)).T
    m1 = index1.shape[0]
    Lambda1 = np.zeros((p * q, m1))
    # Vectorized computation
    diff = Beta_mat[:, index1[:, 0]] - Beta_mat[:, index1[:, 1]]
    Lambda1 = Lambda1_old + theta * (V1 - diff)
    return Lambda1


def update_Lambda2(Beta, V2, Lambda2_old, index2, theta, p, q, n):
    Beta_mat = Beta.reshape((n, p * q)).T
    Beta_newmat = np.zeros((p * n, q))
    for j in range(q):
        start_index = j * p
        end_index = start_index + p
        Beta_newmat[:, j] = Beta_mat[start_index:end_index, :].T.flatten()
    m2 = index2.shape[0]
    Lambda2 = np.zeros((p * n, m2))
    # Vectorized computation
    diff = Beta_newmat[:, index2[:, 0]] - Beta_newmat[:, index2[:, 1]]
    Lambda2 = Lambda2_old + theta * (V2 - diff)
    return Lambda2


# Main ADMM function
def biclustr_admm(
    inv_UTY_result,
    oridata_list,
    Y_list,
    D_d,
    Beta_ini,
    n,
    q,
    p,
    gamma1,
    gamma2,
    theta,
    tau,
    max_iter,
    eps_abs,
    eps_rel,
):
    index1 = np.array(list(combinations(range(n), 2)))
    index2 = np.array(list(combinations(range(q), 2)))
    m1, m2 = index1.shape[0], index2.shape[0]

    # Parameters initialization
    Beta_old = Beta_ini.copy()
    Beta_old_mat1 = Beta_old.reshape(n, p * q).T
    Beta_old_mat2 = np.zeros((p * n, q))
    for j in range(q):
        Beta_old_mat2[:, j] = Beta_old_mat1[j * p : (j + 1) * p,].T.flatten()
    V1_old = np.zeros((p * q, m1))
    V2_old = np.zeros((p * n, m2))
    for l in range(m1):
        i, j = index1[l, 0], index1[l, 1]
        V1_old[:, l] = Beta_old_mat1[:, i] - Beta_old_mat1[:, j]
    for l in range(m2):
        i, j = index2[l, 0], index2[l, 1]
        V2_old[:, l] = Beta_old_mat2[:, i] - Beta_old_mat2[:, j]
    Lambda1_old = np.zeros((p * q, m1))
    Lambda2_old = np.zeros((p * n, m2))
    
    # Get matrix and RHS from inv_UTY_result (now contains matrix instead of inverse)
    matrix = inv_UTY_result["matrix"]
    UTY = inv_UTY_result["UTY"]
    
    iters = 1
    stop_con = True

    # Iteration
    while stop_con and iters <= max_iter:
        Beta_new = update_beta(
            matrix,  # Pass matrix instead of inv
            UTY,
            index1,
            index2,
            V1_old,
            V2_old,
            Lambda1_old,
            Lambda2_old,
            n,
            q,
            p,
            theta,
        )
        V1_new = update_v1(
            Beta_new, Lambda1_old, index1, p, q, n, tau, theta, gamma2
        )
        V2_new = update_v2(
            Beta_new, Lambda2_old, index2, p, q, n, tau, theta, gamma2 * 0.5
        )
        Lambda1_new = update_Lambda1(
            Beta_new, V1_new, Lambda1_old, index1, theta, p, q, n
        )
        Lambda2_new = update_Lambda2(
            Beta_new, V2_new, Lambda2_old, index2, theta, p, q, n
        )
        
        Beta_new_mat1 = Beta_new.reshape(n, p * q).T
        Beta_new_mat2 = np.zeros((p * n, q))
        for j in range(q):
            Beta_new_mat2[:, j] = Beta_new_mat1[
                j * p : (j + 1) * p :,
            ].T.flatten()
        
        # Compute residuals efficiently (avoid large matrices)
        # Primal residual 1
        pri_resi1_sq = 0
        pri_tol1_max1 = 0
        pri_tol1_max2 = 0
        for l in range(m1):
            i, j = index1[l, 0], index1[l, 1]
            diff = Beta_new_mat1[:, i] - Beta_new_mat1[:, j]
            residual = diff - V1_new[:, l]
            pri_resi1_sq += np.sum(residual ** 2)
            pri_tol1_max1 = max(pri_tol1_max1, np.sum(diff ** 2))
            pri_tol1_max2 = max(pri_tol1_max2, np.sum(V1_new[:, l] ** 2))
        pri_resi1 = np.sqrt(pri_resi1_sq)
        pri_tol1 = np.sqrt(m1 * p * q) * eps_abs + eps_rel * max(
            np.sqrt(pri_tol1_max1), np.sqrt(pri_tol1_max2)
        )
        
        # Primal residual 2
        pri_resi2_sq = 0
        pri_tol2_max1 = 0
        pri_tol2_max2 = 0
        for l in range(m2):
            i, j = index2[l, 0], index2[l, 1]
            diff = Beta_new_mat2[:, i] - Beta_new_mat2[:, j]
            residual = diff - V2_new[:, l]
            pri_resi2_sq += np.sum(residual ** 2)
            pri_tol2_max1 = max(pri_tol2_max1, np.sum(diff ** 2))
            pri_tol2_max2 = max(pri_tol2_max2, np.sum(V2_new[:, l] ** 2))
        pri_resi2 = np.sqrt(pri_resi2_sq)
        pri_tol2 = np.sqrt(m2 * p * n) * eps_abs + eps_rel * max(
            np.sqrt(pri_tol2_max1), np.sqrt(pri_tol2_max2)
        )
        
        # Dual residual 1
        dual_resi1_sq = 0
        dual_tol1_sq = 0
        # k = 0
        diff0 = np.sum(V1_new[:, index1[:, 0] == 0], axis=1) - np.sum(V1_old[:, index1[:, 0] == 0], axis=1)
        dual_resi1_sq += np.sum(diff0 ** 2)
        dual_tol1_sq += np.sum(np.sum(Lambda1_new[:, index1[:, 0] == 0], axis=1) ** 2)
        # k = 1 to n-2
        for k in range(1, n - 1):
            diff_k = (np.sum(V1_new[:, index1[:, 0] == k], axis=1) - 
                     np.sum(V1_new[:, index1[:, 1] == k], axis=1) -
                     np.sum(V1_old[:, index1[:, 0] == k], axis=1) + 
                     np.sum(V1_old[:, index1[:, 1] == k], axis=1))
            dual_resi1_sq += np.sum(diff_k ** 2)
            lambda_diff = (np.sum(Lambda1_new[:, index1[:, 0] == k], axis=1) - 
                          np.sum(Lambda1_new[:, index1[:, 1] == k], axis=1))
            dual_tol1_sq += np.sum(lambda_diff ** 2)
        # k = n-1
        diff_last = -np.sum(V1_new[:, index1[:, 1] == n - 1], axis=1) + np.sum(V1_old[:, index1[:, 1] == n - 1], axis=1)
        dual_resi1_sq += np.sum(diff_last ** 2)
        dual_tol1_sq += np.sum(np.sum(Lambda1_new[:, index1[:, 1] == n - 1], axis=1) ** 2)
        dual_resi1 = np.sqrt(dual_resi1_sq)
        dual_tol1 = np.sqrt(n * p * q) * eps_abs + eps_rel * np.sqrt(dual_tol1_sq)
        
        # Dual residual 2
        dual_resi2_sq = 0
        dual_tol2_sq = 0
        for i in range(n):
            # k = 0
            diff0_2 = (np.sum(V2_new[(i * p):(i * p + p), index2[:, 0] == 0], axis=1) - 
                      np.sum(V2_old[(i * p):(i * p + p), index2[:, 0] == 0], axis=1))
            dual_resi2_sq += np.sum(diff0_2 ** 2)
            dual_tol2_sq += np.sum(np.sum(Lambda2_new[(i * p):(i * p + p), index2[:, 0] == 0], axis=1) ** 2)
            # k = 1 to q-2
            for k in range(1, q - 1):
                diff_k_2 = (np.sum(V2_new[(i * p):(i * p + p), index2[:, 0] == k], axis=1) -
                           np.sum(V2_new[(i * p):(i * p + p), index2[:, 1] == k], axis=1) -
                           np.sum(V2_old[(i * p):(i * p + p), index2[:, 0] == k], axis=1) +
                           np.sum(V2_old[(i * p):(i * p + p), index2[:, 1] == k], axis=1))
                dual_resi2_sq += np.sum(diff_k_2 ** 2)
                lambda_diff_2 = (np.sum(Lambda2_new[(i * p):(i * p + p), index2[:, 0] == k], axis=1) -
                                np.sum(Lambda2_new[(i * p):(i * p + p), index2[:, 1] == k], axis=1))
                dual_tol2_sq += np.sum(lambda_diff_2 ** 2)
            # k = q-1
            diff_last_2 = (-np.sum(V2_new[(i * p):(i * p + p), index2[:, 1] == q - 1], axis=1) + 
                          np.sum(V2_old[(i * p):(i * p + p), index2[:, 1] == q - 1], axis=1))
            dual_resi2_sq += np.sum(diff_last_2 ** 2)
            dual_tol2_sq += np.sum(np.sum(Lambda2_new[(i * p):(i * p + p), index2[:, 1] == q - 1], axis=1) ** 2)
        
        dual_resi2 = np.sqrt(dual_resi2_sq)
        dual_tol2 = np.sqrt(n * p * q) * eps_abs + eps_rel * np.sqrt(dual_tol2_sq)
        
        stop_con = (
            (pri_resi1 > pri_tol1)
            or (pri_resi2 > pri_tol2)
            or (dual_resi1 > dual_tol1)
            or (dual_resi2 > dual_tol2)
        )

        # Update parameters
        iters += 1
        Beta_old, V1_old, V2_old, Lambda1_old, Lambda2_old = (
            Beta_new,
            V1_new,
            V2_new,
            Lambda1_new,
            Lambda2_new,
        )

    # Return parameters
    Beta_list_1 = [Beta_new[i * p * q : (i + 1) * p * q] for i in range(n)]
    Beta_list = [
        [Beta_list_1[i][j * p : (j + 1) * p] for j in range(q)]
        for i in range(n)
    ]
    return {
        "sample number": n,
        "feature number": q,
        "Beta": Beta_list,
        "V1": V1_new,
        "V2": V2_new,
        "Lambda1": Lambda1_new,
        "Lambda2": Lambda2_new,
        "iter": iters,
    }
