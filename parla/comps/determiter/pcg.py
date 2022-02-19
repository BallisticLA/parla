import numpy as np
import scipy.linalg as la


def pcg(mv_mat, rhs, mv_pre, iter_lim, tol, x0):
    # mv_mat is a function handle, representing mv_mat(vec) = mat @ vec
    # for a positive definite matrix mat.
    #
    # Use PCG to solve mat @ x == rhs.
    #
    # mv_pre is a function handle, representing mv_pre(vec) = M @ M.T @ vec
    # where M.T @ mat @ M is a better-conditioned positive definite matrix than mat.
    #
    # residuals[i] is the error ||mat x - rhs||_2^2 at iteration i.
    x = x0.copy()
    residuals = -np.ones(iter_lim)
    r = rhs - mv_mat(x)

    d = mv_pre(r)
    delta1_old = np.dot(r, d)
    delta1_new = delta1_old
    cur_err = la.norm(r)
    rel_tol = tol * cur_err

    i = 0
    while i < iter_lim and cur_err > rel_tol:
        # TODO: provide the option of recording  || r ||_2^2, not just ||M' r||_2^2.
        #residuals[i] = delta1_old
        residuals[i] = cur_err
        q = mv_mat(d)
        den = np.dot(d, q)  # equal to d'*mat*d
        alpha = delta1_new / den
        x += alpha * d
        if i % 10 == 0:
            r = rhs - mv_mat(x)
        else:
            r -= alpha * q
        cur_err = la.norm(r)
        s = mv_pre(r)
        delta1_old = delta1_new
        delta1_new = np.dot(r, s)  # equal to ||M'r||_2^2.
        beta = delta1_new / delta1_old
        d = s + beta * d
        i += 1
    residuals = residuals[:i]

    return x, residuals
