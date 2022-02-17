import numpy as np


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
    rel_sq_tol = (delta1_old * tol) * tol
    rel_sq_tol = max(rel_sq_tol, 1e-20)

    i = 0
    while i < iter_lim and delta1_new > rel_sq_tol:
        residuals[i] = delta1_old
        q = mv_mat(d)
        alpha = delta1_new / np.dot(d, q)
        x += alpha * d
        if i % 10 == 0:
            r = rhs - mv_mat(x)
        else:
            r -= alpha * q
        s = mv_pre(r)
        delta1_old = delta1_new
        delta1_new = np.dot(r, s)
        beta = delta1_new / delta1_old
        d = s + beta * d
        i += 1
    residuals = residuals[:i]

    return x, residuals
