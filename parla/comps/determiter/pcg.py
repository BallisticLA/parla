import warnings
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
    # residuals[i] is the error ||M' (mat x - rhs)||_2 at iteration i.
    #
    x = x0.copy()
    x_opt = x.copy()
    i_opt = 0
    residuals = -np.ones(iter_lim)
    r = rhs - mv_mat(x)

    d = mv_pre(r)
    delta1_old = np.dot(r, d)
    delta1_new = delta1_old
    cur_err = np.sqrt(delta1_new)
    rel_tol = tol * cur_err

    i = 0
    while i < iter_lim and cur_err > rel_tol and i <= i_opt + 10:
        # TODO: provide the option of recording  || r ||_2^2, not just ||M' r||_2^2.
        residuals[i] = cur_err
        q = mv_mat(d)
        den = np.dot(d, q)  # equal to d'*mat*d
        alpha = delta1_new / den
        x += alpha * d
        if i % 10 == 0:
            r = rhs - mv_mat(x)
        else:
            r -= alpha * q
        s = mv_pre(r)
        delta1_old = delta1_new
        delta1_new = np.dot(r, s)  # equal to ||M'r||_2^2.
        beta = delta1_new / delta1_old
        d *= beta
        d += s
        cur_err = np.sqrt(delta1_new)
        if cur_err < residuals[i]:
            i_opt = i
            x_opt[:] = x[:]
        i += 1
    if i > i_opt + 1:
        msg = f'PCG terminated after iteration {i-1}, but found its best iterate at the end of step {i_opt}.'
        warnings.warn(msg)
        residuals = residuals[:(i_opt + 1)]
    else:
        residuals = residuals[:i]

    return x_opt, residuals
