import scipy.linalg as la
import scipy.sparse.linalg as sparla
import numpy as np


def a_lift(A, scale):
    if scale == 0:
        return A
    else:
        A = np.row_stack((A, scale*np.eye(A.shape[1])))
        # Explicitly augmenting A seems like overkill.
        # However, it's faster than the LinearOperator approach I tried.
        return A

#TODO: consolidate the r and M cases below.
#   use a flag for upper-triangular.

#TODO: rename functions here ("times" and "inv_r" and "m" is a mess).


def a_times_inv_r(A, delta, R, k=1):
    """Return a linear operator that represents [A; sqrt(delta)*I] @ inv(R)
    """
    if k != 1:
        raise NotImplementedError()

    sqrt_delta = np.sqrt(delta)
    A_lift = a_lift(A, sqrt_delta)

    def forward(arg, work):
        np.copyto(work, arg)
        work = la.solve_triangular(R, work, lower=False, check_finite=False,
                                   overwrite_b=True)
        out = A_lift @ work
        return out

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        if delta > 0:
            work += sqrt_delta * arg
        out = la.solve_triangular(R, work, 'T', lower=False, check_finite=False)
        return out

    vec_work = np.zeros(A.shape[1])
    mv = lambda vec: forward(vec, vec_work)
    rmv = lambda vec: adjoint(vec, vec_work)
    # if k != 1 then we'd need to allocate workspace differently.
    # (And maybe use workspace differently.)

    A_precond = sparla.LinearOperator(shape=A.shape,
                                      matvec=mv, rmatvec=rmv)

    M_fwd = lambda z: la.solve_triangular(R, z, lower=False)
    M_adj = lambda w: la.solve_triangular(R, w, 'T', lower=False)

    return A_precond, M_fwd, M_adj


def a_times_m(A, delta, M, k=1):
    if k != 1:
        raise NotImplementedError()

    sqrt_delta = np.sqrt(delta)
    A_lift = a_lift(A, sqrt_delta)

    def forward(arg, work):
        np.dot(M, arg, out=work)
        out = A_lift @ work
        return out

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        if delta > 0:
            work += sqrt_delta * arg
        return M.T @ work

    vec_work = np.zeros(A.shape[1])
    mv = lambda x: forward(x, vec_work)
    rmv = lambda y: adjoint(y, vec_work)
    # if k != 1 then we'd need to allocate workspace differently.
    # (And maybe use workspace differently.)

    A_precond = sparla.LinearOperator(shape=(A.shape[0], M.shape[1]),
                                      matvec=mv, rmatvec=rmv)

    M_fwd = lambda z: M @ z
    M_adj = lambda w: M.T @ w

    return A_precond, M_fwd, M_adj
