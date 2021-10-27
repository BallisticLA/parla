from scipy.linalg import solve_triangular
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


def a_times_inv_r(A, delta, R, k=1):
    """Return a linear operator that represents [A; sqrt(delta)*I] @ inv(R)
    """
    if k != 1:
        raise NotImplementedError()

    sqrt_delta = np.sqrt(delta)
    A_lift = a_lift(A, sqrt_delta)

    def forward(arg, work):
        np.copyto(work, arg)
        work = solve_triangular(R, work, lower=False, check_finite=False,
                                overwrite_b=True)
        out = A_lift @ work
        return out

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        if delta > 0:
            work += sqrt_delta * arg
        out = solve_triangular(R, work, 'T', lower=False, check_finite=False)
        return out

    vec_work = np.zeros(A.shape[1])
    mv = lambda vec: forward(vec, vec_work)
    rmv = lambda vec: adjoint(vec, vec_work)
    # if k != 1 then we'd need to allocate workspace differently.
    # (And maybe use workspace differently.)

    A_precond = sparla.LinearOperator(shape=A.shape,
                                      matvec=mv, rmatvec=rmv)
    return A_precond


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
    return A_precond


def lr_precond_gram(A, R):
    """Return a linear operator that represents (A @ inv(R)).T @ (A @ inv(R))"""
    #TODO: provide matmat and rmatmat implementations
    work1 = np.zeros(A.shape[1])
    work2 = np.zeros(A.shape[0])

    def mv(vec):
        np.copyto(work1, vec)
        work1 = solve_triangular(R, work1, lower=False, check_finite=False,
                                 overwrite_b=True)
        np.dot(A, work1, out=work2)
        np.dot(A.T, work2, out=work1)
        res = solve_triangular(R, work1, 'T', lower=False, check_finite=False)
        return res

    AtA_precond = sparla.LinearOperator(shape=(A.shape[1], A.shape[1]),
                                        matvec=mv, rmatvec=mv)
    return AtA_precond
