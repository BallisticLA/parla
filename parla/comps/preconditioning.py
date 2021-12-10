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


def a_lift_precond(A, delta, R, upper_tri=False, k=1):
    if k != 1:
        raise NotImplementedError()

    sqrt_delta = np.sqrt(delta)
    A_lift = a_lift(A, sqrt_delta)
    m = A.shape[0]

    if upper_tri:

        def forward(arg, work):
            np.copyto(work, arg)
            work = la.solve_triangular(R, work, lower=False, check_finite=False,
                                       overwrite_b=True)
            out = A_lift @ work
            return out

        def adjoint(arg, work):
            np.dot(A.T, arg[:m], out=work)
            if delta > 0:
                work += sqrt_delta * arg[m:]
            out = la.solve_triangular(R, work, 'T', lower=False, check_finite=False)
            return out

        M_fwd = lambda z: la.solve_triangular(R, z, lower=False)
        M_adj = lambda w: la.solve_triangular(R, w, 'T', lower=False)

    else:

        def forward(arg, work):
            np.dot(R, arg, out=work)
            out = A_lift @ work
            return out

        def adjoint(arg, work):
            np.dot(A.T, arg[:m], out=work)
            if delta > 0:
                work += sqrt_delta * arg[m:]
            return R.T @ work

        M_fwd = lambda z: R @ z
        M_adj = lambda w: R.T @ w

    vec_work = np.zeros(A.shape[1])
    mv = lambda x: forward(x, vec_work)
    rmv = lambda y: adjoint(y, vec_work)
    # if k != 1 then we'd need to allocate workspace differently.
    # (And maybe use workspace differently.)

    A_precond = sparla.LinearOperator(shape=(A_lift.shape[0], R.shape[1]),
                                      matvec=mv, rmatvec=rmv)
    return A_precond, M_fwd, M_adj


def svd_right_precond(A_ske):
    U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                          full_matrices=False)
    eps = np.finfo(float).eps
    rank = np.count_nonzero(sigma > sigma[0] * A_ske.shape[1] * eps)
    Vh = Vh[:rank, :]
    U = U[:, :rank]
    sigma = sigma[:rank]
    M = Vh.T / sigma
    return M, U, sigma, Vh
