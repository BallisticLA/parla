from scipy.linalg import solve_triangular
import scipy.sparse.linalg as sparla
import numpy as np


def a_times_inv_r(A, delta, R, k=1):
    """Return a linear operator that represents [A; sqrt(delta)*I] @ inv(R)
    """

    m = A.shape[0]
    m_lift = m + A.shape[1]
    sqrt_delta = np.sqrt(delta)

    def forward(arg, work):
        np.copyto(work, arg)
        work = solve_triangular(R, work, lower=False, check_finite=False,
                                overwrite_b=True)
        if delta == 0:
            out = A @ work
        else:
            shape = (m_lift,) if arg.ndim == 1 else (m_lift, arg.shape[1])
            out = np.empty(shape=shape)
            out[:m] = A @ work
            out[m:] = sqrt_delta * work
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

    if k == 1:
        A_precond = sparla.LinearOperator(shape=A.shape,
                                          matvec=mv, rmatvec=rmv)
    else:
        #TODO: write tests for this case
        mat_work = np.zeros(A.shape[1], k)
        mm = lambda mat: forward(mat, mat_work)
        rmm = lambda mat: adjoint(mat, mat_work)
        A_precond = sparla.LinearOperator(shape=A.shape,
                                          matvec=mv, rmatvec=rmv,
                                          matmat=mm, rmatmat=rmm)
    return A_precond


def a_times_m(A, delta, M, k=1):
    #TODO: update to set up the matvecs and matmats like a_times_inv_r.
    # similarly update for delta > 0.
    _m, n = A.shape
    sqrt_delta = np.sqrt(delta)
    m_lift = _m + n

    def forward(arg, work):
        np.dot(M, arg, out=work)
        if delta == 0:
            out = A @ work
        else:
            shape = (m_lift,) if arg.ndim == 1 else (m_lift, arg.shape[1])
            out = np.empty(shape=shape)
            out[:_m] = A @ work
            out[_m:] = sqrt_delta * work
        return out

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        if delta > 0:
            work += sqrt_delta * arg
        return M.T @ work

    vec_work = np.zeros(n)
    mv = lambda x: forward(x, vec_work)
    rmv = lambda y: adjoint(y, vec_work)

    if k == 1:
        A_precond = sparla.LinearOperator(shape=(_m, M.shape[1]),
                                          matvec=mv, rmatvec=rmv)
    else:
        #TODO: write tests for this case
        mat_work = np.zeros((n, k))
        mm = lambda mat: forward(mat, mat_work)
        rmm = lambda mat: adjoint(mat, mat_work)
        A_precond = sparla.LinearOperator(shape=(_m, M.shape[1]),
                                          matvec=mv, rmatvec=rmv,
                                          matmat=mm, rmatmat=rmm)
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
