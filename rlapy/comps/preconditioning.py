from scipy.linalg import solve_triangular
import scipy.sparse.linalg as sparla
import numpy as np


def a_times_inv_r(A, R, k=1):
    """Return a linear operator that represents A @ inv(R) """

    def forward(arg, work):
        np.copyto(work, arg)
        work = solve_triangular(R, work, lower=False, check_finite=False,
                                overwrite_b=True)
        return A @ work

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        return solve_triangular(R, work, 'T', lower=False, check_finite=False)

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


def a_times_m(A, M, k=1):
    _m, n = A.shape
    work = np.zeros(n)

    def mv(vec):
        np.dot(M, vec, out=work)
        return A @ work

    def rmv(vec):
        np.dot(A.T, vec, out=work)
        return M.T @ work

    if k == 1:
        A_precond = sparla.LinearOperator(shape=(_m, M.shape[1]),
                                          matvec=mv, rmatvec=rmv)
    else:
        #TODO: write tests for this case
        mat_work = np.zeros((n, k))

        def mm(mat):
            np.dot(M, mat, out=mat_work)
            return A @ work

        def rmm(mat):
            np.dot(A.T, mat, out=mat_work)
            return M.T @ work

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
