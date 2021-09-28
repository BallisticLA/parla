from scipy.linalg import solve_triangular
import scipy.sparse.linalg as sparla
from rlapy.comps.lsqr import lsqr
import numpy as np


def a_times_inv_r(A, R, k=1):
    """Return a linear operator that represents A @ inv(R) """

    vec_work = np.zeros(A.shape[1])

    def forward(arg, work):
        np.copyto(work, arg)
        work = solve_triangular(R, work, lower=False, check_finite=False,
                                overwrite_b=True)
        return A @ work

    def adjoint(arg, work):
        np.dot(A.T, arg, out=work)
        return solve_triangular(R, work, 'T', lower=False, check_finite=False)

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


def upper_tri_precond_lsqr(A, b, R, tol, iter_lim, x0=None):
    """
    Run preconditioned LSQR to obtain an approximate solution to
        min{ || A @ x - b ||_2 : x in R^n }
    where A.shape = (m, n) has m >> n, so the problem is over-determined.

    Parameters
    ----------
    A : ndarray
        Data matrix with m rows and n columns. Columns are presumed linearly
        independent (for now).
    b : ndarray
        Right-hand-side. b.shape = (m,) or b.shape = (m, k).
    R : ndarray
        The upper-triangular preconditioner, has R.shape = (n, n).
    tol : float
        Must be positive. Stopping criteria for LSQR.
    iter_lim : int
        Must be positive. Stopping criteria for LSQR.
    x0 : Union[None, ndarray]
        If provided, use as an initial approximate solution to (A'A) x = A' b.
        Internally, we initialize preconditioned lsqr at y0 = R x0.
    Returns
    -------
    The same values as SciPy's lsqr implementation.
    """
    A_pc = a_times_inv_r(A, R)
    if x0 is not None:
        y0 = (R @ x0).ravel()
        result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=y0)
    else:
        result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim)
    z = result[0]
    z = solve_triangular(R, z, lower=False, overwrite_b=True)
    result = (z,) + result[1:]
    return result


def pinv_precond_lsqr(A, b, N, tol, iter_lim):
    """
    Run preconditioned LSQR to obtain an approximate solution to
        min{ || A @ x - b ||_2 : x in R^n }
    where A.shape = (m, n) has m >> n, so the problem is over-determined.

    Parameters
    ----------
    A : ndarray
        Data matrix with m rows and n columns.
    b : ndarray
        Right-hand-side b.shape = (m,) or b.shape = (m, k).
    N : ndarray
        The condition number of A @ N should be near one and its rank should be
        the same as that of A.
    tol : float
        Must be positive. Stopping criteria for LSQR.
    iter_lim : int
        Must be positive. Stopping criteria for LSQR.

    Returns
    -------
    The same values as SciPy's lsqr implementation.
    """
    m, n = A.shape
    work = np.zeros(n)

    def mv(vec):
        np.dot(N, vec, out=work)
        return A @ work

    def rmv(vec):
        np.dot(A.T, vec, out=work)
        return N.T @ work

    if b.ndim == 1:
        A_precond = sparla.LinearOperator(shape=(m, N.shape[1]),
                                          matvec=mv, rmatvec=rmv)
    if b.ndim == 2:
        #TODO: write tests for this case
        mat_work = np.zeros((n, b.shape[1]))

        def mm(mat):
            np.dot(N, mat, out=mat_work)
            return A @ work

        def rmm(mat):
            np.dot(A.T, mat, out=mat_work)
            return N.T @ work

        A_precond = sparla.LinearOperator(shape=(m, N.shape[1]),
                                          matvec=mv, rmatvec=rmv,
                                          matmat=mm, rmatmat=rmm)

    result = lsqr(A_precond, b, atol=tol, btol=tol, iter_lim=iter_lim)
    result = (N @ result[0],) + result[1:]
    return result


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


def upper_tri_precond_cg(A, b, R, tol, iter_lim, x0=None):
    """
    Run conjugate gradients on the positive semidefinite linear system
        ((A R^-1)' (A R^-1)) y == (R^-1)' b
    and set x = R^-1 y, as a means to solve the linear system
        (A' A) x = b.

    Parameters
    ----------
    A : np.ndarray
        Tall data matrix.
    b : np.ndarray
        right-hand-side. b.size == A.shape[0].
    R : np.ndarray
        Nonsingular upper-triangular preconditioner.
        The condition number of (A R^-1) should be near one.
    tol : float
        Stopping criteria for ScPy's cg implementation.
        Considered with respect to the preconditioned system.
    iter_lim : int
        Stopping criteria for SciPy's cg implementation
    x0 : Union[None, np.ndarray]
        If provided, use as an initial solution to (A' A) x = b.

    Returns
    -------
    The same values as SciPy's cg implementation.
    """
    #TODO: write tests
    AtA_precond = lr_precond_gram(A, R)
    b_precond = solve_triangular(R, b, 'T', lower=False, check_finite=False)
    if x0 is not None:
        y0 = (R @ x0).ravel()
        result = sparla.cg(AtA_precond, b_precond, atol=tol, btol=tol,
                           iter_lim=iter_lim, x0=y0)
    else:
        result = sparla.cg(AtA_precond, b_precond, atol=tol, btol=tol,
                           iter_lim=iter_lim)
    z = result[0]
    z = solve_triangular(R, z, lower=False, overwrite_b=True)
    result = (z,) + result[1:]
    return result
