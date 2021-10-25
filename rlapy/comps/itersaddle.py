import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import linalg as sparla

from rlapy.comps.lsqr import lsqr
from rlapy.comps.preconditioning import a_times_inv_r, lr_precond_gram, a_times_m


class IterSaddleSolver:

    def __call__(self, A, b, c, delta, tol, iter_lim, M, z0):
        """
        """
        raise NotImplementedError()


def upper_tri_precond_lsqr(A, b, R, tol, iter_lim, z0=None):
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
    z0 : Union[None, ndarray]
        If provided, use as an initial approximate solution to (Ap'Ap) x = Ap' b,
        where Ap = A @ inv(R) is the preconditioned version of A.
    Returns
    -------
    The same values as SciPy's lsqr implementation.
    """
    A_pc = a_times_inv_r(A, R)
    result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
    z = result[0]
    z = solve_triangular(R, z, lower=False, overwrite_b=True)
    result = (z,) + result[1:]
    return result


def pinv_precond_lsqr(A, b, N, tol, iter_lim, z0=None):
    #TODO: modify this so it accept an initial point
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
    k = 1 if b.ndim == 1 else b.shape[1]
    A_precond = a_times_m(A, N, k)
    result = lsqr(A_precond, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
    result = (N @ result[0],) + result[1:]
    return result


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