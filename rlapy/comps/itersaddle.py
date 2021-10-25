import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as sparla

from rlapy.comps.lsqr import lsqr
from rlapy.comps.preconditioning import a_times_inv_r, lr_precond_gram, a_times_m


class PrecondSaddleSolver:

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        """TODO: update docstring for saddle point system.
        Run preconditioned LSQR to obtain an approximate solution to
            min{ || A @ x - b ||_2 : x in R^n }
        where A.shape = (m, n) has m >> n, so the problem is over-determined.

        Parameters
        ----------
        A : ndarray
            Data matrix with m rows and n columns.
        b : ndarray
            Right-hand-side. b.shape = (m,) or b.shape = (m, k).
        c : ndarray
            ....
        delta : float
            ...
        tol : float
            Must be positive. Stopping criteria for LSQR.
        iter_lim : int
            Must be positive. Stopping criteria for LSQR.
        R : ndarray
            Defines the preconditioner, has R.shape = (n, n).
        upper_tri : bool
            If upper_tri is True, then precondition by M = inv(R).
            If upper_tri is False, then precondition by M = R.
        z0 : Union[None, ndarray]
            If provided, use as an initial approximate solution to (Ap'Ap) x = Ap' b,
            where Ap = A @ M is the preconditioned version of A.
        """
        raise NotImplementedError()


class PcSS2(PrecondSaddleSolver):

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        m, n = A.shape
        k = 1 if b.ndim == 1 else b.shape[1]

        if upper_tri:
            A_pc = a_times_inv_r(A, delta, R, k)
            M_func = lambda z: la.solve_triangular(R, z, lower=False)
        else:
            A_pc = a_times_m(A, delta, R, k)
            M_func = lambda z: R @ z

        if c is None or la.norm(c) == 0:
            # Overdetermined least squares
            if delta > 0:
                b = np.concatenate((b, np.zeros(n)))
            result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
            x = M_func(result[0])
            y = b[:m] - A @ x
            result = (x, y) + result[1:]
            return result

        elif b is None or la.norm(b) == 0:
            # Underdetermined least squares
            c_pc = M_func(c)
            result = lsqr(A_pc.T, c_pc, atol=tol, btol=tol, iter_lim=iter_lim)
            y = result[0]
            if delta > 0:
                y = y[:m]
                x = (c - A.T @ y) / delta
            else:
                x = np.NaN * np.empty(n)
            result = (x, y) + result[1:]
            return result

        else:
            raise ValueError('One of "b" or "c" must be zero.')


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
    b_precond = la.solve_triangular(R, b, 'T', lower=False, check_finite=False)
    if x0 is not None:
        y0 = (R @ x0).ravel()
        result = sparla.cg(AtA_precond, b_precond, atol=tol, btol=tol,
                           iter_lim=iter_lim, x0=y0)
    else:
        result = sparla.cg(AtA_precond, b_precond, atol=tol, btol=tol,
                           iter_lim=iter_lim)
    z = result[0]
    z = la.solve_triangular(R, z, lower=False, overwrite_b=True)
    result = (z,) + result[1:]
    return result
