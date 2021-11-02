import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as sparla

from rlapy.comps.lsqr import lsqr
from rlapy.comps.preconditioning import a_times_inv_r, lr_m_precond_gram, a_times_m, \
    lr_invr_precond_gram


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


class PcSS1(PrecondSaddleSolver):

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        k = 1 if (b is None or b.ndim == 1) else b.shape[1]

        if upper_tri:
            AtA_pc, M_fwd, M_adj = lr_invr_precond_gram(A, delta, R, k)
        else:
            AtA_pc, M_fwd, M_adj = lr_m_precond_gram(A, delta, R, k)

        if b is None:
            b = np.zeros(A.shape[0])

        if c is not None:
            rhs = M_adj(A.T @ b + c)
        else:
            rhs = M_adj(A.T @ b)
        result = sparla.cg(AtA_pc, rhs, atol=tol, tol=tol, maxiter=iter_lim, x0=z0)

        x_star = M_fwd(result[0])
        y_star = b - A @ x_star

        result = (x_star, y_star) + result[2:]

        return result


class PcSS2(PrecondSaddleSolver):

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        m, n = A.shape
        k = 1 if (b is None or b.ndim == 1) else b.shape[1]

        if upper_tri:
            A_pc, M_fwd, M_adj = a_times_inv_r(A, delta, R, k)
        else:
            A_pc, M_fwd, M_adj = a_times_m(A, delta, R, k)

        if c is None or la.norm(c) == 0:
            # Overdetermined least squares
            if delta > 0:
                b = np.concatenate((b, np.zeros(n)))
            result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
            x = M_fwd(result[0])
            y = b[:m] - A @ x
            result = (x, y) + result[1:]
            return result

        elif b is None or la.norm(b) == 0:
            # Underdetermined least squares
            c_pc = M_adj(c)
            result = lsqr(A_pc.T, c_pc, atol=tol, btol=tol, iter_lim=iter_lim)
            y = result[0]
            if delta > 0:
                y = y[:m]
                x = (A.T @ y - c) / delta
            else:
                x = np.NaN * np.empty(n)
            result = (x, y) + result[1:]
            return result

        else:
            raise ValueError('One of "b" or "c" must be zero.')
