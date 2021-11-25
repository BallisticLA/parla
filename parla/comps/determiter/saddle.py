import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as sparla

from parla.comps.determiter.lsqr import lsqr
from parla.comps.determiter.cg import cg
from parla.comps.preconditioning import a_times_inv_r, a_times_m


class PrecondSaddleSolver:

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        """
        The problem data (A, b, c, delta) define a block linear system

             [  I   |     A   ] [y_opt] = [b]           (*)
             [  A'  | -delta*I] [x_opt]   [c].

        The matrix A is m-by-n and tall, b and c are vectors, and delta is >= 0.
        This method produces (x_approx, y_approx) that approximate (x_opt, y_opt).

        This method uses some iterative algorithm with tol and iter_lim as
        termination criteria. The meaning of tol is implementation-dependent.

        The underlying iterative algorithm uses R as a preconditioner and
        initializes x_approx based on the pair (R, z0).

            If upper_tri is True, then we expect that the condition number of
            A_{pc} := (A R^{-1}) isn't large, and we initialize x_approx = R^{-1} z0.

            If upper_tri is False, then we expect that the condition number of
            A_{pc} := (A R) is not large and we initialize x_approx = R z0.

        Parameters
        ----------
        A : ndarray
            Data matrix with m rows and n columns.
        b : ndarray
            Upper block in the right-hand-side. b.shape = (m,).
        c : ndarray
            The lower block the right-hand-side. c.shape = (n,).
        delta : float
            Nonnegative regularization parameter.
        tol : float
            Used as stopping criteria.
        iter_lim : int
            An upper-bound on the number of steps the iterative algorithm
            is allowed to take.
        R : ndarray
            Defines the preconditioner, has R.shape = (n, n).
        upper_tri : bool
            If upper_tri is True, then precondition by M = R^{-1}.
            If upper_tri is False, then precondition by M = R.
        z0 : Union[None, ndarray]
            If provided, use as an initial approximate solution to (Ap'Ap) x = Ap' b,
            where Ap = A M is the preconditioned version of A.

        Returns
        -------
        x_approx : ndarray
            Has size (n,).
        y_approx : ndarray
            Has size (m,). Usually set to y := b - A x_approx, which solves the
            upper block of equations in (*).
        errors : ndarray
            errors[i] is some error metric of (x_approx, y_approx) at iteration i
            of the algorithm. The algorithm took errors.size steps.

        Notes
        -----
        The following characterization holds for x_opt in (*):
            (A' A + delta * I) x_opt = A'b - c.
        We call that system the "normal equations".
        """
        raise NotImplementedError()


class PcSS1(PrecondSaddleSolver):

    ERROR_METRIC_INFO = """
        2-norm of the residual from the normal equations
    """

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
        if b is None:
            b = np.zeros(A.shape[0])
        b = b.astype(R.dtype)
        k = 1 if b.ndim == 1 else b.shape[1]
        if k != 1:
            raise NotImplementedError()
        n = A.shape[1]  # == R.shape[0]

        if upper_tri:
            raise NotImplementedError()

        # Note to Riley: refer to page 28 of your notebook
        is_complex = 'complex' in str(R.dtype)
        work1 = np.zeros(n, dtype=R.dtype)
        work2 = np.zeros(A.shape[0], dtype=R.dtype)

        R_adj = R.T.conj() if is_complex else R.T

        def mv_sp(vec):
            np.dot(R_adj, vec, out=work1)
            return R @ work1

        M_sp = sparla.LinearOperator(shape=(n, n),
                                     matvec=mv_sp, rmatvec=mv_sp,
                                     dtype=R.dtype)

        A_adj = A.T.conj() if is_complex else A.T

        def mv_gram(vec):
            np.dot(A, vec, out=work2)
            res = A_adj @ work2
            res += delta * vec
            return res

        gram = sparla.LinearOperator(shape=(n, n),
                                     matvec=mv_gram, rmatvec=mv_gram,
                                     dtype=R.dtype)

        rhs = (A.T @ b).astype(R.dtype)
        if c is not None:
            c = c.astype(R.dtype)
            rhs -= c

        if z0 is None:
            x0 = None
        else:
            x0 = R @ z0

        result = cg(gram, rhs, x0, tol, iter_lim, M_sp, None, None)
        x_star = result[0].astype(A.dtype)
        y_star = b - A @ x_star

        result = (x_star, y_star, result[2])

        return result


class PcSS2(PrecondSaddleSolver):

    ERROR_METRIC_INFO = """
        2-norm of the residual from the preconditioned normal equations
        (preconditioning on the left and right).
    """

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
                b = np.concatenate((b, np.zeros(n, dtype=b.dtype)))
            b = b.astype(A_pc.dtype)
            result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
            x = M_fwd(result[0]).astype(A.dtype)
            y = b[:m] - A @ x
            result = (x, y, result[7])
            return result

        elif b is None or la.norm(b) == 0:
            # Underdetermined least squares
            c_pc = M_adj(c)
            result = lsqr(A_pc.T, c_pc, atol=tol, btol=tol, iter_lim=iter_lim)
            y = result[0]
            if delta > 0:
                y = y[:m].astype(A.dtype)
                x = (A.T @ y - c) / delta
            else:
                x = np.NaN * np.empty(n)
            result = (x, y, result[7])
            return result

        else:
            raise ValueError('One of "b" or "c" must be zero.')
