import numpy as np
import scipy.linalg as la
from scipy.sparse import linalg as sparla

from parla.comps.determiter.lsqr import lsqr
from parla.comps.determiter.cg import cg
from parla.comps.preconditioning import a_lift_precond


def pcss1(A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
    """Instantiates and calls a PcSS1 PrecondSaddleSolver algorithm."""
    alg = PcSS1()
    return alg(A, b, c, delta, tol, iter_lim, R, upper_tri, z0)


def pcss2(A, b, c, delta, tol, iter_lim, R, upper_tri, z0):
    """Instantiates and calls a PcSS2 PrecondSaddleSolver algorithm."""
    alg = PcSS2()
    return alg(A, b, c, delta, tol, iter_lim, R, upper_tri, z0)


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
        k = 1 if b.ndim == 1 else b.shape[1]
        if k != 1:
            raise NotImplementedError()
        n = A.shape[1]  # == R.shape[0]

        if upper_tri:
            raise NotImplementedError()

        work1 = np.zeros(n)
        work2 = np.zeros(A.shape[0])

        def mv_sp(vec):
            np.dot(R.T, vec, out=work1)
            return R @ work1

        M_sp = sparla.LinearOperator(shape=(n, n),
                                     matvec=mv_sp, rmatvec=mv_sp)

        def mv_gram(vec):
            np.dot(A, vec, out=work2)
            res = A.T @ work2
            res += delta * vec
            return res

        gram = sparla.LinearOperator(shape=(n, n),
                                     matvec=mv_gram, rmatvec=mv_gram)

        rhs = A.T @ b
        if c is not None:
            rhs -= c

        if z0 is None:
            x0 = None
        else:
            x0 = R @ z0

        result = cg(gram, rhs, x0, tol, iter_lim, M_sp, None, None)
        x_star = result[0]
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

        A_pc, M_fwd, M_adj = a_lift_precond(A, delta, R, upper_tri, k)

        if c is None or la.norm(c) == 0:
            # Overdetermined least squares
            if delta > 0:
                b = np.concatenate((b, np.zeros(n)))
            result = lsqr(A_pc, b, atol=tol, btol=tol, iter_lim=iter_lim, x0=z0)
            x = M_fwd(result[0])
            y = b[:m] - A @ x
            result = (x, y, result[7])
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
            result = (x, y, result[7])
            return result

        else:
            raise ValueError('One of "b" or "c" must be zero.')


class PcSS3(PrecondSaddleSolver):
    """Use a no-refresh Newton-sketch style iterative refinement scheme."""

    ERROR_METRIC_INFO = PcSS2.ERROR_METRIC_INFO

    def __init__(self, smart_step_size=True):
        self.smart_step_size = smart_step_size

    def __call__(self, A, b, c, delta, tol, iter_lim, R, upper_tri, z0):

        m, n = A.shape
        assert m == b.shape[0]
        assert b.ndim == 1
        if c is not None:
            raise NotImplementedError()
        if delta > 0:
            raise NotImplementedError()
        if upper_tri:
            raise NotImplementedError()
        else:
            M = R

        # (A'A + \delta I) x = A'b - c
        #  A_pc = [A; \sqrt{\delta}] M,
        #       where the columns of (S A M) form an orthonormal
        #       basis for the range of (S A), and where S was some
        #       sketching operator.
        # Algebra shows that ...
        #
        #   (A_pc' A_pc)^{-1} = M M'.
        #
        # Initialization:
        #       (0a) x = M z0.
        #       (0b) r = A'b - c - A'A x - \delta x
        #
        # Iterate by computing
        #       (2) dx = M M' r
        #       (3) x += dx
        #       (1) r = A'b - c - A'A x - \delta x
        #               Equivalently,  r -= (A'A dx - delta * dx)
        #
        # Termination criteria. We mimick SciPy LSQR w/ preconditioned A.
        #     metric1 = ||M'A' r|| / (scale * ||r||)
        #     metric2 = ||r|| / (||b|| + scale)
        # where scale = ||A M||_{F}. This is equivalent to how we call LSQR,
        # except that LSQR maintains a sequence of non-decreasing values for
        # "scale" that converge to ||A M||_F. We terminate once min(metric1,
        # metric2) falls below "tol".

        # errors = absolute preconditioned normal equation error.

        # workspace and error with x=0
        iter_lim = min(iter_lim, 5*n)
        errors = -np.ones(iter_lim + 2)
        work_gram = np.zeros(m)
        rank = M.shape[1]
        work_rhs1 = np.zeros(rank)
        work_rhs2 = np.zeros(n)
        x = np.zeros(n)
        dx = np.zeros(n)
        rhs = A.T @ b
        np.dot(M.T, rhs, out=work_rhs1)
        err = la.norm(work_rhs1)
        rel_tol = err * tol

        def step_size(dx_):
            Adx = A @ dx_
            num = Adx @ (b - A @ x)
            den = Adx @ Adx
            if den > 0:
                return num / den
            nrm_dx = la.norm(dx_)
            if nrm_dx > 0:
                alpha = x @ dx_ / (nrm_dx ** 2)
                return alpha
            return np.NaN

        # First step: start by computing dx, end by computing error
        if z0 is not None and la.norm(z0) > 0:
            np.dot(M, z0, out=dx)
            alpha = step_size(dx)
            dx *= alpha
            x += dx
            np.dot(A, dx, out=work_gram)
            np.dot(A.T, work_gram, out=work_rhs2)
            rhs -= work_rhs2
            np.dot(M.T, rhs, out=work_rhs1)
            err = la.norm(work_rhs1)

        # main loop; start by computing dx, end by computing error
        it = 1
        iter_lim += 1
        while it < iter_lim and err > rel_tol:
            np.dot(M, work_rhs1, out=dx)  # dx = M M' rhs
            alpha = step_size(dx)
            if np.isnan(alpha) or alpha == 0:
                errors[it] = err
                break
            dx *= alpha
            x += dx
            np.dot(A, dx, out=work_gram)
            np.dot(A.T, work_gram, out=work_rhs2)
            if it % 50 == 0:
                rhs = A.T @ (b - A @ x)
            else:
                rhs -= work_rhs2  # rhs -= alpha * (A'A dx + \delta dx)
            np.dot(M.T, rhs, out=work_rhs1)
            err = la.norm(work_rhs1)
            errors[it] = err
            it += 1

        errors = errors[errors > -1]

        y = b - A @ x
        return x, y, errors
