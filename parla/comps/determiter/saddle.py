import numpy as np
import scipy.linalg as la
from parla.comps.determiter.lsqr import lsqr
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
        m, n = A.shape
        if b is None:
            b = np.zeros(m)
        k = 1 if b.ndim == 1 else b.shape[1]
        if k != 1:
            raise NotImplementedError()

        if upper_tri:
            raise NotImplementedError()
        # inefficiently recover the orthogonal columns of M
        work2 = np.zeros(m)
        pc_dim = R.shape[1]
        work1 = np.zeros(pc_dim)
        fullrank_precond = pc_dim == n
        if not fullrank_precond:
            inv_sing_vals = la.norm(R, axis=0)
            V = R / inv_sing_vals
            R /= inv_sing_vals[-1]
            work3 = np.zeros(n)

        def mv_sp(vec):
            # The preconditioner is RR' + (I - VV')
            np.dot(R.T, vec, out=work1)
            res = np.dot(R, work1)
            if not fullrank_precond:
                res += vec
                np.dot(V.T, vec, out=work1)
                np.dot(V, work1, out=work3)
                res -= work3
            return res

        def mv_gram(vec):
            np.dot(A, vec, out=work2)
            res = A.T @ work2
            res += delta * vec
            return res

        rhs = A.T @ b
        if c is not None:
            rhs -= c

        if z0 is None or (not fullrank_precond):
            # TODO: proper initialization with low-rank preconditioners
            x = np.zeros(n)
        else:
            x = R @ z0

        residuals = -np.ones(iter_lim)
        r = rhs - mv_gram(x)
        d = mv_sp(r)
        delta1_old = np.dot(r, d)
        delta1_new = delta1_old
        rel_sq_tol = (delta1_old * tol) * tol
        rel_sq_tol = max(rel_sq_tol, 1e-20)

        i = 0
        while i < iter_lim and delta1_new > rel_sq_tol:
            residuals[i] = delta1_old
            q = mv_gram(d)
            alpha = delta1_new / np.dot(d, q)
            x += alpha * d
            if i % 10 == 0:
                r = rhs - mv_gram(x)
            else:
                r -= alpha * q
            s = mv_sp(r)
            delta1_old = delta1_new
            delta1_new = np.dot(r, s)
            beta = delta1_new / delta1_old
            d = s + beta*d
            i += 1

        y = b - A @ x
        residuals = residuals[:i]

        result = (x, y, residuals)

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
