import unittest
import numpy as np
import scipy.linalg as la
import parla.drivers.least_squares as rlsq
import parla.utils.sketching as usk
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.stats as ustats
import parla.tests.matmakers as matmakers
from parla.comps.determiter.saddle import PcSS3


def consistent_tall():
    seed = 190489290
    rng = np.random.default_rng(seed)
    m, n = 100, 10
    A = matmakers.simple_mat(m, n, scale=1, rng=rng)
    U, s, Vt = la.svd(A)
    x = rng.standard_normal(n)
    b = A @ x
    ath = AlgTestHelper(A, b, x, U, s, Vt)
    return ath


def consistent_lowrank():
    seed = 8923890298
    rng = np.random.default_rng(seed)
    m, n, rank = 100, 10, 5
    U = usk.orthonormal_operator(m, rank, rng)
    s = rng.random(rank) + 1e-4
    Vt = usk.orthonormal_operator(rank, n, rng)
    A = (U * s) @ Vt
    x = rng.standard_normal(n)
    b = A @ x
    ath = AlgTestHelper(A, b, x, U, s, Vt)
    return ath


def consistent_square():
    seed = 3278992245
    rng = np.random.default_rng(seed)
    n = 10
    U = usk.orthonormal_operator(n, n, rng)
    s = rng.random(n) + 1e-4
    Vt = usk.orthonormal_operator(n, n, rng)
    A = (U * s) @ Vt
    x = rng.standard_normal(n)
    b = A @ x
    ath = AlgTestHelper(A, b, x, U, s, Vt)
    return ath


def inconsistent_orthog():
    seed = 19837647834763
    n, m = 1000, 100
    rng = np.random.default_rng(seed)
    # Generate A
    U = usk.orthonormal_operator(n, m, rng)
    Vt = usk.orthonormal_operator(m, m, rng)
    s = rng.random(m) + 1e-4
    A = (U * s) @ Vt
    # Generate b
    b = rng.standard_normal(n)
    b = b - U @ (U.T @ b)
    b *= 1e2 / la.norm(b)
    # return
    x_opt = np.zeros(m)
    ath = AlgTestHelper(A, b, x_opt, U, s, Vt)
    return ath


def inconsistent_gen():
    seed = 897809809
    rng = np.random.default_rng(seed)
    m, n = 1000, 100
    num_hi = 30
    num_lo = n - num_hi
    # Make A
    hi_spec = 1e5 * np.ones(num_hi) + rng.random(num_hi)
    lo_spec = np.ones(num_lo) + rng.random(num_lo)
    spec = np.concatenate([hi_spec, lo_spec])
    U = usk.orthonormal_operator(m, n, rng)
    Vt = usk.orthonormal_operator(n, n, rng)
    A = (U * spec) @ Vt
    # Make b
    hi_x = rng.standard_normal(num_hi)/1e5
    lo_x = rng.standard_normal(num_lo)
    x = np.concatenate([hi_x, lo_x])
    b_orth = rng.standard_normal(m) * 1e2
    b_orth -= U @ (U.T @ b_orth)  # orthogonal to range(A)
    b = A @ x + b_orth
    # Return
    ath = AlgTestHelper(A, b, x, U, spec, Vt)
    return ath


def inconsistent_stackid():
    seed = 2837592038243
    rng = np.random.default_rng(seed)

    A = np.tile(np.eye(70), (10, 1))
    m, n = A.shape
    A = (A.T * rng.lognormal(size=(m,))).T
    x0 = rng.standard_normal(size=(n,))
    b0 = A @ x0
    b = b0 + rng.standard_normal(size=(m,))

    U, spec, Vt = la.svd(A, full_matrices=False)
    x = (Vt.T @ (U.T @ b) / spec)

    ath = AlgTestHelper(A, b, x, U, spec, Vt)
    return ath

"""
TODO: update these tests to take advantage of logging in least-squares drivers.
The logging will let us test convergence rates without having to re-run the algorithm
many times.
"""


class AlgTestHelper:

    def __init__(self, A, b, x_opt, U, s, Vt):
        self.A = A
        self.b = b
        self.x_opt = x_opt
        self._result = None
        self.x_approx = None
        self.U = U
        self.s = s  # shouldn't have anything below datatype's "eps".
        self.Vt = Vt
        self.tester = None  # unittest.TestCase

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, res):
        self._result = res
        self.x_approx = res[0]
        self.log = res[1]

    def test_x_angle(self, tol):
        """
        x' x_opt >= (1 - tol)*||x|| ||x_opt||
        """
        y_opt = self.Vt @ self.x_opt
        norm_y_opt = la.norm(y_opt)
        y = self.Vt @ self.x_approx
        norm_y = la.norm(y)
        if norm_y_opt < 1e-8:
            # Norm is too small to accurately compute cosine
            self.tester.assertLessEqual(abs(norm_y - norm_y_opt), tol)
        else:
            y_opt /= norm_y_opt
            y /= norm_y
            cosine = np.dot(y, y_opt)
            self.tester.assertGreaterEqual(cosine, 1 - tol)

    def test_x_norm(self, tol):
        """
        (1 - tol)*||x_opt|| <= ||x|| <= (1+tol)*||x_opt|| + tol
        """
        norm = la.norm(self.Vt @ self.x_approx)
        norm_opt = la.norm(self.Vt @ self.x_opt)
        self.tester.assertLessEqual(norm, (1+tol)*norm_opt + tol)
        self.tester.assertLessEqual((1-tol)*norm_opt, norm)

    def test_delta_x(self, tol):
        """
        ||x - x_opt|| <= tol
        """
        delta_x = self.x_opt - self.x_approx
        nrm = la.norm(delta_x, ord=2) / (1 + min(la.norm(self.x_opt),
                                                 la.norm(self.x_approx)))
        self.tester.assertLessEqual(nrm, tol)

    def test_residual_proj(self, tol):
        """
        || U U' (A x - b) || / ||A x - b|| <= tol
        """
        # This test is probably better scaled than the normal equations
        residual = self.A @ self.x_approx - self.b
        residual_proj = self.U @ (self.U.T @ residual)
        nrm = la.norm(residual_proj, ord=2) / la.norm(residual)
        self.tester.assertLessEqual(nrm, tol)

    def test_objective(self, tol):
        """
        ||A x - b|| <= ||A x_opt - b|| + tol
        """
        res_approx = self.b - self.A @ self.x_approx
        res_opt = self.b - self.A @ self.x_opt
        nrm_approx = la.norm(res_approx, ord=2)
        nrm_opt = la.norm(res_opt, ord=2)
        self.tester.assertLessEqual(nrm_approx, nrm_opt + tol)

    def test_normal_eqs(self, tol):
        """
        || A' A x - A' b|| <= tol
        """
        gap = self.A.T @ self.b - self.A.T @ (self.A @ self.x_approx)
        nrm = la.norm(gap, ord=2)
        self.tester.assertLessEqual(nrm, tol)


class TestOverLstsqSolver(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    CONV_RATE = -0.3

    def run_inconsistent(self, ath: AlgTestHelper,
                         ols: rlsq.OverLstsqSolver,
                         alg_tol, iter_lim,
                         test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = ols(ath.A, ath.b, 0.0, alg_tol, iter_lim, rng)
            ath.test_residual_proj(test_tol)
            ath.test_x_angle(test_tol)
            ath.test_x_norm(test_tol)

    def run_consistent(self, ath: AlgTestHelper,
                       ols: rlsq.OverLstsqSolver,
                       alg_tol, iter_lim,
                       test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = ols(ath.A, ath.b, 0.0, alg_tol, iter_lim, rng)
            ath.test_x_norm(test_tol)
            ath.test_x_angle(test_tol)
            ath.test_objective(test_tol)

    def _test_convergence_rate(self, ath, alg, ridge=False):
        rng = np.random.default_rng(34998751340)
        delta = 0.25 if ridge else 0.0
        ath.tester = self
        max_iter = 5 * ath.A.shape[1]
        x, log = alg(ath.A, ath.b, delta, 1e-12, max_iter, rng, logging=True)
        fit, r2 = ustats.loglinear_fit(np.arange(log.errors.size-1),
                                       log.errors[1:])
        self.assertGreaterEqual(r2, 0.95)  # linear convergence
        self.assertLess(fit[1], self.CONV_RATE)  # decay faster than \exp(-0.3 t)
        self.assertLessEqual(log.errors[-1], log.errors[0]*1e-10)
        if ridge:
            m, n = ath.A.shape
            scaled_I = delta**0.5 * np.eye(n)
            ath.x_opt = la.lstsq(np.vstack((ath.A, scaled_I)),
                                 np.hstack((ath.b, np.zeros(n))))[0]
            ath.x_approx = x
            ath.test_delta_x(1e-6)
        pass


class TestSPO(TestOverLstsqSolver):
    """
    Test SPO objects

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * n)-by-m.
        (3) the way they factor the sketch of A.
    """

    def test_srct_qr(self):
        sap = rlsq.SPO(oblivious.SkOpTC(), sampling_factor=2)
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_gaussian_qr(self):
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=2)
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_sjlt_qr(self):
        sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=8), sampling_factor=2)
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_consistent_tall_qr(self):
        ath = consistent_tall()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1)
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square_qr(self):
        ath = consistent_square()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1)
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_inconsistent_orth_qr(self):
        ath = inconsistent_orthog()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen_qr(self):
        ath = inconsistent_gen()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_srct_chol(self):
        sap = rlsq.SPO(oblivious.SkOpTC(), sampling_factor=2, mode='chol')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_gaussian_chol(self):
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=2, mode='chol')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_sjlt_chol(self):
        sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=8), sampling_factor=2, mode='chol')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_consistent_tall_chol(self):
        ath = consistent_tall()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='chol')
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square_chol(self):
        ath = consistent_square()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='chol')
        self.run_consistent(ath, sap, 0.0, 1, 1e-10, self.SEEDS)
        # ^ Slightly lower tolerance than consistent_tall_chol

    def test_inconsistent_orth_chol(self):
        ath = inconsistent_orthog()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='chol')
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen_chol(self):
        ath = inconsistent_gen()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='chol')
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_srct_svd(self):
        sap = rlsq.SPO(oblivious.SkOpTC(), sampling_factor=2, mode='svd')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_gaussian_svd(self):
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=2, mode='svd')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_sjlt_svd(self):
        sap = rlsq.SPO(oblivious.SkOpSJ(vec_nnz=8), sampling_factor=2, mode='svd')
        ath = inconsistent_gen()
        self._test_convergence_rate(ath, sap, ridge=False)
        self._test_convergence_rate(ath, sap, ridge=True)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_consistent_tall_svd(self):
        ath = consistent_tall()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='svd')
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square_svd(self):
        ath = consistent_square()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='svd')
        self.run_consistent(ath, sap, 0.0, 1, 1e-10, self.SEEDS)
        # ^ Slightly lower tolerance than consistent_tall_chol

    def test_inconsistent_orth_svd(self):
        ath = inconsistent_orthog()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen_svd(self):
        ath = inconsistent_gen()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_consistent_lowrank_svd(self):
        ath = consistent_lowrank()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        self.run_consistent(ath, sap, 1-12, 1, 1e-6, self.SEEDS)


class TestSPO_NS(TestOverLstsqSolver):
    # Sketch-and-precondition, using no-refresh Newton sketch for iterative method

    def test_gaussian_svd(self):
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        sap.iterative_solver = PcSS3()
        ath = inconsistent_gen()
        self.CONV_RATE = -0.1
        self._test_convergence_rate(ath, sap, ridge=False)
        ath = inconsistent_stackid()
        self._test_convergence_rate(ath, sap, ridge=False)
        pass

    def test_consistent_tall_svd(self):
        ath = consistent_tall()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='svd')
        sap.iterative_solver = PcSS3()
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square_svd(self):
        ath = consistent_square()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=1, mode='svd')
        sap.iterative_solver = PcSS3()
        self.run_consistent(ath, sap, 0.0, 1, 1e-10, self.SEEDS)

    def test_inconsistent_orth_svd(self):
        ath = inconsistent_orthog()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        sap.iterative_solver = PcSS3()
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen_svd(self):
        ath = inconsistent_gen()
        self.CONV_RATE = -0.1
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        sap.iterative_solver = PcSS3()
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_consistent_lowrank_svd(self):
        ath = consistent_lowrank()
        sap = rlsq.SPO(oblivious.SkOpGA(), sampling_factor=3, mode='svd')
        sap.iterative_solver = PcSS3()
        self.run_consistent(ath, sap, 1e-12, 1, 1e-6, self.SEEDS)


class TestSAS(unittest.TestCase):
    """
    Test SAS objects (which implement sketch-and-solve for least squares).

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * m)-by-m.
        (3) the LAPACK driver they use for the sketched problem. All tests
            here let SciPy make that choice.
    """

    SEEDS = [89349756478, 4838934874, 9834789347]

    def test_convergence_rate_gaussian(self):
        for seed in TestSAS.SEEDS:
            sas = rlsq.SSO1(oblivious.SkOpGA(), np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_srct(self):
        for seed in TestSAS.SEEDS:
            sas = rlsq.SSO1(oblivious.SkOpTC(), np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_sjlt(self):
        for seed in TestSAS.SEEDS:
            sas = rlsq.SSO1(oblivious.SkOpSJ(vec_nnz=8), np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def _test_convergence_rate(self, sas, seed):
        n_rows, n_cols = 1000, 50  # least 10x more rows than cols
        rng = np.random.default_rng(seed)
        A = matmakers.simple_mat(n_rows, n_cols, 5, rng)
        x0 = np.random.randn(n_cols)
        b0 = A @ x0
        b = b0 + 0.05 * rng.standard_normal(n_rows)
        x_star = la.lstsq(A, b)[0]
        errors = []
        sampling_factors = np.arange(start=1, stop=10, step=10 / n_cols)
        for sf in sampling_factors:
            sas.sampling_factor = sf
            rng = np.random.default_rng(seed)
            x_ske = sas(A, b, 0.0, tol=np.NaN, iter_lim=1, rng=rng)[0]
            err = la.norm(x_ske - x_star)
            errors.append(err)
        errors = np.array(errors)
        coeffs, r2 = ustats.loglog_fit(sampling_factors, errors)
        self.assertLessEqual(coeffs[1], -0.5)  # at least 1/sqrt(d)
        self.assertGreaterEqual(r2, 0.7)
        pass
