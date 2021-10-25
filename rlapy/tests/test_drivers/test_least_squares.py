import unittest
import numpy as np
import scipy.linalg as la
import rlapy.drivers.least_squares as rlsq
import rlapy.utils.sketching as usk
import rlapy.utils.stats as ustats
import rlapy.tests.matmakers as matmakers


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
    n, m = 1000, 100
    num_hi = 30
    num_lo = m - num_hi
    # Make A
    hi_spec = 1e5 * np.ones(num_hi) + rng.random(num_hi)
    lo_spec = np.ones(num_lo) + rng.random(num_lo)
    spec = np.concatenate([hi_spec, lo_spec])
    U = usk.orthonormal_operator(n, m, rng)
    Vt = usk.orthonormal_operator(m, m, rng)
    A = (U * spec) @ Vt
    # Make b
    hi_x = rng.standard_normal(num_hi)/1e5
    lo_x = rng.standard_normal(num_lo)
    x = np.concatenate([hi_x, lo_x])
    b_orth = rng.standard_normal(n) * 1e2
    b_orth -= U @ (U.T @ b_orth)  # orthogonal to range(A)
    b = A @ x + b_orth
    # Return
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
        self.x_approx = None
        self.U = U
        self.s = s  # shouldn't have anything below datatype's "eps".
        self.Vt = Vt
        self.tester = None  # unittest.TestCase

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
        norm_opt = la.norm(self.Vt @ self.x_approx)
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
        nrm = la.norm(residual_proj, ord=2) / np.linalg.norm(residual)
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

    def run_inconsistent(self, ath: AlgTestHelper,
                         ols: rlsq.OverLstsqSolver,
                         alg_tol, iter_lim,
                         test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.x_approx = ols(ath.A, ath.b, alg_tol, iter_lim, rng)
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
            ath.x_approx = ols(ath.A, ath.b, alg_tol, iter_lim, rng)
            ath.test_x_norm(test_tol)
            ath.test_x_angle(test_tol)
            ath.test_objective(test_tol)

    def run_batch_sap(self, sap, tol, iter_lim, tolfac1, tolfac2):
        """
        For several random least-squares problems (data (A,b)), compute
            x = sap(A, b, tol, iter_lim, rng)
        and
            x_opt = np.linalg.lstsq(A, b)[0].

        Check that
            ||x - x_opt|| <= tolfac1 * tol
        and check that
            mean(||x - x_opt|| for all random (A, b)) <= tolfac2 * tol.

        You should have tolfac1 > tolfac2 > 1.
        """
        n_rows, n_cols = 2000, 200
        errors = np.zeros(len(self.SEEDS))
        for i, seed in enumerate(self.SEEDS):
            rng = np.random.default_rng(seed)
            A = matmakers.simple_mat(n_rows, n_cols, scale=5, rng=rng)
            x0 = np.random.randn(n_cols)
            b0 = A @ x0
            b = b0 + 0.05 * rng.standard_normal(n_rows)
            x_approx = sap(A, b, tol=tol, iter_lim=iter_lim, rng=rng)
            x_opt = np.linalg.lstsq(A, b, rcond=None)[0]
            error = np.linalg.norm(x_approx - x_opt)
            errors[i] = error
            self.assertLessEqual(error, tol * tolfac1)
        mean_error = np.mean(errors)
        self.assertLessEqual(mean_error, tol * tolfac2)

    pass


class TestSAP1(TestOverLstsqSolver):
    """
    Test SAP1 objects (sketch-and-precondition based on QR).

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * n)-by-m.
    """

    def test_srct(self):
        sap = rlsq.SAP1(usk.srct_operator, sampling_factor=3)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_gaussian(self):
        sap = rlsq.SAP1(usk.gaussian_operator, sampling_factor=2)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_sjlt(self):
        sap = rlsq.SAP1(usk.sjlt_operator, sampling_factor=3)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_consistent_tall(self):
        ath = consistent_tall()
        sap = rlsq.SAP1(usk.gaussian_operator, sampling_factor=1)
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square(self):
        ath = consistent_square()
        sap = rlsq.SAP1(usk.gaussian_operator, sampling_factor=1)
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_inconsistent_orth(self):
        ath = inconsistent_orthog()
        sap = rlsq.SAP1(usk.gaussian_operator, sampling_factor=3)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen(self):
        ath = inconsistent_gen()
        sap = rlsq.SAP1(usk.gaussian_operator, sampling_factor=3)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)


class TestSAP2(TestOverLstsqSolver):
    """
    Test SAP2 objects (sketch-and-precondition based on SVD).

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * n)-by-m.
        (3) a parameter "smart_init", which determines whether the
            algorithm tries to initialize its iterative solver at the 
            result given by sketch-and-solve.
    """

    def test_srct(self):
        sap = rlsq.SAP2(usk.srct_operator,
                        sampling_factor=3, smart_init=True)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_gaussian(self):
        sap = rlsq.SAP2(usk.gaussian_operator,
                        sampling_factor=3, smart_init=True)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)
        sap.smart_init = False
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)

    def test_sjlt(self):
        sap = rlsq.SAP2(usk.sjlt_operator,
                        sampling_factor=3, smart_init=True)
        self.run_batch_sap(sap, 1e-8, 40, 100.0, 10.0)

    def test_consistent_tall(self):
        ath = consistent_tall()
        sap = rlsq.SAP2(usk.gaussian_operator, 3, smart_init=False)
        self.run_consistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)
        sap.sampling_factor = 1
        sap.smart_init = True
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_lowrank(self):
        ath = consistent_lowrank()
        sap = rlsq.SAP2(usk.gaussian_operator, 3, smart_init=False)
        self.run_consistent(ath, sap, 0.0, 100, 1e-6, self.SEEDS)
        sap.sampling_factor = 1
        sap.smart_init = True
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_consistent_square(self):
        ath = consistent_square()
        sap = rlsq.SAP2(usk.gaussian_operator, 1, smart_init=False)
        self.run_consistent(ath, sap, 0.0, 100, 1e-6, self.SEEDS)
        sap.sampling_factor = 1
        sap.smart_init = True
        self.run_consistent(ath, sap, 0.0, 1, 1e-12, self.SEEDS)

    def test_inconsistent_orth(self):
        ath = inconsistent_orthog()
        sap = rlsq.SAP2(usk.gaussian_operator, 3, smart_init=False)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)
        sap.smart_init = True
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)

    def test_inconsistent_gen(self):
        ath = inconsistent_gen()
        sap = rlsq.SAP2(usk.gaussian_operator, 3, smart_init=False)
        self.run_inconsistent(ath, sap, 1e-12, 100, 1e-6, self.SEEDS)
        sap.smart_init = True
        self.run_inconsistent(ath, sap, 1e-12, 50, 1e-6, self.SEEDS)


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
            sas = rlsq.SAS1(usk.gaussian_operator, np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_srct(self):
        for seed in TestSAS.SEEDS:
            sas = rlsq.SAS1(usk.srct_operator, np.NaN)
            self._test_convergence_rate(sas, seed)
        pass

    def test_convergence_rate_sjlt(self):
        for seed in TestSAS.SEEDS:
            sas = rlsq.SAS1(usk.sjlt_operator, np.NaN)
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
            x_ske = sas(A, b, tol=np.NaN, iter_lim=1, rng=rng)
            err = la.norm(x_ske - x_star)
            errors.append(err)
        errors = np.array(errors)
        coeffs, r2 = ustats.loglog_fit(sampling_factors, errors)
        self.assertLessEqual(coeffs[1], -0.5)  # at least 1/sqrt(d)
        self.assertGreaterEqual(r2, 0.7)
        pass
