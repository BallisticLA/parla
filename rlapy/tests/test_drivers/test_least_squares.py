import unittest
import numpy as np
import scipy.linalg as la
import rlapy.drivers.least_squares as rlsq
import rlapy.utils.sketching as usk
import rlapy.utils.stats as ustats
import rlapy.tests.matmakers as matmakers


def simple_mat(n_rows, n_cols, scale, rng):
    rng = np.random.default_rng(rng)
    A = rng.normal(0, 1, (n_rows, n_cols))
    QA, RA = np.linalg.qr(A)
    damp = 1 / np.sqrt(1 + scale * np.arange(n_cols))
    RA *= damp
    A_bad = QA @ RA
    return A_bad


# tall consistent system, no special structure.
def ols_case_0():
    seed = 190489290
    rng = np.random.default_rng(seed)
    m, n = 100, 10
    A = simple_mat(m, n, scale=1, rng=rng)
    x = np.random.randn(n)
    b = A @ x
    data = dict()
    return A, b, data


# tall consistent system, rank deficient
def ols_case_1():
    seed = 8923890298
    rng = np.random.default_rng(seed)
    m, n, rank = 100, 10, 5
    A = matmakers.rand_low_rank(m, n, rank, rng=rng)
    x = np.random.randn(n)
    b = A @ x
    data = dict()
    return A, b, data


# square nonsingular system
def ols_case_2():
    seed = 3278992245
    rng = np.random.default_rng(seed)
    n = 10
    A = simple_mat(n, n, scale=100, rng=rng)
    x = np.random.randn(n)
    b = A @ x
    data = dict()
    return A, b, data


# b orthog to range(A), A full rank
def ols_case_3():
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
    data = dict()
    return A, b, data


# adversarial config
def ols_case_4():
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
    # b *= (m / la.norm(b))
    # Return
    ath = AlgTestHelper(A, b, x, U, spec, Vt)
    return ath


class AlgTestHelper:

    def __init__(self, A, b, x_opt, U, s, Vt):
        self.A = A
        self.b = b
        self.x_opt = x_opt
        self.x_approx = None
        self.U = U
        self.s = s
        self.Vt = Vt
        self.tester = unittest.TestCase()

    def test_x_angle(self, tol):
        """
        x' x_opt >= (1 - tol)*||x|| ||x_opt||
        """
        y = self.Vt @ self.x_approx
        y /= la.norm(y)
        y_opt = self.Vt @ self.x_opt
        y_opt /= la.norm(y_opt)
        cosine = np.dot(y, y_opt)
        self.tester.assertGreaterEqual(cosine, 1 - tol)

    def test_x_norm(self, tol):
        """
        (1 - tol)*||x_opt|| <= ||x|| <= (1+tol)*||x_opt||
        """
        norm = la.norm(self.Vt @ self.x_approx)
        norm_opt = la.norm(self.Vt @ self.x_approx)
        self.tester.assertLessEqual(norm, (1+tol)*norm_opt)
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
        || U U' (A x - b) || / ||A x - b|| <= 10^(-places)
        """
        # This test is probably better scaled than the normal equations
        residual = self.A @ self.x_approx - self.b
        residual_proj = self.U @ (self.U.T @ residual)
        nrm = la.norm(residual_proj, ord=2) / np.linalg.norm(residual)
        self.tester.assertLessEqual(nrm, tol)

    def test_objective(self, tol):
        """ Use this in the presence of multiple optimal solutions
        ||A x - b|| <= ||A x_opt - b|| + tol
        """
        res_approx = self.b - self.A @ self.x_approx
        res_opt = self.b - self.A @ self.x_opt
        nrm_approx = la.norm(res_approx, ord=2)
        nrm_opt = la.norm(res_opt, ord=2)
        self.tester.assertLessEqual(nrm_approx, nrm_opt + tol)

    def test_normal_eqs(self, tol):
        """ Use this when dealing with multiple optimal solutions
        || A' A x - A' b|| <= 10^(-places)
        """
        gap = self.A.T @ self.b - self.A.T @ (self.A @ self.x_approx)
        nrm = la.norm(gap, ord=2)
        self.tester.assertLessEqual(nrm, tol)


def naive_run_lstsq(seed, ols, tol, iter_lim):
    """
    Generate random 2000-by-200 A and random b, call
        x_approx = ols.exec(A, b, tol, iter_lim, rng)
    return
        ||x_approx - x_opt||
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 2000, 200
    A = simple_mat(n_rows, n_cols, scale=5, rng=rng)
    x0 = np.random.randn(n_cols)
    b0 = A @ x0
    b = b0 + 0.05 * rng.standard_normal(n_rows)
    x_approx = ols.exec(A, b, tol=tol, iter_lim=iter_lim, rng=rng)
    x_opt = np.linalg.lstsq(A, b, rcond=None)[0]
    error = np.linalg.norm(x_approx - x_opt)
    return error


class TestOverLstsqSolver(unittest.TestCase):

    def run_core_ols_case(self, ath: AlgTestHelper,
                          sas: rlsq.OverLstsqSolver,
                          alg_tol, iter_lim,
                          test_tol, rng):
        rng = np.random.default_rng(rng)
        ath.x_approx = sas.exec(ath.A, ath.b, alg_tol, iter_lim, rng)
        ath.test_residual_proj(test_tol)
        ath.test_x_angle(test_tol)
        ath.test_x_norm(test_tol)

    pass


class TestSAPs(TestOverLstsqSolver):
    """
    Test least square solvers which have full control over error tolerance.
    Right now, those are only sketch-and-precondition based methods.
    """

    SEEDS = [1, 4, 15, 31, 42]

    def _run_batch_lstsq(self, sap, tol, iter_lim, tolfac1, tolfac2):
        """
        For several random least-squares problems (data (A,b)), compute
            x = sap.exec(A, b, tol, iter_lim, rng)
        and
            x_opt = np.linalg.lstsq(A, b)[0].

        Check that
            ||x - x_opt|| <= tolfac1 * tol
        and check that
            mean(||x - x_opt|| for all random (A, b)) <= tolfac2 * tol.

        You should have tolfac1 > tolfac2 > 1.
        """
        errors = np.zeros(len(TestSAPs.SEEDS))
        for i, seed in enumerate(TestSAPs.SEEDS):
            error = naive_run_lstsq(seed, sap, tol, iter_lim)
            errors[i] = error
            self.assertLessEqual(error, tol * tolfac1)
        mean_error = np.mean(errors)
        self.assertLessEqual(mean_error, tol * tolfac2)

    """
    Test SAP1 objects (sketch-and-precondition based on QR).

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * m)-by-m.
    """

    def test_batch_sap1_srct(self):
        sap = rlsq.SAP1(usk.srct_operator,
                            sampling_factor=3)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_batch_sap1_gaussian(self):
        sap = rlsq.SAP1(usk.gaussian_operator,
                            sampling_factor=2)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_batch_sap1_sjlt(self):
        sap = rlsq.SAP1(usk.sjlt_operator,
                            sampling_factor=3)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_core_case_4(self):
        sap = rlsq.SAP1(usk.gaussian_operator,
                            sampling_factor=3)
        ath = ols_case_4()
        for idx, seed in enumerate(self.SEEDS):
            rng = np.random.default_rng(seed)
            self.run_core_ols_case(ath, sap, 1e-8, 100, 1e-6, rng)

    """
    Test SAP2 objects (sketch-and-precondition based on SVD).

    These objects are characterized by
        (1) the method they use to generate the sketching operator,
        (2) a parameter "sampling_factor", where for an m-by-n matrix
            A, its sketching operator S is of shape (sampling_factor * m)-by-m.
        (3) a parameter "smart_init", which determines whether the
            algorithm tries to initialize its iterative solver at the 
            result given by sketch-and-solve.
    """

    def test_batch_sap2_srct(self):
        sap = rlsq.SAP2(usk.srct_operator,
                            sampling_factor=3,
                            smart_init=True)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)
        pass

    def test_batch_sap2_gaussian(self):
        sap = rlsq.SAP2(usk.gaussian_operator,
                            sampling_factor=3,
                            smart_init=True)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)
        sap.smart_init = False
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)

    def test_batch_sap2_sjlt(self):
        sap = rlsq.SAP2(usk.sjlt_operator,
                            sampling_factor=3,
                            smart_init=True)
        self._run_batch_lstsq(sap, 1e-8, 40, 100.0, 10.0)


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
        A = simple_mat(n_rows, n_cols, 5, rng)
        x0 = np.random.randn(n_cols)
        b0 = A @ x0
        b = b0 + 0.05 * rng.standard_normal(n_rows)
        x_star = la.lstsq(A, b)[0]
        errors = []
        sampling_factors = np.arange(start=1, stop=10, step=10 / n_cols)
        for sf in sampling_factors:
            sas.sampling_factor = sf
            rng = np.random.default_rng(seed)
            x_ske = sas.exec(A, b, tol=0, iter_lim=1, rng=rng)
            err = la.norm(x_ske - x_star)
            errors.append(err)
        errors = np.array(errors)
        coeffs, r2 = ustats.loglog_fit(sampling_factors, errors)
        self.assertLessEqual(coeffs[1], -0.5)  # at least 1/sqrt(d)
        self.assertGreaterEqual(r2, 0.7)
        pass
