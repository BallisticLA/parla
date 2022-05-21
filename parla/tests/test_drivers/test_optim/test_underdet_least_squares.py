import unittest
import numpy as np
import scipy.linalg as la
from parla.drivers.least_squares import UnderLstsqSolver, SPU1
import parla.utils.sketching as usk
import parla.comps.sketchers.oblivious as oblivious


def make_simple_prob(m, n, spectrum, rng):
    rng = np.random.default_rng(rng)

    rank = spectrum.size
    U = usk.orthonormal_operator(m, rank, rng)
    Vt = usk.orthonormal_operator(rank, n, rng)
    A = (U * spectrum) @ Vt

    y0 = rng.standard_normal(m)
    c = A.T @ y0
    y_opt = U @ ((1/spectrum) * (Vt @ c))
    ath = AlgTestHelper(A, spectrum, c, y_opt)
    return ath


class AlgTestHelper:

    def __init__(self, A, spectrum, c, y_opt):
        """
        y_opt solves

            min ||y||
            s.t. A' y = c
        """
        self.A = A
        self.spectrum = spectrum
        self.c = c
        self.y_opt = y_opt
        self.y_approx = None
        self._result = None
        self.tester = unittest.TestCase()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
        self.y_approx = value[0]

    def test_delta_y(self, tol):
        """
        ||y - y_opt|| <= tol
        """
        delta_y = self.y_opt - self.y_approx
        nrm = la.norm(delta_y, ord=2) / (1 + min(la.norm(self.y_opt),
                                                 la.norm(self.y_approx)))
        self.tester.assertLessEqual(nrm, tol)

    def test_residual(self, tol):
        """
        ||A' y - c|| <= tol
        """
        res = self.A.T @ self.y_approx - self.c
        nrm = la.norm(res, ord=2) / self.spectrum[0]
        self.tester.assertLessEqual(nrm, tol)

    def test_objective(self, tol):
        """
        ||y|| <= ||y_opt|| + tol
        """
        nrm_approx = la.norm(self.y_approx, ord=2)
        nrm_opt = la.norm(self.y_opt, ord=2)
        rel_tol = tol * self.spectrum[0]
        self.tester.assertLessEqual(nrm_approx, nrm_opt + rel_tol)


class TestUnderLstsqSolver(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    STD_TEST_TOL = 1e-10

    STRICT_TEST_TOL = 1e-10

    def run_ath(self, ath: AlgTestHelper,
                      uls: UnderLstsqSolver,
                      alg_tol, iter_lim,
                      test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = uls(ath.A, ath.c, alg_tol, iter_lim, rng)
            ath.test_residual(test_tol)
            ath.test_objective(test_tol)
            ath.test_delta_y(test_tol)

    def run_ath_lowrank(self, ath: AlgTestHelper,
                              uls: UnderLstsqSolver,
                              alg_tol, iter_lim,
                              test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = uls(ath.A, ath.c, alg_tol, iter_lim, rng)
            # skip the check for ||y_opt - y_approx||
            ath.test_residual(test_tol)
            ath.test_objective(test_tol)


class TestSPU1(TestUnderLstsqSolver):

    @staticmethod
    def default_config():
        alg = SPU1(
            sketch_op_gen=oblivious.SkOpSJ(),
            sampling_factor=3
        )
        return alg

    def test_linspace_spec(self):
        alg = TestSPU1.default_config()
        rng = np.random.default_rng(0)

        m, n, cond_num = 1000, 100, 1e5
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        ath = make_simple_prob(m, n, spectrum, rng)
        self.run_ath(ath, alg, 1e-12, 50, self.STD_TEST_TOL, self.SEEDS)

    def test_logspace_spec(self):
        alg = TestSPU1.default_config()
        rng = np.random.default_rng(0)

        m, n, cond_num = 1000, 100, 1e5
        spec = np.logspace(np.log10(cond_num) / 2, -np.log10(cond_num) / 2, num=n)

        ath = make_simple_prob(m, n, spec, rng)
        self.run_ath(ath, alg, 1e-12, 50, self.STD_TEST_TOL, self.SEEDS)

    def test_higher_accuracy(self):
        alg = TestSPU1.default_config()
        rng = np.random.default_rng(0)

        m, n, cond_num = 500, 50, 1e5
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        ath = make_simple_prob(m, n, spectrum, rng)
        self.run_ath(ath, alg, 0.0, n, self.STRICT_TEST_TOL, self.SEEDS)

    def test_lowrank_linspace_spec(self):
        alg = TestSPU1.default_config()
        rng = np.random.default_rng(0)

        m, n, rank, cond_num = 1000, 100, 80, 1e5
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=rank)

        ath = make_simple_prob(m, n, spectrum, rng)
        self.run_ath_lowrank(ath, alg, 1e-12, 50, self.STD_TEST_TOL, self.SEEDS)
