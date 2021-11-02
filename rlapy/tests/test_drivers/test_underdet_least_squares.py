import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.least_squares import UnderLstsqSolver, SPU1
import rlapy.utils.sketching as usk
import rlapy.comps.sketchers.oblivious as oblivious


def make_simple_prob(m, n, spectrum, rng):
    rng = np.random.default_rng(rng)

    # Construct the data matrix
    rank = spectrum.size
    U = usk.orthonormal_operator(m, rank, rng)
    Vt = usk.orthonormal_operator(rank, n, rng)
    A = (U * spectrum) @ Vt

    # Make c
    c = rng.standard_normal(n)

    # solve for x_opt, y_opt
    y_opt = la.lstsq(A.T, c, check_finite=False)[0]
    #gram = A.T @ A
    #rhs = - c
    #x_opt = la.solve(gram, rhs, sym_pos=True)
    #y_opt = - A @ x_opt

    # Return
    ath = AlgTestHelper(A, c, y_opt)
    return ath


class AlgTestHelper:

    def __init__(self, A, c, y_opt):
        """
        y_opt solves

            min ||y||
            s.t. A' y = c
        """
        self.A = A
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
        nrm = la.norm(res, ord=2)
        self.tester.assertLessEqual(nrm, tol)

    def test_objective(self, tol):
        """
        ||y|| <= ||y_opt|| + tol
        """
        nrm_approx = la.norm(self.y_approx, ord=2)
        nrm_opt = la.norm(self.y_opt, ord=2)
        self.tester.assertLessEqual(nrm_approx, nrm_opt + tol)


class TestUnderLstsqSolver(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

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


class TestSPU1(TestUnderLstsqSolver):

    def test_simple(self):
        alg = SPU1(oblivious.SkOpSJ(), sampling_factor=3)

        rng = np.random.default_rng(0)
        m, n = 1000, 100
        kappa = 1e5
        spectrum = np.linspace(kappa ** 0.5, kappa ** -0.5, num=n)
        ath = make_simple_prob(m, n, spectrum, rng)

        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)
