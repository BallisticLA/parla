import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.saddlesys import SPS1, SPS2, SaddleSolver
import rlapy.comps.itersaddle as itersad
import rlapy.utils.sketching as usk
import rlapy.comps.sketchers.oblivious as oblivious


def make_simple_prob(m, n, spectrum, delta, rng):
    rng = np.random.default_rng(rng)

    # Construct the data matrix
    rank = spectrum.size
    U = usk.orthonormal_operator(m, rank, rng)
    Vt = usk.orthonormal_operator(rank, n, rng)
    A = (U * spectrum) @ Vt

    # Construct the right-hand-side
    prop_range = 0.7
    b0 = rng.standard_normal(m)
    b_range = U @ (U.T @ b0)
    b_orthog = b0 - b_range
    b_range *= (np.mean(spectrum) / la.norm(b_range))
    b_orthog *= (np.mean(spectrum) / la.norm(b_orthog))
    b = prop_range * b_range + (1 - prop_range) * b_orthog

    # Make c
    c = rng.standard_normal(n)

    # solve for x_opt, y_opt
    gram = A.T @ A + delta * np.eye(n)
    rhs = A.T @ b - c
    try:
        x_opt = la.solve(gram, rhs, sym_pos=True)
    except la.LinAlgError:
        x_opt = la.lstsq(gram, rhs)[0]

    y_opt = b - A @ x_opt

    # Return
    ath = AlgTestHelper(A, spectrum, b, c, delta, x_opt, y_opt)
    return ath


class AlgTestHelper:

    def __init__(self, A, spec, b, c, delta, x_opt, y_opt):
        """
        (x_opt, y_opt) solve ...

             [  I   |     A   ] [y_opt] = [b]
             [  A'  | -delta*I] [x_opt]   [c]

        the following characterization holds for x_opt ...

            (A' A + delta * I) x_opt = A'b - c.

        """
        self.A = A
        self.b = b
        self.c = c
        self.spec = spec  # singular values
        self.delta = delta
        self.x_opt = x_opt
        self.y_opt = y_opt
        self.x_approx = None
        self.y_approx = None
        self._result = None
        self.tester = unittest.TestCase()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value
        self.x_approx = value[0]
        self.y_approx = value[1]

    def test_delta_xy(self, tol):
        """
        ||x - x_opt|| <= tol   and   ||y - y_opt|| <= tol
        """
        delta_x = self.x_opt - self.x_approx
        nrm = la.norm(delta_x, ord=2) / (1 + min(la.norm(self.x_opt),
                                                 la.norm(self.x_approx)))
        self.tester.assertLessEqual(nrm, tol)

        delta_y = self.y_opt - self.y_approx
        nrm = la.norm(delta_y, ord=2) / (1 + min(la.norm(self.y_opt),
                                                 la.norm(self.y_approx)))
        self.tester.assertLessEqual(nrm, tol)

    def test_normal_eq_residual(self, tol):
        """
        || (A' A + delta*I) x - (A' b - c)|| <= tol
        """
        rhs = self.A.T @ self.b - self.c
        lhs = self.A.T @ (self.A @ self.x_approx)
        lhs += self.delta * self.x_approx
        gap = rhs - lhs
        nrm = la.norm(gap, ord=2)
        nrm /= np.max(self.spec)
        self.tester.assertLessEqual(nrm, tol)

    def test_block_residual(self, tol):
        """
          ||   [  I   |     A   ] [y] - [b]   ||   <=   tol
          ||   [ A.T  | -delta*I] [x]   [c]   ||
        """
        block1 = self.y_approx + self.A @ self.x_approx - self.b
        block2 = self.A.T @ self.y_approx - self.delta * self.x_approx - self.c
        gap = np.concatenate([block1, block2])
        rel = la.norm(np.hstack((self.b, self.c)))
        nrm = la.norm(gap, ord=2) / rel
        self.tester.assertLessEqual(nrm, tol)


class TestSaddleSolver(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    def run_ath(self, ath: AlgTestHelper,
                      alg: SaddleSolver,
                      alg_tol, iter_lim,
                      test_tol, seeds):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = alg(ath.A, ath.b, ath.c, ath.delta, alg_tol, iter_lim, rng)
            ath.test_normal_eq_residual(test_tol)
            ath.test_block_residual(test_tol)
            ath.test_delta_xy(test_tol)

    def _test_linspace_spec(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 1000, 100, 1e5
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        delta = 0.0
        ath = make_simple_prob(m, n, spectrum, delta, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

        delta = 1.0
        ath = make_simple_prob(m, n, spectrum, delta, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

    def _test_logspace_spec(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 1000, 100, 1e5
        spec = np.logspace(np.log10(cond_num)/2, -np.log10(cond_num)/2, num=n)

        ath = make_simple_prob(m, n, spec, 0.0, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

        ath = make_simple_prob(m, n, spec, 1.0, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

    def _test_higher_accuracy(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 500, 50, 1e3
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        ath = make_simple_prob(m, n, spectrum, 0.0, rng)
        self.run_ath(ath, alg, 0.0, n, 1e-10, self.SEEDS)

        ath = make_simple_prob(m, n, spectrum, 1.0, rng)
        self.run_ath(ath, alg, 0.0, n, 1e-10, self.SEEDS)


class TestSPS1(TestSaddleSolver):

    @staticmethod
    def default_config():
        alg = SPS1(
            sketch_op_gen=oblivious.SkOpSJ(),
            sampling_factor=3,
            iterative_solver=itersad.PcSS1()
        )
        return alg

    def test_linspace_spec(self):
        alg = TestSPS1.default_config()
        self._test_linspace_spec(alg)

    def test_logspace_spec(self):
        alg = TestSPS1.default_config()
        self._test_logspace_spec(alg)

    def test_higher_accuracy(self):
        alg = TestSPS1.default_config()
        self._test_higher_accuracy(alg)


class TestSPS2(TestSaddleSolver):
    #TODO: add tests that check logging
    
    @staticmethod
    def default_config():
        alg = SPS2(
            sketch_op_gen=oblivious.SkOpSJ(),
            sampling_factor=3,
            iterative_solver=itersad.PcSS2()
        )
        return alg

    def test_linspace_spec(self):
        alg = TestSPS2.default_config()
        self._test_linspace_spec(alg)

    def test_logspace_spec(self):
        alg = TestSPS2.default_config()
        self._test_logspace_spec(alg)

    def test_higher_accuracy(self):
        alg = TestSPS2.default_config()
        self._test_higher_accuracy(alg)
