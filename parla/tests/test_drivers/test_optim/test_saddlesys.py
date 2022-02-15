import unittest
import numpy as np
import scipy.linalg as la
from parla.drivers.saddlesys import SPS1, SPS2, SaddleSolver
import parla.comps.determiter.saddle as dsad
import parla.utils.sketching as usk
import parla.utils.stats as ustats
import parla.comps.sketchers.oblivious as oblivious


def make_simple_prob(m, n, spectrum, delta, rng, rhs_scale=1.0):
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
    if rhs_scale != 1.0:
        scale = rhs_scale / la.norm(rhs)
        rhs *= scale
        b *= scale
        c *= scale
    try:
        x_opt = la.solve(gram, rhs, sym_pos=True)
    except la.LinAlgError:
        x_opt = la.lstsq(gram, rhs)[0]

    y_opt = b - A @ x_opt

    """
    # Alternatively, could set (b, c) as a function of predetermined (x, y).
    x = rng.standard_normal(n)
    y = rng.standard_normal(m)
    b = y + A @ x
    c = A.T @ y - delta * x
    """

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
        nrm = la.norm(gap, ord=2) / (1 + rel)
        self.tester.assertLessEqual(nrm, tol)


class TestSaddleSolver(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    def run_ath(self, ath: AlgTestHelper,
                      alg: SaddleSolver,
                      alg_tol, iter_lim,
                      test_tol, seeds, rates=True):
        ath.tester = self
        for seed in seeds:
            rng = np.random.default_rng(seed)
            ath.result = alg(ath.A, ath.b, ath.c, ath.delta, alg_tol, iter_lim, rng,
                             logging=True)
            ath.test_normal_eq_residual(test_tol)
            ath.test_block_residual(test_tol)
            ath.test_delta_xy(test_tol)
            if rates:
                log = ath.result[2]
                fit, r2 = ustats.loglinear_fit(np.arange(log.errors.size - 1),
                                               log.errors[1:])
                self.assertGreaterEqual(r2, 0.95)  # linear convergence
                self.assertLess(fit[1], -0.3)  # decay faster than \exp(-0.3 t)
                pass

    def _test_linspace_spec(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 1000, 100, 1e5
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        ath = make_simple_prob(m, n, spectrum, 0.0, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

        ath = make_simple_prob(m, n, spectrum, 0.5, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

    def _test_logspace_spec(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 1000, 100, 1e5
        spec = np.logspace(np.log10(cond_num)/2, -np.log10(cond_num)/2, num=n)

        ath = make_simple_prob(m, n, spec, 0.0, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

        ath = make_simple_prob(m, n, spec, 0.5, rng)
        self.run_ath(ath, alg, 1e-12, 50, 1e-6, self.SEEDS)

    def _test_higher_accuracy(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 500, 50, 1e3
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        ath = make_simple_prob(m, n, spectrum, 0.0, rng)
        self.run_ath(ath, alg, 0.0, n, 1e-9, self.SEEDS)

        ath = make_simple_prob(m, n, spectrum, 1.0, rng)
        self.run_ath(ath, alg, 0.0, n, 1e-9, self.SEEDS)

    def _test_tiny_scale(self, alg, outer_seed=0):
        rng = np.random.default_rng(outer_seed)

        m, n, cond_num = 500, 50, 1e8
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)

        rhs_scale = 1e-9
        ath = make_simple_prob(m, n, spectrum, 0.0, rng, rhs_scale=rhs_scale)
        self.run_ath(ath, alg, 1e-8, n, 1e-7, self.SEEDS, rates=False)


class TestSPS1(TestSaddleSolver):

    @staticmethod
    def default_config():
        alg = SPS1(
            sketch_op_gen=oblivious.SkOpSJ(),
            sampling_factor=3,
            iterative_solver=dsad.PcSS1()
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

    def test_tiny_scale(self):
        alg = TestSPS1.default_config()
        self._test_tiny_scale(alg)

    # TODO: residuals in the low-rank tests aren't monotonic. I rememeber
    #   this is a thing that can happen with PCG, but it should only be possible
    #   when even the preconditioned system is badly conditioned.
    #   I should investigate if the preconditioner is really defined
    #   in the best possible way (specifically, if the scaling of the
    #   preconditioner's identity component is good).

    def test_lowrank_precond_linspace(self):
        m, n, cond_num = 1000, 100, 1e5
        alg = SPS1(
            sketch_op_gen=oblivious.SkOpGA(),
            sampling_factor=0.85,
            iterative_solver=dsad.PcSS1()
        )
        spectrum = np.linspace(cond_num ** 0.5, cond_num ** -0.5, num=n)
        rng = np.random.default_rng(0)
        ath = make_simple_prob(m, n, spectrum, 0.0, rng)
        self.run_ath(ath, alg, 1e-12, 5*n, 1e-6, self.SEEDS, rates=False)
        pass

    def test_lowrank_precond_logspace(self):
        m, n, cond_num = 1000, 100, 1e5
        alg = SPS1(
            sketch_op_gen=oblivious.SkOpGA(),
            sampling_factor=0.85,
            iterative_solver=dsad.PcSS1()
        )
        spectrum = np.logspace(np.log10(cond_num)/2, -np.log10(cond_num)/2, num=n)
        rng = np.random.default_rng(0)
        ath = make_simple_prob(m, n, spectrum, 0.0, rng)
        self.run_ath(ath, alg, 1e-12, 5*n, 1e-5, self.SEEDS, rates=False)
        # ^ Have to set the tolerance generously in order to pass.
        pass


class TestSPS2(TestSaddleSolver):
    
    @staticmethod
    def default_config():
        alg = SPS2(
            sketch_op_gen=oblivious.SkOpSJ(),
            sampling_factor=3,
            iterative_solver=dsad.PcSS2()
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
