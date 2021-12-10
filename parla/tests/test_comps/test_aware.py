import unittest
import parla.comps.sketchers.aware as aware
import numpy as np
import scipy.linalg as la
import parla.comps.sketchers.oblivious as skob
import parla.utils.linalg_wrappers as ulaw
import parla.utils.stats as ustats
import parla.tests.matmakers as matmakers

np.set_printoptions(precision=4, linewidth=100)


class TestPRSO1(unittest.TestCase):
    """
    From the RS1 documentation ...

        RS1 objects are used to create n-by-k matrices S for use in sketching
        the rows of an m-by-n matrix A. The qualitative goal is that the range
        of S should be well-aligned with the top-k right singular vectors of A.

        PRSO objects work by applying a power method that starts with an initial
        random matrix with k columns, and then makes alternating applications of
        A and A.T. The tuning parameters in this procedure are:

            How we generate the initial random matrix.
            The number of passes we allow over A (or A.T).
            How we stabilize the power method. E.g., QR or LU factorization.
            How often we stabilize the power method.
    """

    def test_max_eig_orth(self):
        ps = aware.RS1(sketch_op_gen=skob.SkOpGA(),
                       num_pass=np.NaN,  # We'll set this later.
                       stabilizer=ulaw.orth,
                       passes_per_stab=1)
        self._test_max_eig(ps)

    def test_max_eig_lu(self):
        ps = aware.RS1(sketch_op_gen=skob.SkOpGA(),
                       num_pass=np.NaN,  # We'll set this later.
                       stabilizer=ulaw.lu_stabilize,
                       passes_per_stab=1)
        self._test_max_eig(ps)

    def _test_max_eig(self, ps):
        """
        Use a PSO1 object "ps" to the implement the power method for finding
        the largest eigenpair of a Gram matrix M = (A.T @ A).

        The number of steps of the power method is determined by ps.num_pass.
        Calling vec = ps(A, 1, rng) returns a vector that's (essentially)
        the output of the power method on (A.T @ A) with ps.num_pass/2 steps.
        """
        rng = np.random.default_rng(0)
        rows = np.arange(start=20, stop=50, step=10).astype(int)
        cols = np.arange(start=10, stop=20, step=2).astype(int)
        spectrum = np.exp(-np.arange(0, 10))
        # ^ We'll run tests on matrices with varying numbers of rows and
        #   columns. The matrices will always be rank 10, with exponentially
        #   decaying singular values. The max singular value will always be 1.
        fits = {
            'val': {'rates': [], 'r2s': []},
            'vec': {'rates': [], 'r2s': []}
        }
        passes = np.arange(start=1, stop=16, step=3)
        # ^ We'll look at the convergence behavior of the algorithm as we
        #   allow different numbers of passes over A and A.T.
        for n_rows in rows:
            for n_cols in cols:
                A = matmakers.rand_low_rank(n_rows, n_cols, spectrum, rng)
                eigval_gaps = np.zeros(passes.size)
                eigvec_gaps = np.zeros(passes.size)
                for i, num_pass in enumerate(passes):
                    rng = np.random.default_rng(1)
                    ps.num_pass = num_pass
                    vec = ps(A, 1, rng)
                    vec /= la.norm(vec, ord=2)
                    ATAvec = A.T @ (A @ vec)
                    eigval_gaps[i] = 1 - la.norm(ATAvec, ord=2)
                    eigvec_gaps[i] = la.norm(ATAvec - vec, ord=2)
                # Next, we check for linear convergence of error in the
                # approximation of the top eigenpair. The specific test
                # looks at the r-squared of a log-linear fit of "passes"
                # against eigval_gaps and eigvec_gaps.
                coeffs, r2 = ustats.loglinear_fit(passes, eigval_gaps)
                fits['val']['rates'].append(coeffs[1])
                fits['val']['r2s'].append(r2)
                self.assertGreaterEqual(r2, 0.7)
                self.assertLessEqual(coeffs[1], -0.5)
                coeffs, r2 = ustats.loglinear_fit(passes, eigvec_gaps)
                fits['vec']['rates'].append(coeffs[1])
                fits['vec']['r2s'].append(r2)
                self.assertGreaterEqual(r2, 0.7)
                self.assertLessEqual(coeffs[1], -0.5)
        # Finally, we run some tests that check performance
        # in expectation. We can place more stringent demands
        # on convergence rates in this aggregate test.
        mean_val_r2s = np.mean(fits['val']['r2s'])
        self.assertGreaterEqual(mean_val_r2s, 0.8)
        mean_vec_r2s = np.mean(fits['vec']['r2s'])
        self.assertGreaterEqual(mean_vec_r2s, 0.8)
        mean_val_rates = np.mean(fits['val']['rates'])
        self.assertLessEqual(mean_val_rates, -0.7)
        mean_vec_rates = np.mean(fits['vec']['rates'])
        self.assertLessEqual(mean_vec_rates, -0.7)
        pass
