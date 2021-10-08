import unittest
import numpy as np
import scipy.linalg as la
import rlapy.utils.sketching as usk
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.tests.matmakers as matmakers
import rlapy.comps.qb as rqb
import rlapy.comps.sketchers as rsks


def tall_low_exact_rank():
    m, n, k = 200, 50, 15
    rng = np.random.default_rng(89374539423)
    A, U, s, Vt = matmakers.rand_low_rank(m, n, k, rng, factors=True)
    ath = AlgTestHelper(A, U, s, Vt)
    return ath


def wide_low_exact_rank():
    m, n, k = 200, 50, 15  # n rows and m columns (in this function)
    rng = np.random.default_rng(89374539423)
    A, U, s, Vt = matmakers.rand_low_rank(n, m, k, rng, factors=True)
    ath = AlgTestHelper(A, U, s, Vt)
    return ath


class AlgTestHelper:
    # WARNING: this class makes no attempt to check internal
    # consistency between instance variables (e.g., it doesn't
    # enforce the expectation that (U, s, Vt) define an SVD of A).

    def __init__(self, A, U, s, Vt):
        self.A = A
        self.U = U
        self.s = s  # s should be sorted decreasing; rank == s.size.
        self.Vt = Vt
        self.QB = (None, None)  # will be a pair of ndarrays
        # ^ Store Q and B in one place, so they can be updated with one
        #   function call ath.QB = qb_alg(ath.A, k, tol, rng).
        self.tester = unittest.TestCase()

    def test_exact(self, fro_tol):
        Q, B = self.QB
        delta = self.A - Q @ B
        nrm = la.norm(delta, ord='fro')
        self.tester.assertLessEqual(nrm, fro_tol)

    def test_valid_onb(self, fro_tol):
        Q, B = self.QB
        self.tester.assertEqual(Q.shape[0], self.A.shape[0])
        self.tester.assertLessEqual(Q.shape[1], self.A.shape[1])
        gram = Q.T @ Q
        delta = gram - np.eye(Q.shape[1])
        nrm = la.norm(delta, ord='fro')
        self.tester.assertLessEqual(nrm, fro_tol)


class TestQBFactorizer(unittest.TestCase):

    SEEDS = [1, 4, 15, 31, 42]

    @staticmethod
    def run_batch_exact(ath: AlgTestHelper,
                        alg: rqb.QBFactorizer,
                        target_rank, target_tol,
                        test_tol, seeds):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # Call the QB algorithm, store results in AlgTestHelper.
            ath.QB = alg(ath.A, target_rank, target_tol, rng)
            # Test the results of the QB algorithm
            ath.test_valid_onb(test_tol)
            ath.test_exact(test_tol)


class TestQB1(TestQBFactorizer):

    def test_exact(self):
        alg = rqb.QB1(rqb.RF1(rsks.RS1(
            sketch_op_gen=usk.gaussian_operator,
            num_pass=1,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )))
        alg_tol = np.NaN
        test_tol = 1e-8
        # Run the above algorithm on a tall matrices, then wide matrices.
        #
        # In both cases we set the target rank of the approximation matrix
        # to the actual rank of the data matrix.
        ath1 = tall_low_exact_rank()
        alg_rank = ath1.s.size
        self.run_batch_exact(ath1, alg, alg_rank, alg_tol, test_tol, self.SEEDS)
        ath2 = wide_low_exact_rank()
        alg_rank = ath2.s.size
        self.run_batch_exact(ath2, alg, alg_rank, alg_tol, test_tol, self.SEEDS)
        # You could imagine running another test where alg_rank > ath2.s.size.
        # That would be a good test with degenerate inputs.


class TestQB2(TestQBFactorizer):

    def test_exact(self):
        alg = rqb.QB2(
            rf=rqb.RF1(rsks.RS1(
                sketch_op_gen=usk.gaussian_operator,
                num_pass=0,  # oblivious sketching operator
                stabilizer=ulaw.orth,
                passes_per_stab=1)),
            blk=4,
            overwrite_a=False
        )
        # Code from here onward is copied from TestQB1.
        #   That's undesirable.
        alg_tol = np.NaN
        test_tol = 1e-8
        # Run the above algorithm on a tall matrices, then wide matrices.
        #
        # In both cases we set the target rank of the approximation matrix
        # to the actual rank of the data matrix.
        ath1 = tall_low_exact_rank()
        alg_rank = ath1.s.size
        self.run_batch_exact(ath1, alg, alg_rank, alg_tol, test_tol, self.SEEDS)
        ath2 = wide_low_exact_rank()
        alg_rank = ath2.s.size
        self.run_batch_exact(ath2, alg, alg_rank, alg_tol, test_tol, self.SEEDS)
        # You could imagine running another test where alg_rank > ath2.s.size.
        # That would be a good test with degenerate inputs.

