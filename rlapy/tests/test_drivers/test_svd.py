import unittest
import numpy as np
import scipy.linalg as la
import rlapy.drivers.svd as rsvd
import rlapy.comps.qb as rqb
import rlapy.comps.sketchers.aware as rsks
import rlapy.comps.sketchers.oblivious as oblivious
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.tests.test_comps.test_qb as test_qb


class AlgTestHelper:

    @staticmethod
    def convert(ath):
        return AlgTestHelper(ath.A, ath.U, ath.s, ath.Vt)

    def __init__(self, A, U, s, Vt):
        self.A = A
        self.U = U
        self.s = s
        self.Vt = Vt
        # ^ Those U, s, Vt are reference values
        self.UsVt = (None, None, None)
        # ^ To store values computed by a driver
        self.tester = unittest.TestCase()

    def test_conformable(self):
        U, s, Vt = self.UsVt
        self.tester.assertEqual(s.size, U.shape[1])
        self.tester.assertEqual(s.size, Vt.shape[0])
        self.tester.assertEqual(U.shape[0], self.A.shape[0])
        self.tester.assertEqual(Vt.shape[1], self.A.shape[1])

    def test_valid_onb(self, fro_tol):
        U, s, Vt = self.UsVt

        gram_U = U.T @ U
        delta_U = gram_U - np.eye(s.size)
        nrm_U = la.norm(delta_U, ord='fro')
        self.tester.assertLessEqual(nrm_U, fro_tol)

        gram_V = Vt @ Vt.T
        delta_V = gram_V - np.eye(s.size)
        nrm_V = la.norm(delta_V, ord='fro')
        self.tester.assertLessEqual(nrm_V, fro_tol)

    def test_valid_singvals(self):
        s = self.UsVt[1]
        self.tester.assertLessEqual(s.size, self.s.size)
        self.tester.assertGreaterEqual(np.min(s), 0.0)
        s_rev = s[::-1]
        diffs = np.diff(s_rev)
        self.tester.assertGreaterEqual(np.min(diffs), 0.0)

    def test_abs_fro_error(self, abs_tol):
        #TODO: change this to relative tolerance
        U, s, Vt = self.UsVt
        delta = self.A - (U * s) @ Vt
        nrm = la.norm(delta, ord='fro')
        # abs_tol = rel_tol * np.norm(self.s, ord=2)
        # ^ Scale by  Frobenius norm of A.
        # self.tester.assertLessEqual(nrm, abs_tol)
        self.tester.assertLessEqual(nrm, abs_tol)


class TestSVDecomposer(unittest.TestCase):

    SEEDS = [38972, 653, 1222]

    @staticmethod
    def run_batch(ath: AlgTestHelper,
                  alg: rsvd.SVDecomposer,
                  target_rank, target_tol, over,
                  test_tol, seeds):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # Call the SVD algorithm, store the results in AlgTestHelper
            ath.UsVt = alg(ath.A, target_rank, target_tol, over, rng)
            # Test the results
            ath.test_conformable()
            ath.test_valid_singvals()
            ath.test_valid_onb(test_tol)
            if not np.isnan(target_tol):
                ath.test_abs_fro_error(target_tol)


class TestSVD1(TestSVDecomposer):
    # Run tests backed by QB1 and QB2.

    def test_fr(self):
        # For a wide matrix and a tall matrix:
        #   Fixed rank QB algorithm
        #   Three cases:
        #       Target rank < exact rank (no oversampling).
        #       Target rank + oversampling < exact rank
        #       Target rank + oversampling > exact rank
        #
        alg = rsvd.SVD1(rqb.QB1(rqb.RF1(rsks.RS1(
                    sketch_op_gen=oblivious.SkOpGA(),
                    num_pass=1,
                    stabilizer=ulaw.orth,
                    passes_per_stab=1
        ))))

        # Tall matrix
        ath = AlgTestHelper.convert(test_qb.tall_low_exact_rank())
        rank = ath.s.size
        self.run_batch(ath, alg, rank-10, np.NaN, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank-10, np.NaN, 5, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank-3, np.NaN, 5, 1e-8, self.SEEDS)

        # Wide matrix
        ath = AlgTestHelper.convert(test_qb.wide_low_exact_rank())
        rank = ath.s.size
        self.run_batch(ath, alg, rank - 10, np.NaN, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 10, np.NaN, 5, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 3, np.NaN, 5, 1e-8, self.SEEDS)
        pass

    def test_fp_inexact(self):
        # set the error tolerance to 0.25x the Frobenius norm of the matrix
        alg = rsvd.SVD1(rqb.QB2(
            rqb.RF1(rqb.RS1(
                sketch_op_gen=oblivious.SkOpGA(),
                num_pass=0,
                stabilizer=ulaw.orth,
                passes_per_stab=1
            )),
            blk=2,
            overwrite_a=False,
        ))
        #TODO: update tests so that we can verify the returned matrix
        #   has rank < min(A.shape).

        # Tall matrix
        ath = AlgTestHelper.convert(test_qb.tall_full_exact_rank())
        rank = min(ath.A.shape)
        abs_err = 0.25*la.norm(ath.s, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)

        # Wide matrix
        ath = AlgTestHelper.convert(test_qb.wide_full_exact_rank())
        rank = min(ath.A.shape)
        abs_err = 0.25*la.norm(ath.s, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)
        pass

    def test_fp_exact(self):
        # set the (relative) target tolerance to 1e-12.
        alg = rsvd.SVD1(rqb.QB2(
            rqb.RF1(rqb.RS1(
                sketch_op_gen=oblivious.SkOpGA(),
                num_pass=0,
                stabilizer=ulaw.orth,
                passes_per_stab=1
            )),
            blk=2,
            overwrite_a=False,
        ))

        # Tall matrix (low exact rank)
        ath = AlgTestHelper.convert(test_qb.tall_low_exact_rank())
        rank = min(ath.A.shape)
        abs_err = 1e-12*la.norm(ath.s, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)

        # Tall matrix (full rank)
        ath = AlgTestHelper.convert(test_qb.tall_full_exact_rank())
        rank = min(ath.A.shape)
        abs_err = 1e-12*la.norm(ath.s, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 1, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)

        # Wide matrix
        ath = AlgTestHelper.convert(test_qb.wide_low_exact_rank())
        rank = min(ath.A.shape)
        abs_err = 1e-12*la.norm(ath.s, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)
        pass
