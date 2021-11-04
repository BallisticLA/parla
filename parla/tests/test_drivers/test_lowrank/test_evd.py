import unittest
import numpy as np
import scipy.linalg as la
import parla.drivers.evd as revd
import parla.comps.qb as rqb
import parla.comps.sketchers.aware as rsks
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.linalg_wrappers as ulaw
import parla.tests.test_comps.test_qb as test_qb


class AlgTestHelper:

    @staticmethod
    def convert(ath, psd, rng: np.random.Generator):
        if psd:
            lamb = ath.s
        else:
            signs = rng.random(ath.s.size) < 0.5
            signs[~signs] = -1
            lamb = ath.s * signs
        V = ath.U
        A = (V * lamb) @ V.T
        return AlgTestHelper(A, V, lamb)

    def __init__(self, A, V, lamb):
        self.A = A
        self.V = V
        self.lamb = lamb
        self.Vlamb = (None, None)
        # ^ To store values computed by a driver
        self.tester = unittest.TestCase()

    def test_conformable(self):
        V, lamb = self.Vlamb
        self.tester.assertEqual(lamb.size, V.shape[1])
        self.tester.assertEqual(V.shape[0], self.A.shape[0])

    def test_valid_onb(self, fro_tol):
        V, lamb = self.Vlamb
        gram_V = V.T @ V
        delta_V = gram_V - np.eye(lamb.size)
        nrm_V = la.norm(delta_V, ord='fro')
        self.tester.assertLessEqual(nrm_V, fro_tol)

    def test_eigvals(self, psd):
        lamb = self.Vlamb[1]
        self.tester.assertLessEqual(lamb.size, self.lamb.size)
        lamb_rev = lamb[::-1]
        diffs = np.diff(lamb_rev)
        self.tester.assertGreaterEqual(np.min(diffs), 0.0)
        if psd:
            self.tester.assertGreaterEqual(np.min(lamb), 0.0)

    def test_abs_fro_error(self, abs_tol):
        #TODO: change this to relative tolerance
        V, lamb = self.Vlamb
        delta = self.A - (V * lamb) @ V.T
        nrm = la.norm(delta, ord='fro')
        # abs_tol = rel_tol * np.norm(self.s, ord=2)
        # ^ Scale by  Frobenius norm of A.
        # self.tester.assertLessEqual(nrm, abs_tol)
        self.tester.assertLessEqual(nrm, abs_tol)


class TestEVDecomposer(unittest.TestCase):

    SEEDS = [1, 2, 3]

    PSD = False

    INFLATE_TEST_TOL = 1.1

    @staticmethod
    def run_batch(ath: AlgTestHelper,
                  alg: revd.EVDecomposer,
                  target_rank, target_tol, over,
                  test_tol, seeds):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # Call the SVD algorithm, store the results in AlgTestHelper
            ath.Vlamb = alg(ath.A, target_rank, target_tol, over, rng)
            # Test the results
            ath.test_conformable()
            ath.test_eigvals(TestEVDecomposer.PSD)
            ath.test_valid_onb(test_tol)
            if not np.isnan(target_tol):
                # slightly inflate the tolerance to account for minor
                # numerical issues.
                abstol = TestEVDecomposer.INFLATE_TEST_TOL * target_tol
                ath.test_abs_fro_error(abstol)


class TestEVD1(TestEVDecomposer):
    # These tests are more or less copied from TestSVD1
    # Run tests backed by QB1 and QB2.

    def test_fr(self):
        # For a wide matrix and a tall matrix:
        #   Fixed rank QB algorithm
        #   Three cases:
        #       Target rank < exact rank (no oversampling).
        #       Target rank + oversampling < exact rank
        #       Target rank + oversampling > exact rank
        #
        alg = revd.EVD1(rqb.QB1(rqb.RF1(rsks.RS1(
                    sketch_op_gen=oblivious.SkOpGA(),
                    num_pass=1,
                    stabilizer=ulaw.orth,
                    passes_per_stab=1
        ))))
        rng = np.random.default_rng(0)
        ath = AlgTestHelper.convert(test_qb.wide_low_exact_rank(), self.PSD, rng)
        rank = ath.lamb.size
        self.run_batch(ath, alg, rank - 10, np.NaN, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 10, np.NaN, 5, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 3, np.NaN, 5, 1e-8, self.SEEDS)
        pass

    def test_fp_inexact(self):
        # set the error tolerance to 0.25x the Frobenius norm of the matrix
        alg = revd.EVD1(rqb.QB2(
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
        rng = np.random.default_rng(0)
        ath = AlgTestHelper.convert(test_qb.wide_full_exact_rank(), self.PSD, rng)
        rank = min(ath.A.shape)
        abs_err = 0.25*la.norm(ath.lamb, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)
        pass

    def test_fp_exact(self):
        # set the (relative) target tolerance to 1e-12.
        alg = revd.EVD1(rqb.QB2(
            rqb.RF1(rqb.RS1(
                sketch_op_gen=oblivious.SkOpGA(),
                num_pass=0,
                stabilizer=ulaw.orth,
                passes_per_stab=1
            )),
            blk=2,
            overwrite_a=False,
        ))
        # Wide matrix
        rng = np.random.default_rng(0)
        ath = AlgTestHelper.convert(test_qb.wide_low_exact_rank(), self.PSD, rng)
        rank = min(ath.A.shape)
        abs_err = 1e-12*la.norm(ath.lamb, ord=2)
        self.run_batch(ath, alg, rank, abs_err, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank, abs_err, 2, 1e-8, self.SEEDS)
        pass


class TestEVD2(TestEVDecomposer):
    # These tests are more or less copied from TestSVD1
    # Run tests backed by QB1 and QB2.

    PSD = True

    def test_fr(self):
        # For a wide matrix and a tall matrix:
        #   Fixed rank QB algorithm
        #   Three cases:
        #       Target rank < exact rank (no oversampling).
        #       Target rank + oversampling < exact rank
        #       Target rank + oversampling > exact rank
        #
        alg = revd.EVD2(rsks.RS1(
                sketch_op_gen=oblivious.SkOpGA(),
                num_pass=1,
                stabilizer=ulaw.orth,
                passes_per_stab=1
        ))
        rng = np.random.default_rng(0)
        ath = AlgTestHelper.convert(test_qb.wide_low_exact_rank(), self.PSD, rng)
        rank = ath.lamb.size
        self.run_batch(ath, alg, rank - 10, np.NaN, 0, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 10, np.NaN, 5, 1e-8, self.SEEDS)
        self.run_batch(ath, alg, rank - 3, np.NaN, 5, 1e-8, self.SEEDS)
        pass
