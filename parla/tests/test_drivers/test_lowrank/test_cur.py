import unittest
import numpy as np
import scipy.linalg as la
from parla.drivers.interpolative import CUR1, OSID1
from parla.comps.interpolative import ROCS1
from parla.comps.sketchers.aware import RS1
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.linalg_wrappers as ulaw
import parla.tests.matmakers as matmakers
from parla.tests.test_drivers.test_lowrank.test_osid import reference_osid


def run_cur_test(alg, m, n, rank, k, over, test_tol, seed):
    rng = np.random.default_rng(seed)
    A = matmakers.rand_low_rank(m, n, rank, rng)
    Js, U, Is = alg(A, k, over, rng)
    A_id = A[:, Js] @ (U @ A[Is, :])

    err_rand = la.norm(A - A_id, ord='fro')
    if test_tol < 1e-8:
        rel_err = err_rand / la.norm(A, ord='fro')
        assert rel_err < test_tol
    else:
        A_id_ref, _, _ = reference_osid(A, k, 0)
        err_ref = la.norm(A - A_id_ref, ord='fro')
        rel_err = (err_rand - err_ref) / la.norm(A, ord='fro')
        print(rel_err)
    assert rel_err < test_tol


class TestCURDecomposition(unittest.TestCase):

    def _test_simple_exact(self, alg):
        m, n = 100, 30
        run_cur_test(alg, m, n, rank=24, k=25, over=3, test_tol=1e-12, seed=0)
        run_cur_test(alg, m, n, rank=5, k=5, over=1, test_tol=1e-12, seed=2)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_cur_test(alg, m, n, rank=24, k=25, over=3, test_tol=1e-12, seed=0)
        run_cur_test(alg, m, n, rank=5, k=5, over=1, test_tol=1e-12, seed=2)

    def _test_simple_approx(self, alg):
        m, n = 100, 30
        run_cur_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.15, seed=0)
        run_cur_test(alg, m, n, rank=30, k=25, over=4, test_tol=0.15, seed=0)
        run_cur_test(alg, m, n, rank=30, k=5, over=5, test_tol=0.15, seed=0)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_cur_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.15, seed=0)
        run_cur_test(alg, m, n, rank=30, k=25, over=4, test_tol=0.15, seed=0)
        run_cur_test(alg, m, n, rank=30, k=5, over=5, test_tol=0.15, seed=0)


class TestCUR1(TestCURDecomposition):
    # This algorithm gets better approximation error than CUR2.

    def test_simple_exact(self):
        gaussian_operator = oblivious.SkOpGA()
        alg = CUR1(OSID1(RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=0,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )))
        self._test_simple_exact(alg)

    def test_simple_approx(self):
        gaussian_operator = oblivious.SkOpGA()
        alg = CUR1(OSID1(RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=4,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )))
        self._test_simple_approx(alg)

