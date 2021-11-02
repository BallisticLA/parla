import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.interpolative import CURD1
from rlapy.comps.interpolative import ROCS1
from rlapy.comps.sketchers.aware import RS1
import rlapy.comps.sketchers.oblivious as oblivious
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.tests.matmakers as matmakers


def run_cur_test(alg, m, n, rank, k, over, test_tol, seed):
    rng = np.random.default_rng(seed)
    A = matmakers.rand_low_rank(m, n, rank, rng)
    Js, U, Is = alg(A, k, over, rng)
    A_id = A[:, Js] @ (U @ A[Is, :])
    err = la.norm(A - A_id, ord='fro') / la.norm(A, ord='fro')
    assert err < test_tol


class TestCURD1(unittest.TestCase):

    def test_simple_exact(self):
        gaussian_operator = oblivious.SkOpGA()
        alg = CURD1(ROCS1(RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=0,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )))
        m, n = 100, 30
        run_cur_test(alg, m, n, rank=24, k=25, over=3, test_tol=1e-12, seed=0)
        run_cur_test(alg, m, n, rank=5, k=5, over=1, test_tol=1e-12, seed=2)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_cur_test(alg, m, n, rank=24, k=25, over=3, test_tol=1e-12, seed=0)
        run_cur_test(alg, m, n, rank=5, k=5, over=1, test_tol=1e-12, seed=2)

    def test_simple_approx(self):
        gaussian_operator = oblivious.SkOpGA()
        alg = CURD1(ROCS1(RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=4,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )))
        m, n = 100, 30
        run_cur_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_cur_test(alg, m, n, rank=30, k=27, over=1, test_tol=0.1, seed=0)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_cur_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_cur_test(alg, m, n, rank=30, k=27, over=1, test_tol=0.1, seed=0)
