import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.interpolative import CURD1
from rlapy.comps.interpolative import ROCS1
from rlapy.comps.sketchers.aware import RS1
import rlapy.comps.sketchers.oblivious as oblivious
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.tests.matmakers as matmakers


def run_cur_test(m, n, rank, k, over, test_tol, seed=0):
    rng = np.random.default_rng(seed)
    A = matmakers.rand_low_rank(m, n, rank, rng)
    rng = 1
    num_pass = 4
    print(A.shape)
    # ------------------------------------------------------------------------
    # test index_set == False

    gaussian_operator = oblivious.SkOpGA()
    curd = CURD1(ROCS1(RS1(
        sketch_op_gen=gaussian_operator,
        num_pass=num_pass,
        stabilizer=ulaw.orth,
        passes_per_stab=1
    )))
    Js, U, Is = curd(A, k, over, rng)
    A_id = A[:, Js] @ (U @ A[Is, :])
    err = la.norm(A - A_id) / la.norm(A)
    assert err < test_tol


class TestCURD1(unittest.TestCase):

    def test_simple_exact(self):
        m, n = 1000, 300
        run_cur_test(m, n, rank=300, k=300, over=3, test_tol=1e-12, seed=0)
        run_cur_test(m, n, rank=290, k=290, over=5, test_tol=1e-12, seed=2)
        # Re-run tests with wide data matrices
        m, n = 300, 1000
        run_cur_test(m, n, rank=300, k=300, over=3, test_tol=1e-12, seed=0)
        run_cur_test(m, n, rank=290, k=290, over=5, test_tol=1e-12, seed=2)

    def test_simple_approx(self):
        m, n = 100, 30
        run_cur_test(m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_cur_test(m, n, rank=30, k=25, over=4, test_tol=0.3, seed=2)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_cur_test(m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_cur_test(m, n, rank=30, k=25, over=4, test_tol=0.3, seed=2)
