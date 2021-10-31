import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.interpolative import CURD1, ROCS1
from rlapy.comps.sketchers.aware import RS1
import rlapy.comps.sketchers.oblivious as oblivious
import rlapy.utils.linalg_wrappers as ulaw


def run_cur_test(m, rank, k, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, rank)).astype(np.float64)
    A = A.dot(A.T)[:, :(m // 2)]
    rng = 1
    num_pass = 4
    over = 3
    print(A.shape)
    # ------------------------------------------------------------------------
    # test index_set == False

    curd = CURD1(ROCS1(RS1(
        sketch_op_gen=oblivious.SkOpGA(),
        num_pass=num_pass,
        stabilizer=ulaw.orth,
        passes_per_stab=1
    )))
    Is, Js, U = curd(A, k, over, rng)
    A_id = A[:, Js] @ U(A[Is, :])
    err = la.norm(A - A_id) / la.norm(A)
    print("error: ", err)
    assert err < 1e-4


class TestCURD1(unittest.TestCase):

    def test_very_simple(self):
        run_cur_test(1000, 300, 300, seed=0)
