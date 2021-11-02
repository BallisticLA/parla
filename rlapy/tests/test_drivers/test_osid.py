import unittest
import numpy as np
import scipy.linalg as la
from rlapy.drivers.interpolative import OSID1, OSID2
from rlapy.comps.sketchers.aware import RS1
from rlapy.comps.interpolative import qrcp_osid
import rlapy.comps.sketchers.oblivious as oblivious
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.tests.matmakers as matmakers


#TODO: create tests which compare approximation quality against
#   the output of this function
def reference_osid(A, k, axis):
    M, P = qrcp_osid(A, k, axis)
    if axis == 0:
        A_id = M @ A[P, :]
    else:
        A_id = A[:, P] @ M
    return A_id, M, P


def run_osid_test(alg, m, n, rank, k, over, axis, test_tol, seed):
    rng = np.random.default_rng(seed)
    A = matmakers.rand_low_rank(m, n, rank, rng)
    M, P = alg(A, k, over, axis, rng)
    if axis == 0:
        A_id = M @ A[P, :]
        permuted_coeffs = M[P, :]
        delta_norm = la.norm(permuted_coeffs - np.eye(P.size), ord='fro')
        assert delta_norm < 1e-8
    elif axis == 1:
        A_id = A[:, P] @ M
        permuted_coeffs = M[:, P]
        delta_norm = la.norm(permuted_coeffs - np.eye(P.size), ord='fro')
        assert delta_norm < 1e-8
    else:
        raise ValueError()
    err = la.norm(A - A_id, ord='fro') / la.norm(A, ord='fro')
    assert err < test_tol


class TestOSIDs(unittest.TestCase):

    def test_simple_exact(self):
        gaussian_operator = oblivious.SkOpGA()
        rso = RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=0,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )
        for alg in [OSID1(rso), OSID2(rso)]:
            m, n = 100, 30
            # Row IDs
            run_osid_test(alg, m, n, rank=29, k=29, over=0, axis=0, test_tol=1e-12, seed=0)
            run_osid_test(alg, m, n, rank=28, k=28, over=1, axis=0, test_tol=1e-12, seed=2)
            run_osid_test(alg, m, n, rank=3, k=3, over=0, axis=0, test_tol=1e-12, seed=2)
            # Column IDs
            run_osid_test(alg, m, n, rank=29, k=29, over=0, axis=1, test_tol=1e-12, seed=0)
            run_osid_test(alg, m, n, rank=28, k=28, over=1, axis=1, test_tol=1e-12, seed=2)
            run_osid_test(alg, m, n, rank=3, k=3, over=2, axis=1, test_tol=1e-12, seed=2)

    def test_simple_approx(self):
        gaussian_operator = oblivious.SkOpGA()
        rso = RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=2,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )
        for alg in [OSID1(rso), OSID2(rso)]:
            m, n = 100, 30
            run_osid_test(alg, m, n, rank=30, k=27, over=3, axis=0, test_tol=0.05, seed=0)
            run_osid_test(alg, m, n, rank=30, k=25, over=4, axis=0, test_tol=0.3, seed=0)
            # Re-run tests with wide data matrices
            m, n = 30, 100
            run_osid_test(alg, m, n, rank=30, k=27, over=3, axis=1, test_tol=0.05, seed=0)
            run_osid_test(alg, m, n, rank=30, k=25, over=4, axis=1, test_tol=0.3, seed=0)
