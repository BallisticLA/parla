import unittest
import numpy as np
import scipy.linalg as la
from parla.drivers.interpolative import OSID1, OSID2
from parla.comps.sketchers.aware import RS1
from parla.comps.interpolative import qrcp_osid
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.linalg_wrappers as ulaw
import parla.tests.matmakers as matmakers


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
    err_rand = la.norm(A - A_id, ord='fro')
    if test_tol < 1e-8:
        rel_err = err_rand / la.norm(A, ord='fro')
        assert rel_err < test_tol
    else:
        A_id_ref, _, _ = reference_osid(A, k, axis)
        err_ref = la.norm(A - A_id_ref, ord='fro')
        rel_err = (err_rand - err_ref) / la.norm(A, ord='fro')
        print(rel_err)
    assert rel_err < test_tol


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
            run_osid_test(alg, m, n, rank=30, k=25, over=4, axis=0, test_tol=0.05, seed=0)
            run_osid_test(alg, m, n, rank=30, k=5, over=5, axis=0, test_tol=0.1, seed=0)
            # Re-run tests with wide data matrices
            m, n = 30, 100
            run_osid_test(alg, m, n, rank=30, k=27, over=3, axis=1, test_tol=0.05, seed=0)
            run_osid_test(alg, m, n, rank=30, k=25, over=4, axis=1, test_tol=0.05, seed=0)
            run_osid_test(alg, m, n, rank=30, k=5, over=5, axis=0, test_tol=0.1, seed=0)
