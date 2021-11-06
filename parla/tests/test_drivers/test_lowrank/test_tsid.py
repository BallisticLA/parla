import unittest
import numpy as np
import scipy.linalg as la
from parla.drivers.interpolative import OSID1, OSID2, TSID1
from parla.comps.sketchers.aware import RS1
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.linalg_wrappers as ulaw
import parla.tests.matmakers as matmakers
from parla.tests.test_drivers.test_lowrank.test_osid import reference_osid


def run_tsid_test(alg, m, n, rank, k, over, test_tol, seed):
    rng = np.random.default_rng(seed)
    A = matmakers.rand_low_rank(m, n, rank, rng)
    Z, I, X, J = alg(A, k, over, rng)
    A_id = Z @ (A[I, :][:, J] @ X)

    permuted_coeffs = Z[I, :]
    delta_norm = la.norm(permuted_coeffs - np.eye(k), ord='fro')
    assert delta_norm < 1e-8

    permuted_coeffs = X[:, J]
    delta_norm = la.norm(permuted_coeffs - np.eye(k), ord='fro')
    assert delta_norm < 1e-8

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


#TODO: improve these tests
class TestTSIDs(unittest.TestCase):

    def test_simple_exact(self):
        gaussian_operator = oblivious.SkOpGA()
        rso = RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=0,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )
        osid = OSID1(rso)
        alg = TSID1(osid)
        m, n = 1000, 300
        # Algorithm will start with a row ID
        run_tsid_test(alg, m, n, rank=290, k=290, over=0,  test_tol=1e-12, seed=0)
        run_tsid_test(alg, m, n, rank=290, k=290, over=5,  test_tol=1e-12, seed=2)
        run_tsid_test(alg, m, n, rank=30, k=30, over=0,  test_tol=1e-12, seed=2)
        # Algorithm will start with a column ID
        m, n = 300, 1000
        run_tsid_test(alg, m, n, rank=290, k=290, over=0, test_tol=1e-12, seed=0)
        run_tsid_test(alg, m, n, rank=290, k=290, over=5, test_tol=1e-12, seed=2)
        run_tsid_test(alg, m, n, rank=30, k=30, over=0, test_tol=1e-12, seed=2)

    def test_simple_approx(self):
        gaussian_operator = oblivious.SkOpGA()
        rso = RS1(
            sketch_op_gen=gaussian_operator,
            num_pass=2,
            stabilizer=ulaw.orth,
            passes_per_stab=1
        )
        osid = OSID1(rso)
        alg = TSID1(osid)
        m, n = 100, 30
        run_tsid_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_tsid_test(alg, m, n, rank=30, k=25, over=4, test_tol=0.05, seed=0)
        # Re-run tests with wide data matrices
        m, n = 30, 100
        run_tsid_test(alg, m, n, rank=30, k=27, over=3, test_tol=0.05, seed=0)
        run_tsid_test(alg, m, n, rank=30, k=25, over=4, test_tol=0.05, seed=0)
