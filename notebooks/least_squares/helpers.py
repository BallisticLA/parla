import numpy as np
import scipy.linalg as la
import parla.utils.sketching as sk
from parla.tests.test_drivers.test_optim.test_overdet_least_squares import AlgTestHelper


def make_demo_helper(m, n, spectrum, prop_range, rng, only_Ab=False):
    rng = np.random.default_rng(rng)

    # Construct the data matrix
    rank = spectrum.size
    U = sk.orthonormal_operator(m, rank, rng)
    Vt = sk.orthonormal_operator(rank, n, rng)
    A = (U * spectrum) @ Vt

    # Construct the right-hand-side
    b0 = rng.standard_normal(m)
    b_range = U @ (U.T @ b0)
    b_orthog = b0 - b_range
    b_range *= (np.mean(spectrum) / la.norm(b_range))
    b_orthog *= (np.mean(spectrum) / la.norm(b_orthog))
    b = prop_range * b_range + (1 - prop_range) * b_orthog

    if only_Ab:
        return A, b
    else:
        x_opt = (Vt.T / spectrum) @ (U.T @ b)
        dh = LSDemoHelper(A, b, x_opt, U, spectrum, Vt)
        return dh


class LSDemoHelper(AlgTestHelper):

    def __init__(self, A, b, x_opt, U, s, Vt):
        super(LSDemoHelper, self).__init__(A, b, x_opt, U, s, Vt)
        self.scaled_V = Vt.T / s

    def resample_b(self, prop_range, rng):
        rng = np.random.default_rng(rng)
        b0 = rng.standard_normal(self.A.shape[0])
        b_range = self.project_onto_range(b0)
        b_orthog = b0 - b_range
        b_range *= (np.mean(self.s) / la.norm(b_range))
        b_orthog *= (np.mean(self.s) / la.norm(b_orthog))
        self.b = prop_range * b_range + (1 - prop_range) * b_orthog
        self.x_opt = self.solve(self.b)
        self.x_approx = None
        return self.b

    def project_onto_range(self, vec):
        return self.U @ (self.U.T @ vec)

    def solve(self, vec):
        return self.scaled_V @ (self.U.T @ vec)
