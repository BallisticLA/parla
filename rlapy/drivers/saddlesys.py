import scipy.linalg as la
import numpy as np

import rlapy.comps.itersaddle as ris
import rlapy.comps.preconditioning as rpc
from rlapy.drivers.least_squares import dim_checks
import time


class SaddleSolver:

    def __call__(self, A, b, c, delta, tol, iter_lim, rng):
        raise NotImplementedError()


class SPS2(SaddleSolver):

    def __init__(self, sketch_op_gen,
                 sampling_factor: int,
                 iterative_solver: ris.PrecondSaddleSolver):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.iterative_solver = iterative_solver
        self.log = {'time_sketch': -1.0,
                    'time_factor': -1.0,
                    'time_convert': -1.0,
                    'time_presolve': -1.0,
                    'time_iterate': -1.0,
                    'times': np.empty((1,)),
                    'arnorms': np.empty((1,)),
                    'x': np.empty((1,)),
                    'y': np.empty((1,))}
        pass

    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging=False):
        m, n = A.shape
        d = dim_checks(self.sampling_factor, m, n)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)

        if b is None:
            b = np.zeros(m)

        if logging:
            quick_time = time.time()
        else:
            quick_time = lambda: 0.0

        sqrt_delta = np.sqrt(delta)

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, m, rng)
        A_ske = S @ A
        A_ske = rpc.a_lift(A_ske, sqrt_delta)  # returns A_ske when delta=0.
        self.log['time_sketch'] = quick_time() - tic

        # Factor the sketch
        tic = quick_time()
        U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                              full_matrices=False)
        eps = np.finfo(float).eps
        rank = np.count_nonzero(sigma > sigma[0] * n * eps)
        Vh = Vh[:rank, :]
        U = U[:, :rank]
        sigma = sigma[:rank]
        M = Vh.T / sigma
        self.log['time_factor'] = quick_time() - tic

        # Convert to overdetermined least squares (if applicable).
        tic = quick_time()
        A_aug = rpc.a_lift(A, sqrt_delta)  # returns A when delta=0.
        b_aug = np.concatenate((b, np.zeros(n))) if delta > 0 else b.copy()
        if c is not None and la.norm(c) > 0:
            v = U @ ((1/sigma) * (Vh @ c))
            b_aug[:m] -= S.T @ v[:d]
            if delta > 0:
                b_aug[m:] -= v[d:]
        self.log['time_convert'] = quick_time() - tic

        # Presolve
        tic = quick_time()
        z_ske = U[:d, :].T @ (S @ b_aug[:m])
        if delta > 0:
            z_ske += U[d:, :].T @ b_aug[m:]
        x_ske = M @ z_ske
        if la.norm(b_aug - A_aug @ x_ske, ord=2) >= la.norm(b_aug, ord=2):
            z_ske = None
        self.log['time_presolve'] = quick_time() - tic

        # Main iterative phase
        tic = quick_time()
        res = self.iterative_solver(A_aug, b_aug, None, 0.0,
                                    tol, iter_lim, M, False, z_ske)
        self.log['time_iterate'] = quick_time() - tic
        x_star = res[0]
        y_star = b - A @ x_star
        res = (x_star, y_star) + res[2:]

        # Finish timings
        if logging:
            iters = res[3]
            time_setup = self.log['time_sketch']
            time_setup += self.log['time_factor']
            time_setup += self.log['time_convert']
            iterating = np.linspace(0, self.log['time_iterate'],
                                    iters, endpoint=True)
            cumulative = time_setup + self.log['time_presolve'] + iterating
            times = np.concatenate(([time_setup], cumulative))
            self.log['times'] = times
            arnorms = res[8][:iters]
            ar0 = M.T @ (A_aug.T @ b_aug)
            self.log['arnorms'] = np.concatenate(([la.norm(ar0)], arnorms))

            self.log['x'] = x_star
            self.log['y'] = y_star

        return x_star, y_star
