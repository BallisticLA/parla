import warnings
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
                 pcss: ris.PrecondSaddleSolver):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.pcss = pcss
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

        # Sketch the data matrix
        tic = time.time() if logging else 0
        S = self.sketch_op_gen(d, m, rng)
        A_ske = S @ A
        if delta > 0:
            augment = np.sqrt(delta) * np.eye(n)
            A_ske = np.row_stack((A_ske, augment))
            # ^ Need to augment explicitly, since we're about to factor by SVD.
        toc = time.time() if logging else 0
        self.log['time_sketch'] = toc - tic

        # Factor the sketch
        tic = time.time() if logging else 0
        U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                              full_matrices=False)
        eps = np.finfo(float).eps
        rank = np.count_nonzero(sigma > sigma[0] * n * eps)
        M = Vh[:rank, :].T / sigma[:rank]
        U = U[:, :rank]
        toc = time.time() if logging else 0
        self.log['time_factor'] = toc - tic

        # Convert to overdetermined least squares (if applicable).
        tic = time.time() if logging else 0
        if c is not None and la.norm(c) > 0:
            b_aug = np.concatenate((b, np.zeros(n)))
            # if c is zero and delta > 0 then we'll end up calling
            # a function to augment b later on (don't augment here!)
            v = A_ske @ (M @ (M.T @ c))
            b_aug[:m] -= S.T @ v[:d]
            b_aug[m:] -= v[d:]
            # now, c = 0.
            A_aug = rpc.a_lift(A, np.sqrt(delta))
        else:
            A_aug = A
            b_aug = b
        toc = time.time() if logging else 0
        self.log['time_convert'] = toc - tic

        # Presolve
        tic = time.time() if logging else 0
        z_ske = U[:d, :].T @ (S @ b_aug[:m])
        if b_aug.size > m:
            z_ske += U[d:, :].T @ b_aug[m:]
        x_ske = M @ z_ske
        b_aug_remainder = b_aug - A_aug @ x_ske
        if la.norm(b_aug_remainder, ord=2) >= la.norm(b_aug, ord=2):
            z_ske = None
        toc = time.time() if logging else 0
        self.log['time_presolve'] = toc - tic

        # Main iterative phase
        tic = time.time() if logging else 0
        res = self.pcss(A, b, None, 0.0, tol, iter_lim, M, False, z_ske)
        toc = time.time() if logging else 0
        self.log['time_iterate'] = toc - tic
        x_star = res[0]
        y_star = b - A @ x_star
        res = (x_star, y_star) + res[2:]

        # Finish timings
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
        ar0 = M.T @ (A.T @ b)
        self.log['arnorms'] = np.concatenate(([la.norm(ar0)], arnorms))

        self.log['x'] = x_star
        self.log['y'] = y_star

        return x_star, y_star
