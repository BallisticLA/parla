import scipy.linalg as la
import numpy as np
from typing import Union
import parla.comps.itersaddle as ris
import parla.comps.preconditioning as rpc
from parla.drivers.least_squares import dim_checks
from parla.comps.determiter.logging import SketchAndPrecondLog
from parla.utils.timing import fast_timer


NoneType = type(None)


class SaddleSolver:

    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging):
        raise NotImplementedError()


class SPS1(SaddleSolver):

    def __init__(self, sketch_op_gen,
                 sampling_factor: int,
                 iterative_solver: Union[NoneType, ris.PrecondSaddleSolver]):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        if iterative_solver is None:
            iterative_solver = ris.PcSS1()
        self.iterative_solver = iterative_solver
        pass

    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging=False):
        m, n = A.shape
        sqrt_delta = np.sqrt(delta)
        d = dim_checks(self.sampling_factor, m, n)
        rng = np.random.default_rng(rng)

        if b is None:
            b = np.zeros(m)

        quick_time = fast_timer(not logging)
        log = SketchAndPrecondLog()

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, m, rng)
        A_ske = S @ A
        A_ske = rpc.a_lift(A_ske, sqrt_delta)  # returns A_ske when delta=0.
        log.time_sketch = quick_time() - tic

        # Factor the sketch
        tic = quick_time()
        M, U, sigma, Vh = rpc.svd_right_precond(A_ske)
        log.time_factor = quick_time() - tic

        # Presolve ...
        #   (A_ske' A_ske ) x_ske = (A'b - c)                         (1, define)
        #   (V \Sigma^2 V') x_ske = (A'b - c)                         (2)
        #      \Sigma   V'  x_ske = \Sigma^{\dagger} V'(A'b - c)      (3)
        #   x_ske = V \Sigma^{\dagger} z_ske = M z_ske                (4, define)
        #   z_ske = \Sigma^{\dagger} V'(A'b - c)                      (5)
        tic = quick_time()
        rhs = A.T @ b
        if c is not None:
            rhs -= c
        z_ske = (Vh @ rhs) / sigma
        x_ske = M @ z_ske
        rhs_pc = M.T @ rhs
        lhs_ske_pc = M.T @ (A.T @ (A @ x_ske) + delta*x_ske)
        if la.norm(lhs_ske_pc - rhs_pc, ord=2) >= la.norm(rhs_pc, ord=2):
            z_ske = None
        log.time_presolve = quick_time() - tic

        # Main iterative phase
        tic = quick_time()
        res = self.iterative_solver(A, b, c, delta, tol, iter_lim, M, False, z_ske)
        log.time_iterate = quick_time() - tic
        x_star = res[0]
        y_star = res[1]

        # Finish timings
        if logging:
            log.wrap_up(res[2], la.norm(rhs))
            log.error_desc = self.iterative_solver.ERROR_METRIC_INFO

        return x_star, y_star, log


class SPS2(SaddleSolver):

    def __init__(self, sketch_op_gen,
                 sampling_factor: int,
                 iterative_solver: Union[NoneType, ris.PrecondSaddleSolver]):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        if iterative_solver is None:
            iterative_solver = ris.PcSS2()
        self.iterative_solver = iterative_solver
        pass

    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging=False):
        m, n = A.shape
        sqrt_delta = np.sqrt(delta)
        d = dim_checks(self.sampling_factor, m, n)
        rng = np.random.default_rng(rng)

        if b is None:
            b = np.zeros(m)

        quick_time = fast_timer(not logging)
        log = SketchAndPrecondLog()

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, m, rng)
        A_ske = S @ A
        A_ske = rpc.a_lift(A_ske, sqrt_delta)  # returns A_ske when delta=0.
        log.time_sketch = quick_time() - tic

        # Factor the sketch
        tic = quick_time()
        M, U, sigma, Vh = rpc.svd_right_precond(A_ske)
        log.time_factor = quick_time() - tic

        # Convert to overdetermined least squares (if applicable).
        tic = quick_time()
        A_aug = rpc.a_lift(A, sqrt_delta)  # returns A when delta=0.
        b_aug = np.concatenate((b, np.zeros(n))) if delta > 0 else b.copy()
        if c is not None and la.norm(c) > 0:
            v = U @ ((1/sigma) * (Vh @ c))
            b_aug[:m] -= S.T @ v[:d]
            if delta > 0:
                b_aug[m:] -= v[d:]
        log.time_convert = quick_time() - tic

        # Presolve
        tic = quick_time()
        z_ske = U[:d, :].T @ (S @ b_aug[:m])
        if delta > 0:
            z_ske += U[d:, :].T @ b_aug[m:]
        x_ske = M @ z_ske
        if la.norm(b_aug - A_aug @ x_ske, ord=2) >= la.norm(b_aug, ord=2):
            z_ske = None
        log.time_presolve = quick_time() - tic

        # Main iterative phase
        tic = quick_time()
        res = self.iterative_solver(A_aug, b_aug, None, 0.0,
                                    tol, iter_lim, M, False, z_ske)
        log.time_iterate = quick_time() - tic
        x_star = res[0]
        y_star = b - A @ x_star  # recompute; res[1] is for the transformed system
        res = (x_star, y_star, res[2])

        # Finish timings
        if logging:
            ar0 = M.T @ (A_aug.T @ b_aug)
            log.wrap_up(res[2], la.norm(ar0))
            log.error_desc = self.iterative_solver.ERROR_METRIC_INFO
            msg = "The metric above is computed w.r.t. a transformed problem."
            log.error_desc += msg

        return x_star, y_star, log
