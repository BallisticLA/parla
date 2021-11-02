"""
Routines for (approximately) solving over-determined least squares problems

    min{ || A x - b ||_2 + reg || x ||_2 : x in R^n }.
"""
import warnings
import scipy.linalg as la
import numpy as np

import rlapy.comps.itersaddle as ris
import time


class OverLstsqSolver:
    """Solver for overdetermined least squares problems."""

    def __call__(self, A, b, delta, tol, iter_lim, rng):
        """
        Return an approximate solution to
            min{ ||A x - b||_2^2 + delta * ||x||_2^2 : x in R^n }.

        There is no requirement that an implementation is able to control
        the error of its returned solution. Some implementations will produce
        a solution by an iterative method. We can regard non-iterative
        implementations as iterative methods that take only one step.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Tall data matrix for overdetermined ordinary least squares.

        b : ndarray
            Right-hand-side.

        delta : float
            Regularization parameter. Must be nonnegative.

        tol : float
            This parameter is only relevant for implementations that involve
            some iterative method. Those implementations must have some kind
            of error metric (e.g., backward error in the normal equations)
            for a candidate solution. If the implementation's measurement of
            error falls below tol, then it returns the candidate solution.

            If an implementation does not use an iterative method and receives
            tol is not NaN, then a warning will be raised.

        iter_lim : int
            We require iter_lim > 0. Typically, iter_lim << A.shape[1].
            This parameter is only relevant for implementations that involve
            some kind of iterative method. Those implementations must terminate
            after iter_lim iterations.

            If an implementation does not use an iterative method and receives
            iter_lim > 1, then a warning will be raised.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        x_star : ndarray
            x_star.shape == (A.shape[1],). Approximate solution to the least
            squares problem under consideration.

        log : dict
            Keyed by strings; contains algorithm-specific metadata.
        """
        raise NotImplementedError()


def dim_checks(sampling_factor, n_rows, n_cols):
    assert n_rows >= n_cols
    d = int(sampling_factor * n_cols)
    if d > n_rows:
        msg = f"""
        The embedding dimension "d" should not be larger than the 
        number of rows of the data matrix. Here, an embedding dimension
        of d={d} has been requested for a matrix with only {n_rows} rows.
        We will proceed by setting d={n_rows}. This parameter choice will
        result in a very inefficient algorithm!
        """
        # ^ Python 3.6 parses that string to drop-in the symbols d and n_rows.
        warnings.warn(msg)
        d = n_rows
    assert d >= n_cols
    return d


class SSO1(OverLstsqSolver):
    """A sketch-and-solve approach to overdetermined ordinary least squares.

    When constructing objects from this class, users may specify the LAPACK
    driver to be used in solving sketched least squares problems.

    References
    ----------
    The sketch-and-solve approach is attributed to a 2006 paper by Sarlos:
    "Improved approximation algorithms for large matrices via random
     projections." An introduction and summary of this approach can be found
     in [MT:2020, Sections 10.2 -- 10.3].
    """

    def __init__(self, sketch_op_gen, sampling_factor, lapack_driver=None,
                 overwrite_sketch=True):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.lapack_driver = lapack_driver
        self.overwrite_sketch = overwrite_sketch

    def __call__(self, A, b, delta, tol, iter_lim, rng, logging=False):
        if not np.isnan(tol):
            msg = """
            This OverLstsqSolver implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        if iter_lim > 1:
            msg = """
            This OverLstsqSolver implementation is not iterative.
            Parameter "iter_lim" is being ignored.
            """
            warnings.warn(msg)
        n_rows, n_cols = A.shape
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        rng = np.random.default_rng(rng)

        log = {
            'time_sketch': -1.0,
            'time_solve': -1.0,
        }

        if logging:
            quick_time = time.time
        else:
            quick_time = lambda: 0.0

        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        b_ske = S @ b
        log['time_sketch'] = quick_time() - tic

        tic = quick_time()
        if delta > 0:
            A_ske = np.vstack((A_ske, (delta**0.5) * np.eye(n_cols)))
            b_ske = np.concatenate((b_ske, np.zeros(n_cols)))
        res = la.lstsq(A_ske, b_ske,
                       cond=None, overwrite_a=self.overwrite_sketch,
                       overwrite_b=True, check_finite=False,
                       lapack_driver=self.lapack_driver)
        log['time_solve'] = quick_time() - tic

        x_ske = res[0]
        return x_ske, log


class SPO3(OverLstsqSolver):
    """A sketch-and-precondition approach to overdetermined ordinary least
    squares. This implementation uses QR to obtain the preconditioner and
    it uses LSQR for the iterative method.

    Before starting LSQR, we run a basic sketch-and-solve (for free, given
    our QR decomposition of the sketched data matrix) to obtain a solution
    x_ske. If ||A x_ske - b||_2 < ||b||_2, then we initialize LSQR at x_ske.

    This implementation assumes A is full rank.

    References
    ----------
    This implementation was inspired by Blendenpik (AMT:2010). The differences
    relative to the official Blendenpik algorithm [AMT:2010, Algorithm 1] are

        (1) We make no assumption on the distribution of the sketching matrix
            which is used to form the preconditioner. Blendenpik only used
            SRTTs (Walsh-Hadamard, discrete cosine, discrete Hartley).

        (2) We let the user specify the exact embedding dimension, as
            floor(self.oversampling_factor * A.shape[1]).

        (3) We do not zero-pad A with additional rows. Such zero padding
            might be desirable to facilitate rapid application of SRTT
            sketching operators. It is possible to implement an SRTT operator
            so that it performs zero-padding internally.

        (4) We do not perform any checks on the quality of the preconditioner.

        (5) We initialize the iterative solver (LSQR) at the better of the two
            solutions given by either the zero vector or the output of
            sketch-and-solve.
    """

    def __init__(self, sketch_op_gen, sampling_factor: int):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.iterative_solver = ris.PcSS2()  # implements LSQR

    def __call__(self, A, b, delta, tol, iter_lim, rng, logging=False):
        n_rows, n_cols = A.shape
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)

        log = {'time_sketch': -1.0,
               'time_factor': -1.0,
               'time_presolve': -1.0,
               'time_iterate': -1.0,
               'times': np.empty((1,)),
               'arnorms': np.empty((1,))}

        if logging:
            quick_time = time.time
        else:
            quick_time = lambda: 0.0

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        log['time_sketch'] = quick_time() - tic

        # Factor the sketch
        tic = quick_time()
        if delta > 0:
            A_ske = np.vstack((A_ske, np.sqrt(delta)*np.eye(n_cols)))
        Q, R = la.qr(A_ske, overwrite_a=True, mode='economic')
        log['time_factor'] = quick_time() - tic

        # Sketch-and-solve type preprocessing
        tic = quick_time()
        b_ske = S @ b
        z_ske = Q[:n_rows, :].T @ b_ske
        x_ske = la.solve_triangular(R, z_ske, lower=False)
        if np.linalg.norm(A @ x_ske - b) >= np.linalg.norm(b):
            z_ske = None
        log['time_presolve'] = quick_time() - tic

        # Iterative phase
        tic = quick_time()
        res = self.iterative_solver(A, b, None, 0.0, tol, iter_lim, R, True, z_ske)
        log['time_iterate'] = quick_time() - tic

        if logging:
            iters = res[3]
            # Record a vector of cumulative times to (1) sketch and factor, and
            # (2) take an individual step in LSQR (amortized!).
            #
            # Amortizing the time taken by a single step of LSQR is reasonable,
            # because convergence behavior can be seen by how the normal
            # equation error decays from one iteration to the next.
            time_setup = log['time_sketch']
            time_setup += log['time_factor']
            amortized = np.linspace(0, log['time_iterate'],
                                    iters, endpoint=True)
            cumulative = time_setup + log['time_presolve'] + amortized
            times = np.concatenate(([time_setup], cumulative))
            log['times'] = times

            arnorms = res[8][:iters]
            # Record a vector of (preconditioned) normal equation errors. Treat
            # the zero vector as a theoretically valid initialization point which
            # we would use before the "solve" phase of "sketch-and-solve".
            ar0 = A.T @ b
            ar0 = la.solve_triangular(R, ar0, 'T', lower=False, overwrite_b=True)
            ar0norm = la.norm(ar0)
            arnorms = np.concatenate(([ar0norm], arnorms))
            log['arnorms'] = arnorms

        return res[0], log


class SPO1(OverLstsqSolver):
    """A sketch-and-precondition approach to overdetermined ordinary least
    squares. This implementation uses the SVD to obtain the preconditioner
    and it uses LSQR for the iterative method.

    Before starting LSQR, we run a basic sketch-and-solve (for free, given
    our SVD of the sketched data matrix) to obtain a solution x_ske.
    If ||A x_ske - b||_2 < ||b||_2, then we initialize LSQR at x_ske.

    This implementation does not require that A is full-rank.

    References
    ----------
    This implementation was inspired by LSRN. The differences relative to the
    official LSRN algorithm [MSM:2014, Algorithm 1] are

        (1) We make no assumption on the distribution of the sketching operator.
            LSRN uses a Gaussian sketching operator.

        (2) We provide the option of intelligently initializing the iterative
            solver (LSQR) with the better of the two solutions given by the
            zero vector and the result of sketch-and-solve.
    """

    def __init__(self, sketch_op_gen, sampling_factor, smart_init):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.smart_init = smart_init
        self.iterative_solver = ris.PcSS2()  # LSQR

    def __call__(self, A, b, delta, tol, iter_lim, rng, logging=False):
        n_rows, n_cols = A.shape
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)

        log = {'time_sketch': -1.0,
               'time_factor': -1.0,
               'time_presolve': -1.0,
               'time_iterate': -1.0,
               'times': np.empty((1,)),
               'arnorms': np.empty((1,))}

        if logging:
            quick_time = time.time
        else:
            quick_time = lambda: 0.0

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        log['time_sketch'] = quick_time() - tic

        # Factor the sketch
        #   We also measure the time to scale the right singular vectors
        #   as needed for the preconditioner. SPO3 doesn't have a
        #   directly comparable cost.
        tic = quick_time()
        if delta > 0:
            reg_I = np.sqrt(delta) * np.eye(n_cols)
            A_ske = np.vstack((A_ske, reg_I))
        U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                              full_matrices=False)
        eps = np.finfo(float).eps
        rank = np.count_nonzero(sigma > sigma[0] * n_cols * eps)
        N = Vh[:rank, :].T / sigma[:rank]
        log['time_factor'] = quick_time() - tic

        if self.smart_init:
            tic = quick_time()
            b_ske = S @ b
            z_ske = (U[:n_rows, :][:, :rank]).T @ b_ske
            x_ske = N @ z_ske
            b_remainder = b - A @ x_ske
            if la.norm(b_remainder, ord=2) >= la.norm(b, ord=2):
                z_ske = None
            log['time_presolve'] = quick_time() - tic

            tic = quick_time()
            res = self.iterative_solver(A, b, None, delta, tol, iter_lim, N, False, z_ske)
            toc = quick_time()
            log['time_iterate'] = toc - tic
            x_star = res[0]
        else:
            # No presolve
            log['time_presolve'] = 0

            # Iterative phase
            tic = quick_time()
            res = self.iterative_solver(A, b, None, delta, tol, iter_lim, N, False, None)
            toc = quick_time()
            log['time_iterate'] = toc - tic
            x_star = res[0]

        if logging:
            iters = res[3]
            # Record a vector of cumulative times to (1) sketch and factor, and
            # (2) take an individual step in LSQR (amortized!).
            #
            # Amortizing the time taken by a single step of LSQR is reasonable,
            # because convergence behavior can be seen by how the normal
            # equation error decays from one iteration to the next.
            time_setup = log['time_sketch']
            time_setup += log['time_factor']
            amortized = np.linspace(0, log['time_iterate'],
                                    iters, endpoint=True)
            cumulative = time_setup + log['time_presolve'] + amortized
            times = np.concatenate(([time_setup], cumulative))
            log['times'] = times

            arnorms = res[8][:iters]
            # Record a vector of (preconditioned) normal equation errors. Treat
            # the zero vector as a theoretically valid initialization point which
            # we would use before the "solve" phase of "sketch-and-solve".
            ar0 = N.T @ (A.T @ b)
            ar0norm = la.norm(ar0)
            arnorms = np.concatenate(([ar0norm], arnorms))
            log['arnorms'] = arnorms

        return x_star, log


class UnderLstsqSolver:
    """Solver for underdetermined least squares problems"""

    def __call__(self, A, c, tol, iter_lim, rng, logging=False):
        raise NotImplementedError()


class SPU1(UnderLstsqSolver):
    """
    SVD-based sketch-and-precondition for underdetermined least squares

        min ||y||
        s.t. A' y = c

    where A is tall.
    """

    def __int__(self, sketch_op_gen, sampling_factor: int):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.iterative_solver = ris.PcSS2()  # implements LSQR

    def __call__(self, A, c, tol, iter_lim, rng, logging=False):
        n_rows, n_cols = A.shape
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)

        log = {'time_sketch': -1.0,
               'time_factor': -1.0,
               'time_iterate': -1.0,
               'times': np.empty((1,)),
               'arnorms': np.empty((1,))}

        if logging:
            quick_time = time.time
        else:
            quick_time = lambda: 0.0

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        log['time_sketch'] = quick_time() - tic

        # Factor the sketch
        #   We also measure the time to scale the right singular vectors
        #   as needed for the preconditioner. SPO3 doesn't have a
        #   directly comparable cost.
        tic = quick_time()
        U, sigma, Vh = la.svd(A_ske, overwrite_a=True, check_finite=False,
                              full_matrices=False)
        eps = np.finfo(float).eps
        rank = np.count_nonzero(sigma > sigma[0] * n_cols * eps)
        M = Vh[:rank, :].T / sigma[:rank]
        log['time_factor'] = quick_time() - tic

        # No presolve
        log['time_presolve'] = 0

        # Iterative phase
        tic = quick_time()
        res = self.iterative_solver(A, None, c, 0.0, tol, iter_lim, M, False, None)
        toc = quick_time()
        log['time_iterate'] = toc - tic
        y_star = res[1]

        return y_star, log

