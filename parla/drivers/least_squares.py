"""
Routines for (approximately) solving strongly overdetermined or strongly
underdetermined least squares problems.
"""
import warnings
import scipy.linalg as la
import numpy as np
import parla.comps.sketchers.oblivious as sko
import parla.comps.preconditioning as rpc
import parla.comps.determiter.saddle as dsad
import time
import parla.utils.misc as misc
import parla.utils.sketching as usk
from parla.comps.determiter.logging import SketchAndPrecondLog


class OverLstsqSolver:
    """Solver for overdetermined least squares problems."""

    TEMPLATE_DOC_STR = \
    """
    Given a tall m-by-n data matrix A, return an approximate solution to

        min{ ||A x - b||_2^2 + delta * ||x||_2^2 : x in R^n }.
    %s
    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Tall data matrix.

    b : ndarray
        Right-hand-side. Should have b.ndim == 1.

    delta : float
        Nonnegative regularization parameter.

    tol : float
        %s

    iter_lim : int
        %s

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages any and all
        randomness in this function call.

    Returns
    -------
    x_star : ndarray
        x_star.shape == (n,). Approximate solution to the least
        squares problem under consideration.
    %s
    """

    #NOTE: the strings below are also used in UnderLstsqSolver and SaddleSolver.
    INTERFACE_FIELDS = (
    """
    There is no requirement that an implementation is able to control
    the error of its returned solution.
    """,
    """This parameter is only relevant for implementations that involve
        some iterative method. Those implementations must have some kind
        of error metric for a candidate solution. If the implementation's
        measurement of error falls below tol, then it returns the candidate solution.
    
        If an implementation does not use an iterative method and receives
        tol that is is not NaN, then a warning will be raised.""",
    """We require iter_lim > 0. Typically, iter_lim << n.
        This parameter is only relevant for implementations that involve
        some kind of iterative method. Those implementations must terminate
        after iter_lim iterations.
    
        If an implementation does not use an iterative method and receives
        iter_lim > 1, then a warning will be raised.""",
    """
    log : Union[dict, SketchAndPrecondLog]
        If a dict, then log is keyed by strings. It contains runtime information.
        If a SketchAndPrecondLog, then it contains runtime and error metric
        information (refer to SketchAndPrecondLog docs for more info).
    """
    )

    DOC_STR = TEMPLATE_DOC_STR % INTERFACE_FIELDS

    @misc.set_docstring(DOC_STR)
    def __call__(self, A, b, delta, tol, iter_lim, rng):
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


#TODO: write docstring
def sso1(A, b, delta, rng, sampling_factor=3, vec_nnz=8, lapack_driver='gelsd'):
    skop = sko.SkOpSJ(vec_nnz)
    alg = SSO1(skop, sampling_factor, lapack_driver, overwrite_sketch=True)
    return alg(A, b, delta, np.NaN, 1, rng, logging=True)


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

    INTERFACE_FIELDS = (
        """
    This is a one-shot method, suitable for finding only rough approximations.
        """,
        "Not processed. Set to NaN to avoid warning messages.",
        "Not processed. Set to 1 to avoid warning messages.",
        """
    log : dict
        log['time_sketch'] is the time spent sketching (A, b).
        log['time_solve'] is the time spent solving the sketched problem.
        """
    )

    CALL_DOC = OverLstsqSolver.TEMPLATE_DOC_STR % INTERFACE_FIELDS

    def __init__(self, sketch_op_gen, sampling_factor, lapack_driver=None,
                 overwrite_sketch=True):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.lapack_driver = lapack_driver
        self.overwrite_sketch = overwrite_sketch

    @misc.set_docstring(CALL_DOC)
    def __call__(self, A, b, delta, tol, iter_lim, rng, logging=True):
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

        log = {'time_sketch': -1.0, 'time_solve': -1.0}

        quick_time = time.time if logging else lambda: 0

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


#TODO: write docstring
def spo1(A, b, delta, tol, iter_lim, rng, sampling_factor=3, vec_nnz=8):
    skop = sko.SkOpSJ(vec_nnz)
    alg = SPO(skop, sampling_factor, mode='svd')
    return alg(A, b, delta, tol, iter_lim, rng, logging=True)


#TODO: write docstring
def spo3(A, b, delta, tol, iter_lim, rng, sampling_factor=3, vec_nnz=8, mode='qr'):
    skop = sko.SkOpSJ(vec_nnz)
    alg = SPO(skop, sampling_factor, mode)
    return alg(A, b, delta, tol, iter_lim, rng, logging=True)


class SPO(OverLstsqSolver):
    """A sketch-and-precondition approach to overdetermined ordinary least
    squares. This implementation uses QR, Cholesky, or SVD to obtain the
    preconditioner, and it uses LSQR for the iterative method.
    Before starting LSQR, we run a basic sketch-and-solve (for free, given
    our decomposition of the sketched data matrix) to obtain a solution
    x_ske. If ||A x_ske - b||_2 < ||b||_2, then we initialize LSQR at x_ske.
    If A is rank-deficient, then the preconditioner must be obtained by SVD.
    References
    ----------
    This implementation was inspired by Blendenpik (AMT:2010) and LSRN (MSM:2014).
    The differences relative to [AMT:2010, Algorithm 1] are
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
        (6) We let the user choose whether the upper-triangular preconditioner
            is obtained by QR or Cholesky. (They are equivalent in exact
            arithmetic but have different numerical profiles.)
    The differences relative to [MSM:2014, Algorithm 1] are
        (1) LSRN uses the Chebyshev semi-iterative method instead of LSQR.
        (1) We make no assumption on the distribution of the sketching operator.
            LSRN uses a Gaussian sketching operator.
        (2) We provide the option of intelligently initializing the iterative
            solver (LSQR) with the better of the two solutions given by the
            zero vector and the result of sketch-and-solve.
    """

    INTERFACE_FIELDS = (
        """
    This method can compute solutions to high accuracy. It computes either
        (1) the unpivoted QR decomposition of a sketch of A,
        (2) the Cholesky decomposition of a sketched Gram matrix, or
        (3) the SVD of a sketch of A
    and then calls a preconditioned version of LSQR.
        """,
        """Termination criteria used by SciPy's LSQR implementation,
        as applied to a preconditioned version of the problem.""",
        "Maximum number of iterations allowed by SciPy's LSQR.",
        """
    log : SketchAndPrecondLog
        Contains runtime and per-iterate error metric information.

        Let M denote the preconditioner obtained by sketching.
        The error of an individual iterate x_i is measured as\n
                || (A_new M)' (A_new x_i - b_new) ||_2,\n
        where A_new is formed by stacking A on top of an identity matrix scaled
        by \\sqrt{delta} and b_new is formed by stacking b on top of the zero vector.

        Under typical parameter settings, the condition number of (A_new M) is <= 10.
        Run help(log) or help(SketchAndPrecondLog) for more information.
        """
    )

    CALL_DOC = OverLstsqSolver.TEMPLATE_DOC_STR % INTERFACE_FIELDS

    def __init__(self, sketch_op_gen, sampling_factor: int, mode='qr'):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.mode = mode
        self.iterative_solver = dsad.PcSS2()  # implements LSQR

    @misc.set_docstring(CALL_DOC)
    def __call__(self, A, b, delta, tol, iter_lim, rng, logging=True, logging_condnum_precond=False):
        n_rows, n_cols = A.shape
        sqrt_delta = np.sqrt(delta)
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        rng = np.random.default_rng(rng)

        quick_time = time.time if logging else lambda: 0
        log = SketchAndPrecondLog()

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        log.time_sketch = quick_time() - tic

        # Factor the sketch. Start sketch-and-solve style presolve.
        if self.mode == 'qr':
            tic = quick_time()
            if delta > 0:
                A_ske = np.vstack((A_ske, sqrt_delta * np.eye(n_cols)))
            Q, R = la.qr(A_ske, overwrite_a=True, mode='economic')
            if logging_condnum_precond:
                A_pre = la.solve_triangular(R, A.T, trans='T').T
                log.condnum_precond = np.linalg.cond(A_pre)
                #log.condnum_precond = np.linalg.cond(A @ la.inv(R))
            log.time_factor = quick_time() - tic
            tic = quick_time()
            b_ske = S @ b
            z_ske = Q[:d, :].T @ b_ske
            x_ske = la.solve_triangular(R, z_ske, lower=False)
        elif self.mode == 'chol':
            tic = quick_time()
            G = np.eye(n_cols)
            G = la.blas.dsyrk(1.0, A_ske, beta=delta, c=G, trans=1, lower=False,
                              overwrite_c=True)
            rows, cols = np.triu_indices(n_cols)
            G[cols, rows] = G[rows, cols]
            R = la.cholesky(G, overwrite_a=True, check_finite=False)
            log.time_factor = quick_time() - tic
            tic = quick_time()
            b_ske = S @ b
            z_ske = la.solve_triangular(R, A_ske.T @ b_ske, lower=False, trans='T')
            x_ske = la.solve_triangular(R, z_ske, lower=False)
        elif self.mode == 'svd':
            tic = quick_time()
            R, U, sigma, Vh = rpc.svd_right_precond(A_ske)
            log.time_factor = quick_time() - tic
            tic = quick_time()
            b_ske = S @ b
            z_ske = U[:d, :].T @ b_ske
            if delta == 0:
                x_ske = R @ z_ske
            else:
                x_ske = Vh.T @ (z_ske / np.sqrt(sigma**2 + delta))
        else:
            raise ValueError()

        # Complete sketch-and-solve style presolve
        r = A @ x_ske - b
        if delta > 0:
            r = np.concatenate((r, sqrt_delta * x_ske))
        rel_err = la.norm(r) / la.norm(b)
        if rel_err >= 1 or (rel_err > 1e-15 and R.shape[0] != R.shape[1]):
            # Either the zero vector is a better solution, or we have an inconsistent
            # rank-deficient problem (which forces us to initialize at the origin).
            z_ske = None
        log.time_presolve = quick_time() - tic

        # Iterative phase
        tic = quick_time()
        tri = self.mode in {'qr', 'chol'}
        res = self.iterative_solver(A, b, None, delta, tol, iter_lim, R, tri, z_ske)
        log.time_iterate = quick_time() - tic
        if len(res) == 4:
            log.return_code = res[3]

        if logging:
            ar0 = A.T @ b
            if tri:
                ar0 = la.solve_triangular(R, ar0, 'T', lower=False, overwrite_b=True)
            else:
                ar0 = R.T @ ar0
            log.wrap_up(res[2], la.norm(ar0))
            log.error_desc = self.iterative_solver.ERROR_METRIC_INFO

        return res[0], log


class UnderLstsqSolver:
    TEMPLATE_DOC_STR = """
    Given a tall m-by-n matrix A and an n-vector c, compute an approximate
    solution to

        min ||y||
        s.t. A' y = c.

    If the solution is computed to low accuracy, then the equality
    constraint "A' y = c" might be violated by a large margin.
    %s
    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Tall data matrix, where A' specifies an underdetermined least-squares problem.

    c : ndarray
        Right-hand-side. Should have c.ndim == 1.

    tol : float
        %s

    iter_lim : int
        %s

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages any and all
        randomness in this function call.

    Returns
    -------
    y_star : ndarray
        y_star.shape == (m,). Approximate solution to the underdetermined
        least squares problem under consideration.
    %s
    """

    INTERFACE_FIELDS = ('',) + OverLstsqSolver.INTERFACE_FIELDS[1:]

    DOC_STR = TEMPLATE_DOC_STR % INTERFACE_FIELDS

    @misc.set_docstring(DOC_STR)
    def __call__(self, A, c, tol, iter_lim, rng, logging=False):
        raise NotImplementedError()


#TODO: write docstring
def spu1(A, c, tol, iter_lim, rng, sampling_factor=3, vec_nnz=8):
    skop = sko.SkOpSJ(vec_nnz)
    alg = SPU1(skop, sampling_factor)
    return alg(A, c, tol, iter_lim, rng, logging=True)


class SPU1(UnderLstsqSolver):
    """
    SVD-based sketch-and-precondition for underdetermined least-squares.
    We parameterize underdetermined least-squares with a tall m-by-n
    data matrix A and an n-vector c. In full generality, this algorithm
    finds an m-vector y that approximately minimizes

        ||y - pinv(A') c||_2.
    """

    INTERFACE_FIELDS = (
        """
    This method can compute solutions to high accuracy. It computes the SVD of
    a sketch of A, and then calls a preconditioned version of LSQR.
        """,) + SPO.INTERFACE_FIELDS[1:3] + (
        """
    log : SketchAndPrecondLog
        Contains runtime and per-iterate error metric information.
        
        Let M denote the preconditioner obtained by random sketching.
        The error of an individual iterate y_i is measured as\n
                || (A M) ((A M)' y_i - M' c) ||_2.\n
                
        Under typical parameter settings, the condition number of A M is <= 10.
        Run help(log) or help(SketchAndPrecondLog) for more information.""",)

    CALL_DOC = UnderLstsqSolver.TEMPLATE_DOC_STR % INTERFACE_FIELDS

    def __init__(self, sketch_op_gen, sampling_factor: int):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        self.iterative_solver = dsad.PcSS2()  # implements LSQR

    @misc.set_docstring(CALL_DOC)
    def __call__(self, A, c, tol, iter_lim, rng, logging=True):
        n_rows, n_cols = A.shape
        d = dim_checks(self.sampling_factor, n_rows, n_cols)
        rng = np.random.default_rng(rng)

        quick_time = time.time if logging else lambda: 0
        log = SketchAndPrecondLog()

        # Sketch the data matrix
        tic = quick_time()
        S = self.sketch_op_gen(d, n_rows, rng)
        A_ske = S @ A
        log.time_sketch = quick_time() - tic

        # Factor the sketch
        tic = quick_time()
        M, U, sigma, Vh = rpc.svd_right_precond(A_ske)
        log.time_factor = quick_time() - tic

        # Iterative phase
        tic = quick_time()
        res = self.iterative_solver(A, None, c, 0.0, tol, iter_lim, M, False, None)
        toc = quick_time()
        log.time_iterate = toc - tic
        y_star = res[1]

        if logging:
            log.wrap_up(res[2], la.norm(A @ (M @ (M.T @ c))))
            log.error_desc = """
            The logs produced by this algorithm measure error as\n
                || (A M) (A M)' y - (A M) (M' c) ||_2,\n
            where "M" is a right-preconditioner for A. Under typical
            parameter settings, the condition number of A M is <= 10.
            """

        return y_star, log

