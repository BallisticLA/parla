import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator
import numpy as np
from typing import Union
import parla.comps.determiter.saddle as dsad
import parla.comps.preconditioning as rpc
import parla.comps.sketchers.oblivious as sko
from parla.comps.sketchers.aware import RS1
from parla.drivers.least_squares import dim_checks, OverLstsqSolver
from parla.drivers.evd import EVD2
from parla.comps.determiter.logging import SketchAndPrecondLog
import time
import parla.utils.misc as misc
import parla.utils.linalg_wrappers as ulaw


NoneType = type(None)


class SaddleSolver:

    TEMPLATE_DOC_STR = \
    """
    Given a tall m-by-n data matrix A, an m-vector b, an n-vector c, and a
    nonnegative scalar delta, compute an approximate solution to the linear system
    
             [  I   |     A   ] [y_opt] = [b]           (*)
             [  A'  | -delta*I] [x_opt]   [c].
    
    The x_opt component of solutions to (*) is characterized by the normal equations
    
            (A' A + delta * I) x_opt = A'b - c.         (**)      
    %s
    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Tall data matrix.

    b : ndarray
        Long block vector in the right-hand-side. Should have b.shape = (m,).
        
    c : ndarray
        Short block vector in the right-hand-side. Should have c.shape = (n,).

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
    x_approx : ndarray
        x_approx.shape == (n,). Short block in the solution to (*).
    
    y_approx : ndarray
        y_approx.shape == (m,). Long block in the solution to (*).
    %s
    """

    INTERFACE_FIELDS = OverLstsqSolver.INTERFACE_FIELDS

    DOC_STR = TEMPLATE_DOC_STR % INTERFACE_FIELDS

    @misc.set_docstring(DOC_STR)
    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging):
        raise NotImplementedError()


def sps(A, b, c, delta, tol, iter_lim, rng, sampling_factor=3, vec_nnz=8, method='pcg'):
    skop = sko.SkOpSJ(vec_nnz)
    if method == 'pcg':
        solver = dsad.PcSS1()
    elif method == 'lsqr':
        solver = dsad.PcSS2()
    else:
        raise ValueError(f'Method {method} not recognized. Use "pcg" or "lsqr".')
    alg = SPS1(skop, sampling_factor, solver)
    return alg(A, b, c, delta, tol, iter_lim, rng, logging=True)


class SPS1(SaddleSolver):
    """
    SVD-based sketch-and-precondition for solving saddle point systems.
    """

    INTERFACE_FIELDS = (
        """
    This method can compute solutions to high accuracy. It computes the SVD of
    a sketch of A, and then calls SciPy's implementation of preconditioned 
    conjugate gradients (PCG).
        """,
        "Termination criteria used by SciPy's PCG implementation.",
        "Maximum number of iterations allowed by SciPy's PCG.",
        """
    log : SketchAndPrecondLog
        Contains runtime and per-iterate error metric information.
        The error of an individual iterate (x_i, y_i) is measured as\n
                || (A'A + delta * I) x_i - (A'b - c) ||_2.\n
        Note that y_i does not appear in that metric!
        Run help(log) or help(SketchAndPrecondLog) for more information.
        """
    )

    DOC_STR = SaddleSolver.TEMPLATE_DOC_STR % INTERFACE_FIELDS

    NYSTROM_STATEGIES = {'left-first', 'right-first'}

    def __init__(self, sketch_op_gen,
                 sampling_factor: int,
                 iterative_solver: Union[NoneType, dsad.PrecondSaddleSolver]):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        if iterative_solver is None:
            iterative_solver = dsad.PcSS1()
        self.iterative_solver = iterative_solver
        self.nystrom_strategy = 'left-first'
        pass

    @misc.set_docstring(DOC_STR)
    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging=True):
        m, n = A.shape
        sqrt_delta = np.sqrt(delta)
        d = int(self.sampling_factor * n)
        rng = np.random.default_rng(rng)
        assert self.nystrom_strategy in self.NYSTROM_STATEGIES

        if b is None:
            b = np.zeros(m)

        quick_time = time.time if logging else lambda: 0
        log = SketchAndPrecondLog()

        nystrom_like = d < n and delta == 0.0

        if not nystrom_like:
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
        else:  # delta == 0
            if self.nystrom_strategy == 'right_first':
                # Computes a Nystrom approximation of A'A.
                # This squares the condition num. of the preconditioner gen problem
                tic = quick_time()
                gram = lambda arg: A.T @ (A @ arg)
                gram_lo = LinearOperator(shape=(n, n), matvec=gram, matmat=gram)
                rso_ = RS1(self.sketch_op_gen, 0, ulaw.orth, 1)
                evd_ = EVD2(rso_)
                V, lamb = evd_(gram_lo, k=d, tol=np.NaN, over=0, rng=rng)
                lamb += delta
                M = V / np.sqrt(lamb)
                log.time_factor = quick_time() - tic
                log.time_sketch = 0.0  # attribute everything (incorrectly) to factoring
            else:
                # This computes P that solves min||P - A'A|| s.t. ker(P) = ker(A_ske),
                # where A_ske = S @ A.
                tic = quick_time()
                S = self.sketch_op_gen(d, m, rng)
                A_ske = S @ A
                log.time_sketch = quick_time() - tic
                # Process the sketch
                tic = quick_time()
                V = ulaw.orth(A_ske.T)  # A_ske.T is just a sample from the range of A'A.
                A_sample = A @ V
                U, sigma, Wt = la.svd(A_sample)
                sigma += (delta**0.5)
                M = V @ (Wt.T / sigma)
                log.time_factor = quick_time() - tic
            # end if: preconditioner generation via Nystrom
        # end if: preconditioner generation

        rhs = A.T @ b
        if c is not None:
            rhs -= c
        # Presolve ...
        #   (A_ske' A_ske ) x_ske = (A'b - c)                         (1, define)
        #   (V \Sigma^2 V') x_ske = (A'b - c)                         (2)
        #      \Sigma   V'  x_ske = \Sigma^{\dagger} V'(A'b - c)      (3)
        #   x_ske = V \Sigma^{\dagger} z_ske = M z_ske                (4, define)
        #   z_ske = \Sigma^{\dagger} V'(A'b - c)                      (5)
        tic = quick_time()
        if not nystrom_like:
            z_ske = (Vh @ rhs) / sigma
            x_ske = M @ z_ske
            rhs_pc = M.T @ rhs
            lhs_ske_pc = M.T @ (A.T @ (A @ x_ske) + delta*x_ske)
            if la.norm(lhs_ske_pc - rhs_pc, ord=2) >= la.norm(rhs_pc, ord=2):
                z_ske = None
        else:
            #TODO: properly initialize when using a low-rank sketch
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
    """Sketch, reduced to overdetermined least squares, and precondition.
    Use SVD to obtain the preconditioner and LSQR as the iterative solver."""

    INTERFACE_FIELDS = (
        """
    This method can compute solutions to high accuracy. It starts by computing the
    SVD of a sketch of A. It uses that SVD to convert the saddle point system into 
    an equivalent overdetermined least squares problem, and then it solves that
    problem by a preconditioned version of LSQR.
        """,
        """Termination criteria used by SciPy's LSQR implementation,
        as applied to a preconditioned version of the problem.""",
        "Maximum number of iterations allowed by SciPy's LSQR.",
        """
    log : SketchAndPrecondLog
        Contains runtime and per-iterate error metric information.
        
        Let M denote the preconditioner obtained by sketching.
        The error of an individual iterate (x_i, y_i) is measured as\n
                || (A_new M)' (A_new x_i - b_new) ||_2,\n
        where A_new is formed by stacking A on top of an identity matrix scaled
        by \\sqrt{delta} and b_new is chosen so that was chosen so that
        x_opt = argmin{ ||A_new x - b_new||_2 } solves (**).
        
        Under typical parameter settings, the condition number of (A_new M) is <= 10.
        Run help(log) or help(SketchAndPrecondLog) for more information.
        """
    )

    DOC_STR = SaddleSolver.TEMPLATE_DOC_STR % INTERFACE_FIELDS

    def __init__(self, sketch_op_gen,
                 sampling_factor: int,
                 iterative_solver: Union[NoneType, dsad.PrecondSaddleSolver]):
        self.sketch_op_gen = sketch_op_gen
        self.sampling_factor = sampling_factor
        if iterative_solver is None:
            iterative_solver = dsad.PcSS2()
        self.iterative_solver = iterative_solver
        pass

    @misc.set_docstring(DOC_STR)
    def __call__(self, A, b, c, delta, tol, iter_lim, rng, logging=False):
        m, n = A.shape
        sqrt_delta = np.sqrt(delta)
        d = dim_checks(self.sampling_factor, m, n)
        rng = np.random.default_rng(rng)

        if b is None:
            b = np.zeros(m)

        quick_time = time.time if logging else lambda: 0
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
