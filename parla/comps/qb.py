import warnings

import numpy as np
import scipy.linalg as la
import parla.comps.sketchers.oblivious as oblivious
from parla.comps.rangefinders import RangeFinder, RF1
from parla.comps.sketchers.aware import RowSketcher, RS1
import parla.utils.linalg_wrappers as ulaw


###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def qb(num_passes, A, k, rng):
    """
    Return matrices (Q, B) from a rank-k QB factorization of A.
    Use a Gaussian sketching matrix and pass over A a total of
    num_passes times.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A. We require num_passes >= 2.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    k : int
        Target rank for the approximation of A: 0 < k < min(A.shape).
        This parameter includes any oversampling. For example, if you
        want to be near the optimal (Eckhart-Young) error for a rank 20
        approximation of A, then you might want to set k=25.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    B : ndarray
        Has shape (k, A.shape[1]).

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.

    References
    ----------
    This algorithm computes Q and then sets B = Q.T @ A. Conceptually, we
    compute Q by using Algorithm 4.3 (see also Algorithm 4.4) from

        Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
        "Finding structure with randomness: Probabilistic algorithms for
        constructing approximate matrix decompositions."
        SIAM review 53.2 (2011): 217-288.
        (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    The precise subspace iteration technique is similar to that of Algorithm
    3.3 from

         Bolong Zhang and Michael Mascagni.
         "Pass-Efficient Randomized LU Algorithms for Computing Low-Rank
         Matrix Approximation"
         arXiv:2002.07138 (2020).

    There are two differences between this implementation and Zhang and
    Mascagni's Algorithm 3.3: we use QR decompositions where they use LU
    decompositions, and we allow 0 steps or 1 step of subspace iteration,
    where their implementation requires >= 2 steps.
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), num_passes - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB1(rf_)
    Q, B = qb_(A, k, np.NaN, rng)
    return Q, B


def qb_b(inner_num_pass, blk, overwrite_A, A, k, tol, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1)  || A - Q B ||_Fro <= tol * || A ||_Fro
    or
        (2) Q has k columns.

    Each iteration involves sketching A from the right by a sketching
    matrix with "blk" columns, and reading through A inner_num_pass times.

    Parameters
    ----------
    inner_num_pass : int
        Number of passes over A in each iteration of this blocked QB
        algorithm. We require inner_num_pass >= 2.

    blk : int
        The block size in this blocked QB algorithm. Add this many columns
        to Q at each iteration (except possibly the final iteration).

    overwrite_A : bool
        If True, then this method modifies A in-place. If False, then
        we start the algorithm by constructing a complete copy of A.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    k : int
        Terminate if Q.shape[1] == k. Assuming k < rank(A), setting tol=0 is a
        valid way of ensuring Q.shape[1] == k on exit.

    tol : float
        Terminate if ||A - Q B||_Fro <= tol * || A ||_Fro. Setting k = min(A.shape)
        is a valid way to ensure that this is the exit condition.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has the same number of rows of A, and orthonormal columns.

    B : ndarray
        Has the same number of columns of A.

    Notes
    -----
    The number of columns in Q increase by "blk" at each iteration, unless
    that would bring Q.shape[1] > k. In that case, the final iteration only
    adds enough columns to Q so that Q.shape[1] == k.

    We perform (inner_num_pass - 2) steps of subspace iteration for each
    block of the QB factorization. We stabilize subspace iteration with
    QR factorization at each step.

    The implementation is built up as
        RS1(RowSketcher) --> RF1(RangeFinder) --> QB2(QBDecomposer)

    References
    ----------
    This implements a variant of Algorithm 2 from YGL:2018. There are two
    main differences.

        (1) We allow subspace iteration when building a new block (Qi, Bi)
            of the QB factorization. That is, [YGL:2018, Algorithm 2] requires
            inner_num_pass = 2.

        (2) We have to explicitly update A.

    Straightforward changes to the implementation of QB2.__call__(...) would remove
    the second of these differences. Refer to QB2.__call__(...) for more
    information.
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), inner_num_pass - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB2(rf_, blk, overwrite_A)
    Q, B = qb_(A, k, tol, rng)
    return Q, B


def qb_b_pe(num_passes, blk, A, k, tol, rng):
    """
    Iteratively build an approximate QB factorization of A,
    which terminates once either of the following conditions
    is satisfied
        (1) || A - Q B ||_Fro <= tol * || A ||_Fro
    or
        (2) Q has k columns.

    We start by obtaining a sketching matrix of shape (A.shape[1], k),
    using (num_passes - 1) steps of subspace iteration on a random Gaussian
    matrix with k columns. Then we perform two more passes over A before
    beginning iterative construction of (Q, B). Each iteration adds at most
    "blk" columns to Q and rows to B.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A in an efficient implementation
        of this algorithm (see Notes). We require num_passes >= 1.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    blk : int
        The block size in this blocked QB algorithm. Add this many
        columns to Q at each iteration (except possibly the final iteration).

    k : int
        Terminate if Q.shape[1] == k.

    tol : float
        Terminate if ||A - Q B||_Fro <= tol * || A ||_Fro.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    Q : ndarray
        Has the same number of rows of A, and orthonormal columns.

    B : ndarray
        Has the same number of columns of A.

    Notes
    -----
    With its current implementation, this function requires num_passes + 1
    passes over A. An efficient implementation using two-in-one sketching
    could run this algorithm using only num_passes passes over A.

    We stabilize subspace iteration with a QR factorization at each step.

    References
    ----------
    This implements a variant of [YGL:2018, Algorithm 4]. The difference is
    that [YGL:2018, Algorithm 4]'s subspace iteration requires an even number
    of passes over A, while our subspace iteration can perform any number of
    passes over A.
    """
    rng = np.random.default_rng(rng)
    sk_op = RS1(oblivious.SkOpGA(), num_passes, ulaw.orth, 1)
    Q, B = QB3(sk_op, blk)(A, k, tol, rng)
    return Q, B


###############################################################################
#       Object-oriented interfaces
###############################################################################

class QBDecomposer:

    TOL_CONTROL = 'none'

    def __call__(self, A, k, tol, rng):
        """
        Return a matrix Q with orthonormal columns and a matrix B where
        the product Q B stands in as an approximation of A.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to be approximated.

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            Typically, k << min(A.shape). Conformant implementations ensure
            Q has at most k columns. For certain implementations it's
            reasonable to choose k as large as k = min(A.shape), in which
            case the implementation returns only once a specified error
            tolerance has been met.

        tol : float
            0 <= tol < 1. Target for the relative error  ||A - Q B|| / ||A|| measured
            in some norm (often Frobenius). Only certain implementations are able to
            control approximation error. Those implementations may return a matrix Q
            with fewer than k columns if ||A - Q B|| <= ||A|| * tol. Assuming
            k < rank(A) and that the implementation can compute ||A - Q B|| accurately,
            setting tol=0 means the implementation will return (Q, B) with exact rank k.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Has the same number of rows as A, and orthonormal columns.

        B : ndarray
            B = Q.T @ A (although not necessarily computed in the way).
        """
        raise NotImplementedError()


class QB1(QBDecomposer):
    """
    Direct reduction to the rangefinder problem. Given a rangefinder's output
    Q, we set B = Q.T @ A and return (Q, B).
    """

    TOL_CONTROL = 'unknown'  # depends on implementation of rangefinder

    def __init__(self, rf: RangeFinder):
        """
        Parameters
        ----------
        rf : RangeFinder
            Q = rf(A, k, tol, rng) has orthonormal columns. Its range
            is supposed to approximate the space spanned by the leading left
            singular vectors of A. The implementation constructed Q with
            target rank "k" and target tolerance "tol".
        """
        self.rangefinder = rf

    def __call__(self, A, k, tol, rng):
        """
        Rely on a rangefinder to obtain the matrix Q for the decomposition
        A \approx Q B. Once we have Q, we construct B = Q.T @ A and return
        (Q, B). This function is agnostic to the implementation of the
        rangefinder: it might build a rank-k matrix Q all at once or construct
        successively larger matrices Q by an iterative process. We make no
        assumptions on the rangefinder's termination criteria beyond those
        listed below.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            This parameter is passed directly to the rangefinder.

        tol : float
            Target for the error ||A - Q B||: 0 <= tol < np.inf.
            This parameter is passed directly to the rangefinder.
            Note that since we construct B := Q.T @ A, we have
            ||A - Q B|| = ||A  - Q Q' A||.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Output of the underlying rangefinder.

        B : ndarray
            Equal to Q.T @ A.
        """
        assert k > 0
        assert k <= min(A.shape)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < 1
        rng = np.random.default_rng(rng)
        Q = self.rangefinder(A, k, tol, rng)
        B = Q.T @ A
        return Q, B


class QB2(QBDecomposer):
    """
    Common uses of a QB2 object "qb_alg"

        qb_alg(A, min(A.shape), tol, rng) will return
        an approximation (Q, B) where ||A - Q B ||_Fro <= tol * ||A||_Fro.

        qb_alg(A, k, 0.0, rng) will return (Q, B)
        where Q has k columns.
    """

    TOL_CONTROL = 'full'

    def __init__(self, rf: RangeFinder, blk: int, overwrite_a: bool):
        self.rangefinder = rf
        self.blk = blk
        self.overwrite_a = overwrite_a

    def __call__(self, A, k, tol, rng):
        """
        Build a QB factorization by iteratively adding columns to Q
        and rows to B. The algorithm modifies A in-place. If
        self.overwrite_a = False, then a copy of A is made at the start
        of this function call. We start by initializing Q, B with shapes
        (A.shape[0], 0) and (0, A.shape[1]), setting abs_tol = ||A||_Fro * tol,
        and we roughly proceed as follows

            cur_blk = min(k - Q.shape[1], self.blk)
            if cur_blk == 0 or ||A||_Fro <= abs_tol:
                return Q, B
            Qi = rangefinder(A, cur_blk, 0.0, rng)
            Bi = Qi.T @ A
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            A -= Qi @ Bi

        This function differs from the code above in how it stabilizes
        certain computations and avoids recomputing the Frobenius norm
        of A at each iteration.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            If Q has k columns, then return (Q, B). Assuming k < rank(A),
            setting tol=0 is a valid way of ensuring that Q has k columns
            on exit.

        tol : float
            Target for the relative error  ||A - Q B||_Fro / ||A||_Fro:
            0 <= tol < 1. If the relative error drops below  tol,
            then return (Q, B). Setting k = min(A.shape) is a valid way
            of ensuring ||A - Q B||_Fro / ||A||_Fro <= tol on exit.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Output of the underlying rangefinder.

        B : ndarray
            Equal to Q.T @ A (although not computed in that way).

        References
        ----------

        This implements a variant of Algorithm 2 from YGL:2018. There are two
        main differences.

            (1) We allow any rangefinder when building a new block (Qi, Bi),
                while [YGL:2018, Algorithm 2] uses the elementary single-pass
                rangefinder Yi = A @ Si, Qi = orth(Yi).

            (2) We have to explicitly update A.

        The second difference can be reconciled by (mathematically) calling the
        rangefinder on the linear operator L = A - Q @ B. In particular, although
        our algorithm forms Bi = Qi.T @ A on the updated version of A, it is
        mathematically equivalent to perform that step on the original matrix A
        (as is done in [YGL:2018, Algorithm 2, Line 7]).
        """
        if not self.overwrite_a:
            A = np.copy(A)
        assert k > 0
        small_dim = min(A.shape)
        if not k <= small_dim:
            msg = f"""
            The target rank k = {k} is larger than min({A.shape}).
            We will proceed with target rank k = {small_dim}.
            """
            k = small_dim
            warnings.warn(msg)
        assert k <= min(A.shape)
        use_tol = not np.isnan(tol) and tol > 0
        if use_tol:
            sq_norm_A = la.norm(A, ord='fro') ** 2
            sq_tol = tol ** 2
            abs_sq_tol = sq_norm_A * sq_tol
        rng = np.random.default_rng(rng)
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        blk = self.blk
        while True:
            if B.shape[0] + blk > k:
                blk = k - B.shape[0]  # final block
            # Standard QB, but step in to make extra sure that
            #   the columns of "Qi" are orthogonal to cols of current "Q".
            Qi = self.rangefinder(A, blk, np.NaN, rng)
            Qi = project_out(Qi, Q, as_list=False)
            Qi = la.qr(Qi, mode='economic')[0]
            Bi = Qi.T @ A
            # Update the full factorization
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            A -= Qi @ Bi
            if use_tol:
                sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro') ** 2
                if sq_norm_A <= abs_sq_tol:
                    break
            if B.shape[0] >= k:
                break
        return Q, B


class QB3(QBDecomposer):

    TOL_CONTROL = 'early stopping'

    def __init__(self, sk_op: RowSketcher, blk: int):
        self.sk_op = sk_op
        self.blk = blk

    def __call__(self, A, k, tol, rng):
        """
        Build a QB factorization of A by constructing a suitable sketching
        operator S with S.shape = (A.shape[1], k) and then constructing
        G = A @ S and H = A.T @ G. Once (G, H, S) are in hand, we process these
        matrices in blocks of size "self.blk" at a time (except in the last
        iteration, where we might process a smaller block). While processing
        the blocks we monitor tolerance-based early stopping criteria.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            Target for the number of columns in Q: 0 < k < min(A.shape).
            If Q has k columns, then return (Q, B). Assuming k < rank(A),
            setting tol=0 is a valid way of ensuring that Q has k columns
            on exit. Note that this implementation requires strict
            inequality k < min(A.shape).

        tol : float
            Early-stopping target for the error relative ||A - Q B||_Fro / || A ||_Fro.
            If the relative error falls below tol, then return (Q, B).
            There is no way of ensuring that this target is achieved. Setting tol=0
            skips the computations that are typically necessary for monitoring
            early-stopping.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Output of the underlying rangefinder.

        B : ndarray
            Equal to Q.T @ A (although not computed in that way).

        Notes
        -----
        With its current implementation, this function requires
        self.num_passes + 1 passes over A. An efficient implementation
        using two-in-one sketching could run this algorithm using only
        self.num_passes passes over A.

        References
        ----------
        The basic approach for this algorithm comes from [YGLLL:2017,
        Algorithm 3], which is expanded upon in [YGL:2018, Algorithm 4].
        The latter algorithm includes support for subspace iteration,
        which serves the purpose of aligning the columns of the sketching
        matrix S with the leading right singular vectors of A.

        This implementation generalizes [YGL:2018, Algorithm 4] by being
        agnostic to how S is formed. We obtain it by calling S =
        self.sk_op(A, k, rng), where self.sk_op is a RowSketcher.

        Subspace iteration can be used to implement a RowSketcher's
        "__call__" function, but that is not the only possible implementation.
        Refer the the RowSketcher interface for more information.
        """
        assert k > 0
        assert k < min(A.shape)
        use_tol = not np.isnan(tol) and tol > 0
        if use_tol:
            sq_norm_A = la.norm(A, ord='fro') ** 2
            sq_tol = tol ** 2
            abs_sq_tol = sq_norm_A * sq_tol
        Q = np.empty(shape=(A.shape[0], 0), dtype=float)
        B = np.empty(shape=(0, A.shape[1]), dtype=float)
        rng = np.random.default_rng(rng)
        blk = self.blk
        S = self.sk_op(A, k, rng)
        if not isinstance(S, np.ndarray):
            msg = """
            This implementation requires the sketching routine to return a 
            dense matrix, as represented by a numpy ndarray. We received a 
            matrix of type %s
            """ % str(type(S))
            raise RuntimeError(msg)
        G = A @ S
        H = A.T @ G
        for i in range(int(np.ceil(k/blk))):
            blk_start = i*blk
            blk_end = min((i+1)*blk, k)
            Si = S[:, blk_start:blk_end]
            BSi = B @ Si
            Yi = G[:, blk_start:blk_end] - Q @ BSi
            Qi, Ri = la.qr(Yi, mode='economic')
            Qi = project_out(Qi, Q, as_list=False)  # Qi = Qi - Q @ (Q.T @ Qi)
            Qi, Rihat = la.qr(Qi, mode='economic')
            Ri = Rihat @ Ri
            Bi = H[:, blk_start:blk_end].T - (Yi.T @ Q) @ B - BSi.T @ B
            Bi = la.solve_triangular(Ri, Bi, trans='T', overwrite_b=True)
            Q = np.column_stack((Q, Qi))
            B = np.row_stack((B, Bi))
            if use_tol:
                sq_norm_A = sq_norm_A - la.norm(Bi, ord='fro')**2
                if sq_norm_A <= abs_sq_tol:
                    break  # early stopping
        return Q, B


###############################################################################
#      Helper functions
###############################################################################


def project_out(Qi, Q, as_list=False):
    #TODO: perform operation in-place.
    if as_list:
        #TODO: implement and use in qb_b_fet.
        # NOTE: Q is accessed in a few different places in
        #       qb_b_pe, so this wouldn't be enough to avoid
        #       updating Q to be contiguous at each iteration.
        raise NotImplementedError()
    else:
        Qi = Qi - Q @ (Q.T @ Qi)
        return Qi
