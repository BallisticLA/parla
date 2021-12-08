import numpy as np
import scipy.linalg as la
from parla.comps.qb import QBDecomposer, QB2
from parla.comps.rangefinders import RF1
from parla.comps.sketchers.aware import RS1
import parla.comps.sketchers.oblivious as oblivious
import parla.utils.linalg_wrappers as ulaw


class SVDecomposer:

    def __call__(self, A, k, tol, over, rng):
        """
        Return U, s, Vh where, for some integer ell <= k,
            U is A.shape[0]-by-ell,
            s is a vector of length ell,
            Vh is ell-by-A.shape[1],
        so that
            A \approx U @ diag(s) @ Vh

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate

        k : int
            The returned SVD will be truncated to at most rank k:
            0 < k <= min(A.shape). Setting k=min(A.shape) and over=0
            ensures ||A - U @ diag(s) @ Vh|| / || A || <= tol on exit. However,
            setting k=min(A.shape) may trivially return the SVD of
            A in some implementations.

        tol : float
            The target error used by the randomized part of the algorithm.
            When over = 0, this parameter controls ||A - U @ diag(s) @ Vh|| / || A ||
            (usually in Frobenius norm). The precise meaning of "tol" when over > 0
            is implementation-dependent.

        over : int
            The randomized part of the algorithm uses k+over as the target rank;
            we require over >= 0 and k+over <= min(A.shape).
            In a conformant implementation, that part of the algorithm will
            never return a factorization of rank greater than k+over.

            Setting over > 0 will likely result in truncating the SVD obtained
            from the randomized part of the algorithm. If you want to control
            the truncation step yourself, then you should set over=0 and
            increase the value of k. E.g., a function call with over=5 and
            k=20 can avoid truncation by setting k=25 and over=0.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.
        """
        raise NotImplementedError()


def svd1(A, k, over, tol, inner_num_pass, block_size, rng):
    """
    Return ndarrays (U, s, Vh) that define a matrix "A_approx" through its SVD:

        A_approx = (U * s) @ Vh.

    The columns of U approximate the leading left singular vectors of A.
    The rows of Vh approximate of the leading right singular vectors of A.
    The entries of s are the corresponding approximate singular values.

    This function builds A_approx by an iterative process. Each iteration
    involves sketching A with a matrix that has "block_size" columns, and
    reading through A inner_num_pass times. The iterative process stops once
    A_approx has reached sufficiently high rank (at most k + over) or becomes
    sufficiently close to A in Frobenius norm. A_approx is always truncated
    to rank (at most) k before returning.


    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate

    k : int
        The returned SVD will be truncated to at most rank k:
        0 < k <= min(A.shape). Setting k=min(A.shape) and over=0
        ensures ||A - U @ diag(s) @ Vh||_F / || A ||_F <= tol on exit.

    over : int
        The randomized part of the algorithm uses k + over as the target rank;
        we require over >= 0 and k+over <= min(A.shape).

        Setting over > 0 will likely result in truncating the matrix "A_approx"
        before the function returns. If you want to control the truncation step
        yourself, then you should set over=0 and increase the value of k.

    tol : float
        The target error used by the randomized part of the algorithm.
        When over = 0, this parameter controls || A - A_approx ||_F / || A ||_F.

    inner_num_pass : int
        Number of passes over A in each iteration of the algorithm.
        We require inner_num_pass >= 2.

    block_size : int
        The rank of the working approximation "A_approx" is increased by "block_size"
        at each iteration.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    (U, s, Vh) where, for some integer ell <= k,
            U is A.shape[0]-by-ell,
            s is a vector of length ell,
            Vh is ell-by-A.shape[1].
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), inner_num_pass - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB2(rf_, block_size, overwrite_a=False)
    alg = SVD1(qb_)
    U, s, Vh = alg(A, k, tol, over, rng)
    return U, s, Vh


class SVD1(SVDecomposer):
    """
    Use the Phase 1 / Phase 2 approach from HMT2011, where Phase 1 is
    implemented by an arbitrary QB factorization method.
    """

    def __init__(self, qb: QBDecomposer):
        """

        Parameters
        ----------
        qb : QBDecomposer
            qb(A, ell, tol, rng) returns a QB factorization of A with
            target rank "ell", target tolerance "tol", using the numpy
            Generator object np.random.default_rng(rng).

        Notes
        -----
        Typical implementation structures include ...
            RS1(RowSketcher)
            --> RF1(RangeFinder)
            --> QB1(QBFactorizer)
            --> SVD1(SVDecomposer)
        or
            RS1(RowSketcher)
            --> RF1(RangeFinder)
            --> QB2(QBFactorizer)
            --> SVD1(SVDecomposer)
        or
            RS1(RowSketcher)
            --> QB3(QBFactorizer)
            --> SVD1(SVDecomposer)
        """
        self.qb = qb

    def __call__(self, A, k, tol, over, rng):
        rng = np.random.default_rng(rng)
        Q, B = self.qb(A, k + over, tol, rng)
        U, s, Vh = la.svd(B, full_matrices=False)
        if over > 0:
            cutoff = min(k, s.size)
            U = U[:, :cutoff]
            s = s[:cutoff]
            Vh = Vh[:cutoff, :]
        drop = s < 10*np.finfo(float).eps
        if np.any(drop):
            U = U[:, ~drop]
            s = s[~drop]
            Vh = Vh[~drop, :]
        U = Q @ U
        return U, s, Vh
