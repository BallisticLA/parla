"""
Routines for the orthogonal rangefinder problem:

    Given an m-by-n input matrix A, find a matrix Q with k << min(m, n)
    orthogonal columns where the operator norm || A - Q Q' A ||_2 is small.

    This problem can be considered where k is given or when a tolerance "tol"
    is given and we require || A - Q Q' A ||_2 <= tol.
"""
import warnings
import numpy as np
import scipy.linalg as la
import rlapy.utils.linalg_wrappers as ulaw
import rlapy.comps.sketchers.oblivious as oblivious
from rlapy.comps.sketchers.aware import RowSketcher, RS1

###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def power_rangefinder(A, k, num_pass, rng):
    """
    Return a matrix Q with k orthonormal columns, where Range(Q) is
    an approximation for the span of A's top k left singular vectors.

    This implementation uses a Gaussian sketching matrix with k columns
    and requires a total of num_pass over A. We use QR to orthogonalize
    the output of every matrix multiply that involves A or A'.

    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix whose range is to be approximated.

    k : int
        Q.shape[1] == k. We require 0 < k <= min(A.shape).

    num_pass : int
        Total number of passes over A.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages any and all
        randomness in this function call.

    Returns
    -------
    Q : ndarray
        Q.shape = (A.shape[0], k) has orthonormal columns.

    Notes
    -----
    The implementation is built up as
         RS1(RowSketcher) --> RF1(RangeFinder)

    References
    ----------
    Conceptually, we compute Q by using [HMT:2011, Algorithm 4.3] and
    [HMT:2011, Algorithm 4.4]. However, is a difference in how we perform
    subspace iteration. Our subspace iteration still uses QR to stabilize
    computations at each step, but computations are structured along the
    lines of [ZM:2020, Algorithm 3.3] to allow for any number of passes over A.
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(sketch_op_gen=oblivious.SkOpGA(),
               num_pass=num_pass - 1,
               stabilizer=ulaw.orth,
               passes_per_stab=1)
    rf_ = RF1(rso_)
    Q = rf_(A, k, 0.0, rng)
    return Q

###############################################################################
#       Object oriented interfaces
###############################################################################


class RangeFinder:

    def __call__(self, A, k, tol, rng):
        """
        Return a matrix Q with orthonormal columns, where range(Q) is
        "reasonably" well aligned with A's leading left singular vectors.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            Typically, k << min(A.shape). Conformant implementations ensure
            Q has at most k columns. For certain implementations it's
            reasonable to choose k as large as k = min(A.shape), in which
            case the implementation returns only once a specified error
            tolerance has been met.

        tol : float
            Target for the error  ||A - Q Q' A||: 0 <= tol < np.inf. Only
            certain implementations are able to control approximation error.
            Those implementations may return a matrix Q with fewer than k
            columns if ||A - Q Q' A|| <= tol. Assuming k < rank(A) and that the
            implementation can compute ||A - Q Q' A|| accurately, setting
            tol=0 means the implementation will return Q with exactly k columns.

            Implementations that cannot control error should raise a warning
            if tol is not NaN.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        Here is the simplest possible implementation:

            rng = np.random.default_rng(rng)
            S = rng.standard_normal((A.shape[1], k))
            Y = A @ S
            Q = la.qr(Y, mode='economic')[0]

        """
        raise NotImplementedError()


class RF1(RangeFinder):

    def __init__(self, rso: RowSketcher):
        self.rso = rso

    def __call__(self, A, k, tol, rng):
        """
        Return a matrix Q with k orthonormal columns, where Range(Q) is
        an approximation for the span of A's top k left singular vectors.

        This function works by
            (1) Using a RowSketcher object to generate a matrix S of
                shape (A.shape[1], k),
            (2) Computing Y = A @ S
            (3) Returning the factor Q from a QR factorization of Y.

        The most common implementation of RowSketchingOpertaor is "RS1",
        which uses a randomized power method to amplify the sampling power
        of an oblivious sketching matrix.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.

        k : int
            Q.shape[1] == k.

        tol : float
            Refer to the RangeFinder interface for the general meaning
            of this parameter. This implementation cannot control solution
            accuracy and so will raise a warning if tol is not NaN.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Q.shape = (A.shape[0], k) and has orthonormal columns.

        References
        ----------
        This RangeFinder implementation originally relied on a subspace
        iteration routine (along the lines of [ZM:2020, Algorithm 3.3]).
        However, subspace iteration is merely a means to an end, which is
        to produce a matrix S where range(S) is reasonably well aligned with
        A's leading right singular vectors.
        """
        assert k > 0
        assert k <= min(A.shape)
        if not np.isnan(tol):
            msg = """
            This RangeFinder implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        rng = np.random.default_rng(rng)
        S = self.rso(A, k, rng)
        Y = A @ S
        Q = la.qr(Y, mode='economic')[0]
        return Q
