import numpy as np
import scipy.linalg as la
from rlapy.comps.qb import QBFactorizer


class SVDecomposer:

    def exec(self, A, k, tol, over, rng):
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
            ensures ||A - U @ diag(s) @ Vh|| <= tol on exit. However,
            setting k=min(A.shape) may trivially return the SVD of
            A in some implementations.

        tol : float
            The target error used by the randomized part of the algorithm.
            When over = 0, this parameter controls ||A - U @ diag(s) @ Vh||.
            The precise meaning of "tol" when over > 0 is implementation
            dependent.

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


class SVD1(SVDecomposer):
    """
    Use the Phase 1 / Phase 2 approach from HMT2011, where Phase 1 is
    implemented by an arbitrary QB factorization method.
    """

    def __init__(self, qb: QBFactorizer):
        """

        Parameters
        ----------
        qb : QBFactorizer
            qb.exec(A, ell, tol, rng) returns a QB factorization of A with
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

    def exec(self, A, k, tol, over, rng):
        rng = np.random.default_rng(rng)
        Q, B = self.qb.exec(A, k + over, tol, rng)
        U, s, Vh = la.svd(B, full_matrices=False)
        if over > 0:
            cutoff = min(k, s.size)
            U = U[:, :cutoff]
            s = s[:cutoff]
            Vh = Vh[:cutoff, :]
        U = Q @ U
        return U, s, Vh
