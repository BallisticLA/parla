import warnings
import numpy as np
import scipy.linalg as la
import rlapy.utils.linalg_wrappers as ulaw
from rlapy.comps.qb import QBFactorizer
from rlapy.comps.sketchers import RowSketcher


class LUDecomposer:

    def __call__(self, A, k, tol, over, rng):
        """
        Let A be m-by-n. For some integer ell <= k, return
            Pl: an m-by-m permutation matrix,
            L: a lower-triangular matrix of shape (m, ell),
            U: an upper-triangular matrix of shape (ell, n),
            Pu: an n-by-n permutation matrix,
        so that
            A \approx Pl @ L @ U @ Pu.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate.

        k : int
            The returned LU decomposition will have rank at most k:
            0 < k <= min(A.shape). Setting k=min(A.shape) and over=0
            ensures ||A - Pl @ L @ U @ Pu|| <= tol on exit. However,
            setting k=min(A.shape) may trivially return a full LU decomposition
            of A in some implementations.

        tol : float
            The target error used by the randomized part of the algorithm.
            When over = 0, this parameter is a desired bound on
            ||A - Pl @ L @ U @ Pu||. The precise meaning of "tol" when over
            > 0 is implementation dependent.

            Some LU implementations have no direct control over approximation
            error. Those implementations should raise a warning if tol > 0.
            The rationale for this behavior is that setting tol > 0 indicates
            an intention on the user's part that approximation error play a
            role in the stopping criteria.

        over : int
            The randomized part of the algorithm uses k+over as the target rank;
            we require over >= 0 and k+over <= min(A.shape).
            In a conformant implementation, that part of the algorithm will
            never return a factorization of rank greater than k+over.

            Setting over > 0 will likely result in truncating the the LU
            factorization obtained in the randomized part of the algorithm.
            You can avoid undesired truncation by setting over=0 and
            increasing the value of k. E.g., a function call with over=5 and
            k=20 can avoid truncation by setting k=25 and over=0.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.
        """
        raise NotImplementedError()


class LU1(LUDecomposer):

    def __init__(self, qb: QBFactorizer):
        self.qb = qb

    def __call__(self, A, k, tol, over, rng):
        """
        TODO: describe algorithm and document parameters. Comment
         that there's a very long note about how the parameter "over"
         is handled.

        Notes
        -----
        This implementation draws from ZM2020's Algorithm 3.1.
        When over > 0, we have to truncate an intermediate factorization.
        Our truncation approach is different from that of ZM2020 Algorithm 3.1.

        Specifically, ZM2020 implements truncation by taking the leading
        k columns from Q after a QB decomposition. (ZM2020 doesn't use
        QB as an explicit subroutine, but their algorithm essentially
        starts with a QB step based on a rangefinder, which in turn is
        based on subspace iteration.) Such a truncation method seems strange
        to me (Riley) because the approximation A \approx QB is invariant
        under permutations that act simultaneously on the columns of Q and
        the rows of B. So for the truncation approach in ZM2020 to have
        theoretical justifications, I think we would need to make detailed
        assumptions on how Q is computed.

        Our truncation approach is based on randomized LU algorithms in
        SSAA2018, which truncate the output of the first LU factorization.
        In the context of SSAA2018, this truncation strategy required the
        first LU factorization to be rank revealing. We don't use RRLU.
        Moreover, we have not established any theoretical justification for
        this truncation strategy in the larger context of this algorithm.
        """
        assert k > 0
        assert k <= min(A.shape)
        assert tol < np.inf
        rng = np.random.default_rng(rng)
        Q, B = self.qb(A, k + over, tol, rng)
        # ^ We have A \approx Q B
        #   Note: the orthogonality of Q isn't used
        L1, U1, P1 = ulaw.lup(B)  # B = L1 @ U1 @ P1
        cutoff = min(k, U1.shape[0])
        U1 = U1[:cutoff, :]
        L1 = L1[:, :cutoff]
        Y = Q @ L1
        P2, L2, U2 = la.lu(Y)
        # When over = 0, we have ...
        #   Q B = Q (L1 U1 P1)
        #       = (Q L1) (U1 P1)
        #       = (P2 L2 U2) (U1 P1)
        #       = P2 L2 (U2 U1) (P1)
        U = U2 @ U1
        return P2, L2, U, P1


class LU2(LUDecomposer):

    def __init__(self, sk_op: RowSketcher, lstsq: la.lstsq):
        self.sk_op = sk_op
        self.lstsq = lstsq

    def __call__(self, A, k, tol, over, rng):
        """
        This is inspired by [SSAA:2018, Alg 4.1].
        """
        #TODO: describe algorithm and document parameters.
        #   Explain that this algorithm has no control over tol.
        assert k > 0
        assert k <= min(A.shape)
        assert tol < np.inf
        if tol > 0:
            msg = """
            This LU implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        rng = np.random.default_rng(rng)
        S = self.sk_op(A, k + over, rng)
        Y = A @ S
        # get a useful basis for range(Y); Py @ Ly
        Py, Ly, Uy = la.lu(Y)
        if over > 0:
            # truncate, as though we used RRLU.
            Ly = Ly[:, :k]
        # next, compute Z = pinv(Py @ Ly) @ A
        pyt = np.where(Py.T)[1]
        PyTA = A[pyt, :]
        B = self.lstsq(Ly, PyTA)
        Lb, Ub, Pb = ulaw.lup(B)
        L = Ly @ Lb
        U = Ub
        return Py, L, U, Pb
