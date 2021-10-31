import numpy as np
import scipy.linalg as la
import rlapy.comps.sketchers.oblivious as oblivious
from rlapy.comps.rangefinders import RangeFinder, RF1
from rlapy.comps.sketchers.aware import RowSketcher, RS1
from rlapy.comps.qb import QBDecomposer, QB1
import rlapy.utils.linalg_wrappers as ulaw


###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def evd1(num_passes, A, k, tol, over, rng):
    """
    Return the eigen decomposition matrices (V, lamb),
    based on a rank-k QB factorization of A.
    Use a Gaussian sketching matrix and pass over A a total of
    num_passes times.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A. We require num_passes >= 2.

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate. A must be an n × n Hermitian matrix.

    k : int
        Target rank for the approximation of A: 0 < k < min(A.shape).
        This parameter includes any oversampling. For example, if you
        want to be near the optimal (Eckhart-Young) error for a rank 20
        approximation of A, then you might want to set k=25.
        
    tol : float
        Target accuracy for the oversampled approximation of A: 0 < tol < np.inf.
        This parameter inherits from the QBDecomposer or RangeFinder class.
        
    over : int
        Auxiliary parameter for the QBDecomposer or RangeFinder.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    lamb : ndarray
        Has shape (k,), the vector of estimated eigenvalues of A

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.

    References
    ----------
    This algorithm computes V and then sets B = Q.T @ A. Conceptually, we
    compute Q by adapting Algorithm 5.3 from

        Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
        "Finding structure with randomness: Probabilistic algorithms for
        constructing approximate matrix decompositions."
        SIAM review 53.2 (2011): 217-288.
        (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    EVDecomposer: adapted from the QB-based [HMT11, Algorithm 5.3]
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), num_passes - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB1(rf_)
    evd_ = EVD1(qb_)
    V, lamb = evd_(A, k, tol, over, rng)
    return V, lamb


def evd2(num_passes, A, k, tol, over, rng):
    """
    Return the eigen decomposition matrices (V, lamb),
    based on a rank-k QB factorization of A.
    Use a Gaussian sketching matrix and pass over A a total of
    num_passes times.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A. We require num_passes >= 2.
        
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate. A must be an n × n Hermitian PSD matrix.
        But we did not check PSD

    k : int
        Target rank for the approximation of A: 0 < k < min(A.shape).
        This parameter includes any oversampling. For example, if you
        want to be near the optimal (Eckhart-Young) error for a rank 20
        approximation of A, then you might want to set k=25.
        
    tol = np.nan : NaN by default
        Target accuracy for the oversampled approximation of A: 0 < tol < np.inf.
        This parameter inherits from the QBDecomposer or RangeFinder class.
        
    over : int
        Auxiliary parameter for the QBDecomposer or RangeFinder.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    lamb : ndarray
        Has shape (k,). lamb contains the estimated eigenvalues of A.

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.

    References
    ----------
    This algorithm computes V and then sets B = Q.T @ A. Conceptually, we
    compute Q by adapting Algorithm 5.3 from

        Joel A Tropp, Alp Yurtsever, Madeleine Udell, and Volkan Cevher.
        "Fixed-rank approximation of a positive-semidefinite matrix from streaming data."
        Advances in neural information processing systems, 2017.
        (available at `arXiv <https://arxiv.org/abs/1706.05736>`_).
    EVDecomposer: psd matrices only, [TYUC17a, Algorithm 3]
    """
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), num_passes - 2, ulaw.orth, 1)
    evd_ = EVD2(rso_)
    V, lamb = evd_(A, k, tol, over, rng)
    return V, lamb

###############################################################################
#       Object-oriented interfaces
###############################################################################


class EVDecomposer:

    TOL_CONTROL = 'none'

    def __call__(self, A, k, tol, over, rng):
        """
        Return a matrix V with orthonormal columns and a real vector lamb
        where A is approximated by A_hat = V @ diag(lamb) @ V.T.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate. A must be an n × n Hermitian matrix. 

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            Typically, k << min(A.shape). Conformant implementations ensure
            Q has at most k columns. For certain implementations it's
            reasonable to choose k as large as k = min(A.shape), in which
            case the implementation returns only once a specified error
            tolerance has been met.

        tol : float
            Target accuracy for the oversampled approximation of A: 0 < tol < np.inf.
            This parameter inherits from the QBDecomposer or RangeFinder class.
            
        over : int
            Auxiliary parameter for the QBDecomposer or RangeFinder.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        V : ndarray
            Has shape (A.shape[0], k). Columns are orthonormal.

        lamb : ndarray
            Has shape (k,). lamb contains the estimated eigenvalues of A.
        """
        raise NotImplementedError()


class EVD1(EVDecomposer):

    TOL_CONTROL = 'unknown'  # depends on implementation of QB

    def __init__(self, qb: QBDecomposer):
        self.qb = qb

    def __call__(self, A, k, tol, over, rng):
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
        num_passes : int
            Total number of passes over A. We require num_passes >= 2.

        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate. A must be an n × n Hermitian matrix.

        k : int
            Target rank for the approximation of A: 0 < k < min(A.shape).
            This parameter includes any oversampling. For example, if you
            want to be near the optimal (Eckhart-Young) error for a rank 20
            approximation of A, then you might want to set k=25.
            
        tol : float
            Target accuracy for the oversampled approximation of A: 0 < tol < np.inf.
            This parameter inherits from the QBDecomposer or RangeFinder class.
            
        over : int
            Auxiliary parameter for the QBDecomposer or RangeFinder.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        V : ndarray
            Has shape (A.shape[0], k). Columns are orthonormal.

        lamb : ndarray
            Has shape (k,). lamb contains the estimated eigenvalues of A.

        """
        assert k > 0
        assert k <= min(A.shape)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)
        Q, B = self.qb(A, k + over, tol, rng)
        # B=Q^*A is necessary
        C = B @ Q
        # d = number of columns in Q, d ≤ k + s
        d = Q.shape[1]
        if d > k + over:
            msg = """
            This implementation the dimension of Q matrix <= k + s.
            """ 
            raise RuntimeError(msg)

        lamb, U = la.eigh(C)
        alamb = np.abs(lamb)
        r = min(k, d, np.count_nonzero(alamb > np.finfo(float).eps))
        I = np.argsort(-1*np.abs(alamb))[:r]
        # indices of r largest components of |λ|
        U = U[:, I]
        lamb = lamb[I] 
        V = Q @ U
        return V, lamb


class EVD2(EVDecomposer):

    TOL_CONTROL = 'unknown'  # depends on implementation of rangefinder

    def __init__(self, rs: RowSketcher):
        """
        Parameters
        ----------
        rs : RowSketcher
        """
        self.rowsketcher = rs

    def __call__(self, A, k, tol, over, rng):
        """
        Rely on a RowSketcher to obtain the matrix S for the sketching matrix
        S. Once we have S, we construct Y = A @ S and return Cholesky decomposition of 
        S^T @ Y. This function is agnostic to the implementation of the
        RowSketcher: it might has different ways of row sketching. We make no
        assumptions on the RowSketcher's termination criteria beyond those
        listed below.

        Return the eigen decomposition matrices (V, lamb),
        based on a row sketched version of A.
        Use a Gaussian sketching matrix and pass over A a total of
        num_passes times.

        Parameters
        ----------
        num_passes : int
            Total number of passes over A. We require num_passes >= 2.
            
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate. A must be an n × n Hermitian PSD matrix.
            But we did not check PSD

        k : int
            Target rank for the approximation of A: 0 < k < min(A.shape).
            This parameter includes any oversampling. For example, if you
            want to be near the optimal (Eckhart-Young) error for a rank 20
            approximation of A, then you might want to set k=25.
            
        tol = np.nan : NaN by default
            Target accuracy for the oversampled approximation of A: 0 < tol < np.inf.
            This parameter inherits from the QBDecomposer or RangeFinder class.
            
        over : int
            Auxiliary parameter for the QBDecomposer or RangeFinder.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        V : ndarray
            Has shape (A.shape[0], k). Columns are orthonormal.

        lamb : ndarray
            Has shape (k,). lamb contains the estimated eigenvalues of A.

        """
        assert k > 0
        assert k <= min(A.shape)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)
        S = self.rowsketcher(A, k + over, rng)
        n = A.shape[0]
        Y = A @ S
        epsilon_mach = np.finfo(float).eps # a temporary regularization parameter
        nu = np.sqrt(n)*epsilon_mach*la.norm(Y)
        # a temporary regularization parameter
        Y = Y + nu*S
        R = la.cholesky(S.T @ Y, lower=True)
        # R is upper-triangular and R^T @ R = S^T @ Y = S^T @ (A + nu*I)S
        # B = Y @ la.inv(R.T)
        B = (la.solve_triangular(R, Y.T, lower=True)).T
        # B has n rows and k + s columns
        V, sigma, Wh = la.svd(B)
        
        comp_list = [k]
        for i in range(min(k, n)-1):
            if sigma[(i+1)]**2 <= nu:
                comp_list.append(i)
        r = min(comp_list)
        # drop components that relied on regularization
        lamb = (sigma**2)[:r]-nu
        V = V[:, :r]
        return V, lamb
