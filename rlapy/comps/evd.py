#####To-be removed
rlapy_path='/media/hrluo/ALL/rlapy'
# insert at 1, 0 is the script path (or '' in REPL)
import sys
sys.path.insert(0, rlapy_path)

import warnings

import numpy as np
import scipy.linalg as la
from rlapy.utils.sketching import gaussian_operator
from rlapy.comps.rangefinders import RangeFinder, RF1
from rlapy.comps.sketchers import RowSketcher, RS1
from rlapy.comps.qb import QBDecomposer, QB1
#from rlapy.comps.svd import SVDecomposer, SVD1
import rlapy.utils.linalg_wrappers as ulaw

###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def evd1(num_passes, A, k, epsilon, s, rng):
    """
    Return the eigen decomposition matrices (V, lambda_matrix),
    based on a rank-k QB factorization of A.
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
        
    epsilon : float
        Target accuracy for the oversampled approximation of A: 0 < epsilon < np.inf.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    lambda_matrix : ndarray
        Has shape (k,k). lambda_matrix is a diagonal matrix storing the eigenvalues of 
        the matrix A

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
    rso_ = RS1(gaussian_operator, num_passes - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB1(rf_)#We need QB1 not QB2, since B=Q^*A is necessary
    Q, B = qb_(A, k+s, np.NaN, rng)
    #C = B @ Q# is the expression in design document, but is this correct?
    C = Q @ B
    
    d = Q.shape[1]
    if d > k+s:
        msg = """
        This implementation the dimension of Q matrix <= k + s.
        """ 
        raise RuntimeError(msg)
    U, lambda_matrix, Vh = la.svd(C)

    r = min(k,d)
    I = np.argsort(-1*np.abs(lambda_matrix))[range(r)]
    U = U[:,I]
    lambda_matrix = lambda_matrix[I]
    # V = Q @ U# is the expression in design document, but is this correct?
    V = Q.T @ U
    return V, lambda_matrix

def evd2(num_passes, A, k, epsilon, s, rng):
    """
    Return the eigen decomposition matrices (V, lambda_matrix),
    based on a rank-k QB factorization of A.
    Use a Gaussian sketching matrix and pass over A a total of
    num_passes times.

    Parameters
    ----------
    num_passes : int
        Total number of passes over A. We require num_passes >= 2.

    np.nan : NaN
        A space-holder for unified interface.

    k : int
        Target rank for the approximation of A: 0 < k < min(A.shape).
        This parameter includes any oversampling. For example, if you
        want to be near the optimal (Eckhart-Young) error for a rank 20
        approximation of A, then you might want to set k=25.
        
    epsilon : float
        Target accuracy for the oversampled approximation of A: 0 < epsilon < np.inf.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (A.shape[0], k). Columns are orthonormal.

    lambda_matrix : ndarray
        Has shape (k,k). lambda_matrix is a diagonal matrix storing the eigenvalues of 
        the matrix A

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
    rso_ = RS1(gaussian_operator, num_passes - 2, ulaw.orth, 1)
    S = rso_(A, k + s,rng)
    n = A.shape[0]
    Y = A @ S
    epsilon_mach = epsilon # a temporary regularization parameter
    nu = np.sqrt(n)*epsilon_mach*la.norm(Y)
    
    Y = Y + nu*S
    R = la.cholesky(S.T @ Y, lower=True)
    # R is upper-triangular and R^T @ R = S^T @ Y = S^T @ (A + nu*I)S
    B = Y @ la.inv(R.T)
    V, Sigma_matrix, Wh = la.svd(B)
    W = Wh.T
    comp_list = [k]
    for i in range(min(k,n)):
        if Sigma_matrix[(i+1)]**2<=nu:
            comp_list.append(i)
    r = min(comp_list) # drop components that relied on regularization
    lambda_matrix = (Sigma_matrix**2)[range(r)]-nu
    V = V[:,range(r)]
    return V, lambda_matrix

###############################################################################
#       Object-oriented interfaces
###############################################################################

class EVDecomposer:

    TOL_CONTROL = 'none'

    def __call__(self, A, k, epsilon, s, rng):
        """
        Return a matrix V with orthonormal columns and a lambda matrix (in vector form) lambda_matrix where
        the diagonal are the eigen values of 

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

        epsilon : float
            Target accuracy for the oversampled approximation of A: 0 < epsilon < np.inf.
            
        s  : int
            Auxiliary parameter for the QBDecomposer or RangeFinder.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        V : ndarray
            Has shape (A.shape[0], k). Columns are orthonormal.

        lambda_matrix : ndarray
            Has shape (k,k). lambda_matrix is a diagonal matrix storing the eigenvalues of 
            the matrix A
        """
        raise NotImplementedError()


class EVD1(EVDecomposer):

    TOL_CONTROL = 'unknown'  # depends on implementation of rangefinder

    def __init__(self, rf: RangeFinder):
        """
        Parameters
        ----------
        rf : RangeFinder
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
        if not np.isnan(epsilon):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)
        Q = self.rangefinder(A, k, tol, rng)
        B = Q.T @ A
        #C = B @ Q# is the expression in design document, but is this correct?
        C = Q @ B
        
        d = Q.shape[1]
        if d > k+s:
            msg = """
            This implementation the dimension of Q matrix <= k + s.
            """ 
            raise RuntimeError(msg)
        U, lambda_matrix, Vh = la.svd(C)

        r = min(k,d)
        I = np.argsort(-1*np.abs(lambda_matrix))[range(r)]
        U = U[:,I]
        lambda_matrix = lambda_matrix[I]
        # V = Q @ U# is the expression in design document, but is this correct?
        V = Q.T @ U
        return V, lambda_matrix

class EVD2(EVDecomposer):

    TOL_CONTROL = 'unknown'  # depends on implementation of rangefinder

    def __init__(self, rs: RowSketcher):
        """
        Parameters
        ----------
        rs : RowSketcher
        """
        self.rowsketcher

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
        if not np.isnan(epsilon):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)
        S = self.rowsketcher(A, k + s,rng)
        n = A.shape[0]
        Y = A @ S
        epsilon_mach = epsilon # a temporary regularization parameter
        nu = np.sqrt(n)*epsilon_mach*la.norm(Y)
        
        Y = Y + nu*S
        R = la.cholesky(S.T @ Y, lower=True)
        # R is upper-triangular and R^T @ R = S^T @ Y = S^T @ (A + nu*I)S
        B = Y @ la.inv(R.T)
        V, Sigma_matrix, Wh = la.svd(B)
        W = Wh.T
        comp_list = [k]
        for i in range(min(k,n)):
            if Sigma_matrix[(i+1)]**2<=nu:
                comp_list.append(i)
        r = min(comp_list) # drop components that relied on regularization
        lambda_matrix = (Sigma_matrix**2)[range(r)]-nu
        V = V[:,range(r)]
        return V, lambda_matrix


