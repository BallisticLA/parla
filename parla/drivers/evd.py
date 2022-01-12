import numpy as np
import scipy.linalg as la
import parla.comps.sketchers.oblivious as oblivious
import warnings
from parla.comps.rangefinders import RF1
from parla.comps.sketchers.aware import RowSketcher, RS1
from parla.comps.qb import QBDecomposer, QB2
import parla.utils.linalg_wrappers as ulaw


###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def evd1(A, k, tol, over, inner_num_pass, block_size, rng):
    """
    Return ndarrays (V, lamb) that define a symmetric matrix "A_approx" through
    its eigen-decomposition:

        A_approx = (V * lamb) @ V.T.

    The function assumes A is symmetric.
    The columns of V are approximations of the dominant eigenvectors of A.
    The entries of lamb are the corresponding approximate eigenvalues.

    Parameters
    ----------

    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate. Must be n × n and symmetric matrix.

    k : int
        Maximum rank for the approximation of A: 0 < k <= n.
        If you set tol=0, then the returned approximation will have rank
        min(k, rank(A)).

    tol : float
        Target relative accuracy for the oversampled approximation of A.
        If you set k = n and over=0 then the returned approximation should
        satisfy ||A - A_approx||_F / || A ||_F <= tol.

    over : int
        Perform internal calculations with a sketch of rank (k + over).
        This is usually a small constant, e.g., 5 to 25. In some situations
        it's useful to set over = k. It's valid to set over = 0.

    inner_num_pass : int
        Total number of passes the algorithm makes over A in a single iteration.
        This must be at least 2 and is usually not more than 10.
        Increasing this parameter is one way to obtain better
        approximations, especially at lower ranks.

    block_size : int
        The approximation is built incrementally, updating the rank by
        block_size at each iteration (with safeguards, so we never exceed
        the rank of A). If block_size = k + over, then the algorithm accesses
        A "inner_num_pass" times.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (n, d) where d <= min(k, rank(A)). Columns are orthonormal.

    lamb : ndarray
        Has shape (d,) where d <= min(k, rank(A)).
        The vector of estimated eigenvalues of A.

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.
    Subspace iteration is initialized with a Gaussian sketching operator.

    References
    ----------
    This function adapts Algorithm 5.3 from

        Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
        "Finding structure with randomness: Probabilistic algorithms for
        constructing approximate matrix decompositions."
        SIAM review 53.2 (2011): 217-288.
        (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    assert inner_num_pass >= 2
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), inner_num_pass - 2, ulaw.orth, 1)
    rf_ = RF1(rso_)
    qb_ = QB2(rf_, block_size, overwrite_a=False)
    evd_ = EVD1(qb_)
    V, lamb = evd_(A, k, tol / 2, over, rng)
    return V, lamb


def evd2(A, k, over, num_passes, rng):
    """
    Return ndarrays (V, lamb) that define a symmetric positive semidefinite matrix
    "A_approx" through its eigen-decomposition:

        A_approx = (V * lamb) @ V.T.

    The function assumes A is symmetric positive semidefinite.
    The columns of V are approximations of the dominant eigenvectors of A.
    The entries of lamb are the corresponding approximate eigenvalues.

    The approximation is "fixed rank." The array V has min(k, rank(A)) columns,
    and there is no direct control over the approximation error ||A_approx - A||.
    Increasing "num_passes" and "over" should result in better approximations.

    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate. A must be an n × n positive semidefinite matrix.

    k : int
        Target rank for the approximation of A: 0 < k < n.

    over : int
        Perform internal calculations with a sketch of rank (k + over).
        This is usually a small constant, e.g., 5 to 25. In some situations
        it's useful to set over = k.

    num_passes : int
        Total number of passes the algorithm is allowed over A.
        We require num_passes >= 1, and usually we have num_passes <= 10.
        Increasing this parameter is one way to obtain better
        approximations, especially at lower ranks.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    V : ndarray
        Has shape (n, min(k, rank(A)). Columns are orthonormal.

    lamb : ndarray
        Has shape (min(k, rank(A)),), the vector of estimated eigenvalues of A.

    Notes
    -----
    We perform (num_passes - 2) steps of subspace iteration, and
    stabilize subspace iteration by a QR factorization at every step.

    References
    ----------
    This function adapts Algorithm 3 from

        Joel A Tropp, Alp Yurtsever, Madeleine Udell, and Volkan Cevher.
        "Fixed-rank approximation of a positive-semidefinite matrix from streaming data."
        Advances in neural information processing systems, 2017.
        (available at `arXiv <https://arxiv.org/abs/1706.05736>`_).
    """
    assert num_passes >= 1
    rng = np.random.default_rng(rng)
    rso_ = RS1(oblivious.SkOpGA(), num_passes - 1, ulaw.orth, 1)
    evd_ = EVD2(rso_)
    V, lamb = evd_(A, k, np.NaN, over, rng)
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
            Data matrix to approximate. A must be an n × n symmetric matrix.

        k : int
            Target rank for the approximation of A: 0 < k < n.

        tol : float
            Relative target accuracy for the approximation of A.

        over : int
            Perform internal calculations with a sketch of rank (k + over).
            This is usually a small constant, e.g., 5 to 25. In some situations
            it's useful to set over = k.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        V : ndarray
            Has shape (n, min(k, rank(A)). Columns are orthonormal.

        lamb : ndarray
            Has shape (min(k, rank(A)),), the vector of estimated eigenvalues of A.
        """
        raise NotImplementedError()


class EVD1(EVDecomposer):

    TOL_CONTROL = 'unknown'  # depends on implementation of QB

    def __init__(self, qb: QBDecomposer):
        self.qb = qb

    def __call__(self, A, k, tol, over, rng):
        """
        Return ndarrays (V, lamb) that define a symmetric matrix "A_approx" through
        its eigen-decomposition:

            A_approx = (V * lamb) @ V.T.

        The function assumes A is symmetric.
        The columns of V are approximations of the dominant eigenvectors of A.
        The entries of lamb are the corresponding approximate eigenvalues.

        This function can accommodate target accuracies if (and only if) the
        underlying QBDecomposer can accommodate target accuracies.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate. A must be an n × n symmetric matrix.

        k : int
            Target rank for the approximation of A: 0 < k <= n.
            
        tol : float
            Relative target accuracy for the oversampled approximation of A.
            If k = n and over = 0, then we should have
            ||A - A_approx||_F / || A ||_F <= tol.
            
        over : int
            Perform internal calculations using a sketch of rank k + over.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        V : ndarray
            Has shape (n, min(k, rank(A)). Columns are orthonormal.

        lamb : ndarray
            Has shape (min(k, rank(A)),). lamb contains the estimated eigenvalues of A.

        Notes
        -----
        This function adapts Algorithm 5.3 from

            Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
            "Finding structure with randomness: Probabilistic algorithms for
            constructing approximate matrix decompositions."
            SIAM review 53.2 (2011): 217-288.
            (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
        """
        assert k > 0
        assert k <= min(A.shape)
        if not np.isnan(tol):
            assert tol >= 0
            assert tol < np.inf
        rng = np.random.default_rng(rng)
        Q, B = self.qb(A, k + over, tol / 2, rng)
        # B=Q^*A is necessary
        C = B @ Q
        lamb, U = la.eigh(C)
        alamb = np.abs(lamb)
        # d = number of columns in Q, d ≤ k + s
        d = Q.shape[1]
        r = min(k, d, np.count_nonzero(alamb > 10*np.finfo(float).eps))
        I = np.argsort(-1*np.abs(alamb))[:r]
        # indices of r largest components of |λ|
        U = U[:, I]
        lamb = lamb[I] 
        V = Q @ U
        return V, lamb


class EVD2(EVDecomposer):

    TOL_CONTROL = 'none'

    def __init__(self, sk_op: RowSketcher):
        self.sk_op = sk_op

    def __call__(self, A, k, tol, over, rng):
        """
        Return ndarrays (V, lamb) that define a symmetric positive semidefinite matrix
        "A_approx" through its eigen-decomposition:

            A_approx = (V * lamb) @ V.T.

        The function assumes A is symmetric positive semidefinite.
        The columns of V are approximations of the dominant eigenvectors of A.
        The entries of lamb are the corresponding approximate eigenvalues.

        The approximation is produced by truncating a rank (k + over) Nystrom
        approximation

            (A S) (S' A S)^{\\dagger} (A S)'

        to rank k. The matrix S is obtained by S = self.sk_op(A, k + over, rng).

        Parameters
        ----------
            
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix to approximate. A must be an n × n symmetric PSD matrix.
            We do not check that A is PSD (doing so would be too expensive).

        k : int
            Target rank for the approximation of A: 0 < k < n.
            
        tol : np.NaN
            This class cannot control accuracy, and ignores this parameter.
            
        over : int
            Define the initial Nystrom approximation with a sketch of rank (k + over).

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages randomness
            in this function call.

        Returns
        -------
        V : ndarray
            Has shape (n, k). Columns are orthonormal.

        lamb : ndarray
            Has shape (k,). lamb contains the estimated eigenvalues of A.

        Notes
        -----
        This function adapts Algorithm 3 from

            Joel A Tropp, Alp Yurtsever, Madeleine Udell, and Volkan Cevher.
            "Fixed-rank approximation of a positive-semidefinite matrix from streaming data."
            Advances in neural information processing systems, 2017.
            (available at `arXiv <https://arxiv.org/abs/1706.05736>`_).
        """
        assert k > 0
        n = A.shape[0]
        assert k < n
        if not np.isnan(tol):
            msg = """
            This EVDecomposer implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        rng = np.random.default_rng(rng)
        S = self.sk_op(A, k + over, rng)
        Y = A @ S
        epsilon_mach = np.finfo(float).eps
        nu = np.sqrt(n) * epsilon_mach * la.norm(Y)
        # a temporary regularization parameter
        Y = Y + nu*S
        R = la.cholesky(S.T @ Y, lower=True)
        # R is upper-triangular and R^T @ R = S^T @ Y = S^T @ (A + nu*I)S
        B = (la.solve_triangular(R, Y.T, lower=True)).T
        # B has n rows and k + s columns
        V, sigma, Wh = la.svd(B)
        
        comp_list = [k]
        for i in range(k-1):
            if sigma[(i+1)]**2 <= nu:
                comp_list.append(i)
        r = min(comp_list)
        # drop components that relied on regularization
        lamb = (sigma**2)[:r]-nu
        V = V[:, :r]
        return V, lamb
