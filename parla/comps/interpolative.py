import numpy as np
import scipy.linalg as la
from scipy import linalg as la
from parla.comps.sketchers.aware import RowSketcher
import parla.comps.sketchers.oblivious as osk
import parla.comps.sketchers.aware as ask
import parla.utils.linalg_wrappers as ulaw
import parla.utils.misc as misc


def qrcp_osid(Y, k, axis):
    """
    Use QRCP to deterministically compute a rank-k one-sided ID of Y;
    return the skeleton indices and interpolative coefficient matrix.

    Parameters
    ----------
    Y : ndarray
        This is typically a sketch of a larger matrix.

    k : int
        This is typically relatively close to (and may equal) min(Y.shape).

    axis : int
        0 for a row ID, 1 for a column ID.

    Returns
    -------

    mat : ndarray
        Interpolative coefficient matrix

    idxs : ndarray
        Skeleton indices

    Background
    ----------
    A RowID consists of a matrix "Z" and a length-k index vector "Is" so
    that Y \\approx Z @ Y[Is,:]. The rows of Z must contain a
    possibly-permuted k-by-k identity matrix, such that

        Y[Is, :] = (Z @ Y[Is, :])[Is,:].

    A ColumnID consists of a matrix "X" and a length-k index vector "Js" so
    that Y \\approx Y[:,Js] @ X. The columns of X must contain a
    possibly-permuted k-by-k identity matrix, such that

        Y[:, Js] = (Y[:,Js] @ X)[:, Js].
    """
    if axis == 1:
        # Column ID
        Q, S, J = la.qr(Y, mode='economic', pivoting=True)
        S_trailing = la.solve_triangular(S[:k, :k], S[:k, k:],
                                         overwrite_b=True,
                                         lower=False)
        X = np.zeros((k, Y.shape[1]))
        X[:, J] = np.hstack((np.eye(k), S_trailing))
        Js = J[:k]
        # Y \approx C @ X; C = Y[:, Js]
        return X, Js
    elif axis == 0:
        # Row ID
        X, Is = qrcp_osid(Y.T, k, axis=1)
        Z = X.T
        return Z, Is
    else:
        raise ValueError()


class RowOrColSelection:
    CALL_LEAD_DOC = \
    """
    Return a vector of k row indices (axis=0) or column indices (axis=1) for A.
    The goal is that these rows or columns capture "as much information about A
    as possible."
    """

    CALL_IO_DOC = \
    """
    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix to approximate.

    k : int
        Target rank for the approximation of A: 0 < k < min(A.shape).

    over : int
        Perform internal calculations with a sketch of rank (k + over).
        This is usually a small constant, e.g., 5 to 25. In some situations
        it's useful to set over = k.
    %s
    axis : int
        0 for a row ID, 1 for a column ID

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages randomness
        in this function call.

    Returns
    -------
    idxs : ndarray
        Skeleton indices
    """

    CALL_DOC = CALL_LEAD_DOC + CALL_IO_DOC

    @misc.set_docstring(CALL_DOC % '')
    def __call__(self, A, k, over, axis, gen):
        raise NotImplementedError()


@misc.set_docstring("""
    Return a vector of k row indices (axis=0) or column indices (axis=1) for A.
    Construct the skeleton indices by QRCP on a sketch of A.
    """ + (
        RowOrColSelection.CALL_IO_DOC % """
    p : int
        Total number of passes over A. Use p - 1 passes as part of a
        power iteration method to help find a more accurate solution. 
    """))
def rocs1(A, k, over, p, axis, rng):
    rng = np.random.default_rng(rng)
    skop = osk.SkOpGA()
    rs = ask.RS1(skop, p - 1, ulaw.orth, passes_per_stab=1)
    alg = ROCS1(rs)
    idxs = alg(A, k, over, axis, rng)
    return idxs


class ROCS1(RowOrColSelection):
    """
    Sketch + (QRCP skeleton)
    """

    def __init__(self, sk_op: RowSketcher):
        self.sk_op = sk_op

    @misc.set_docstring(RowOrColSelection.CALL_DOC % '')
    def __call__(self, A, k, over, axis, rng):
        rng = np.random.default_rng(rng)
        if axis == 0:
            # Row ID
            Sk = self.sk_op(A, k + over, rng)
            Y = A @ Sk
            _, _, I = la.qr(Y.T, mode='economic', pivoting=True)
            Is = I[:k]
            return Is
        elif axis == 1:
            # Column ID
            Sk = self.sk_op(A.T, k + over, rng).T
            Y = Sk @ A
            _, _, J = la.qr(Y, mode='economic', pivoting=True)
            Js = J[:k]
            return Js
        else:
            raise ValueError()
