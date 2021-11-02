import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator
from rlapy.comps.interpolative import RowOrColSelection
from rlapy.comps.sketchers.aware import RowSketcher
import rlapy.comps.interpolative as id_comps


class OneSidedID:
    """One-sided ID (row or column)"""

    def __call__(self, A, k, over, axis, gen):
        """
        Run a rank-k RowID (axis=0) or ColumnID (axis=1) on A,
        using oversampling parameter over.

        A RowID consists of a matrix "Z" and a length-k index vector "Is" so
        that A \approx Z @ A[Is,:]. The rows of Z must contain a
        possibly-permuted k-by-k identity matrix, such that

            A[Is, :] = (Z @ A[Is, :])[Is,:].

        Note that if we assume the rows of A[Is, :] are linearly independent, then
        the above equations are equivalent to Z[Is, :] being a k-by-k identity matrix.

        A ColumnID consists of a matrix "X" and a length-k index vector "Js" so
        that A \approx A[:,Js] @ X. The columns of X must contain a
        possibly-permuted k-by-k identity matrix, such that

            A[:, Js] = (A[:,Js] @ X)[:, Js].

        As with row ID, this means X[:, Js] is a k-by-k identity matrix
        """
        raise NotImplementedError()


class OSID1(OneSidedID):
    """
    Sketch + QRCP approach to ID

    See Voronin & Martinsson, 2016, Section 5.1.
    """

    def __init__(self, sk_op: RowSketcher):
        self.sk_op = sk_op

    def __call__(self, A, k, over, axis, rng):
        rng = np.random.default_rng(rng)
        if axis == 0:
            # Row ID
            Sk = self.sk_op(A, k + over, rng)
            Y = A @ Sk
            X, Is = id_comps.qrcp_osid(Y, k, axis=0)
            return X, Is
        elif axis == 1:
            # Column ID
            Sk = self.sk_op(A.T, k + over, rng).T
            Y = Sk @ A
            Z, Js = id_comps.qrcp_osid(Y, k, axis=1)
            return Z, Js
        else:
            raise ValueError()


class OSID2(OneSidedID):
    """
    Sketch + (QRCP skeleton) + (least squares) approach to ID

    See Dong & Martinsson, 2021.
    """

    def __init__(self, sk_op: RowSketcher):
        self.sk_op = sk_op

    def __call__(self, A, k, over, axis, rng):
        rng = np.random.default_rng(rng)
        if axis == 0:
            # Row ID
            Sk = self.sk_op(A, k + over, rng)
            Y = A @ Sk
            _, _, I = la.qr(Y.T, mode='economic', pivoting=True)
            Is = I[:k]
            res = la.lstsq(A[Is, :].T, A.T)  # res[0] = pinv(A[Is,:].T) @ A.T
            X = res[0].T  # X = A @ pinv(A[Is, :])
            return X, Is
        elif axis == 1:
            # Column ID
            Sk = self.sk_op(A.T, k + over, rng).T
            Y = Sk @ A
            _, _, J = la.qr(Y, mode='economic', pivoting=True)
            Js = J[:k]
            res = la.lstsq(A[:, Js], A)
            Z = res[0]
            return Z, Js
        else:
            raise ValueError()


class TwoSidedID:
    """Fixed rank Double ID"""

    def __call__(self, A, k, over, rng):
        """
        Return (Z, Is, X, Js) where
            Z is A.shape[0]-by-k,
            Is is an index vector of length k,
            X is k-by-A.shape[1],
            Js is an index vector of length k,
        so that
            A \approx Z @ A[Is, Js] @ X.

        Use oversampling parameter "over" in the sketching step.
        """
        raise NotImplementedError()


class TSID1(TwoSidedID):
    """
    Obtain a one-sided ID by any means, then deterministically extend
    to a two-sided ID.

    Using OSID1 would make this a "Sketch + QRCP" approach to double ID,
    as described in Voronin & Martinsson, 2016, Sections 2.4 and 4.
    """

    def __init__(self, osid: OneSidedID):
        self.osid = osid

    def __call__(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        if A.shape[0] > A.shape[1]:
            X, Js = self.osid(A, k, over, axis=1, rng=rng)
            Z, Is = id_comps.qrcp_osid(A[:, Js], k, axis=0)
        else:
            Z, Is = self.osid(A, k, over, axis=0, rng=rng)
            X, Js = id_comps.qrcp_osid(A[Is, :], k, axis=1)
        return Z, Is, X, Js


class CURDecomposition:
    """Fixed rank CUR Decomposition"""

    def __call__(self, A, k, over, rng):
        """
        Return (J, U, I) where
            C = A[:, J] has k columns,
            R = A[I, :] has k rows,
            U is a linear operator that applies A[Is, Js]^{-1}B
        so that
            A \approx C @ U @ R.

        Use oversampling parameter "over" in the sketching step.
        """
        raise NotImplementedError()


class CURD1(CURDecomposition):
    """
    Use a randomized method to select k rows of A, then use
    full-rank QRCP to select k columns from the row-submatrix of A,
    then return column indices, linking matrix, and row indices.
    """

    def __init__(self, rocs: RowOrColSelection):
        self.rocs = rocs
    
    def __call__(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        Is = self.rocs(A, k, over, axis=0, rng=rng)
        _, _, Js = la.qr(A[Is, :], mode='economic', pivoting=True)
        Js = Js[:k]
        A_s = A[Is, :][:, Js]

        def u_matmul(x):
            return la.solve(A_s, x)

        def u_rmatmul(y):
            return la.solve(A_s.T, y)

        U = LinearOperator(shape=(Js.size, Is.size),
                           matvec=u_matmul, matmat=u_matmul,
                           rmatvec=u_rmatmul, rmatmat=u_rmatmul)

        return Js, U, Is
