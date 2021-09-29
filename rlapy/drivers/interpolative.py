import numpy as np
import scipy.linalg as la
from rlapy.comps.sketchers import RowSketcher
import rlapy.comps.interpolative as id_comps

"""
Look at SciPy's 
    https://github.com/scipy/scipy/blob/master/scipy/linalg/_interpolative_backend.py
the underlying Fortran implementation accepts function handles for matvecs 
against LinearOperator objects. They branch on "dense or LinearOperator?" at 
the top-level of a given user-facing function. 
"""


class OneSidedID:
    """One-sided ID (row or column)"""

    def exec(self, A, k, over, axis, gen):
        """
        Run a rank-k RowID (axis=0) or ColumnID (axis=1) on A,
        using oversampling parameter over.

        A RowID consists of a matrix "X" and a length-k index vector "Is" so
        that A \approx X @ A[Is,:]. The rows of X must contain a
        possibly-permuted k-by-k identity matrix.

        A ColumnID consists of a matrix "Z" and a length-k index vector "Js" so
        that A \approx A[:,Js] @ Z. The columns of Z must contain a
        possibly-permuted k-by-k identity matrix.
        """
        raise NotImplementedError()


class OSID1(OneSidedID):
    """
    Sketch + QRCP approach to ID

    See Voronin & Martinsson, 2016, Section 5.1.
    """

    def __init__(self, sk_op: RowSketcher):
        self.sk_op = sk_op

    def exec(self, A, k, over, axis, rng):
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

    def exec(self, A, k, over, axis, rng):
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

    def exec(self, A, k, over, rng):
        """
        Return (X, Is, Z, Js) where
            X is A.shape[0]-by-k,
            Is is an index vector of length k,
            Z is k-by-A.shape[1],
            Js is an index vector of length k,
        so that
            A \approx X @ A[Is, Js] @ Z.

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

    def exec(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        # TODO: start with col ID if A is tall
        X, Is = self.osid(A, k, over, axis=0, rng=rng)
        A = A[Is, :]
        Z, Js = id_comps.qrcp_osid(A, k, axis=1)
        return X, Is, Z, Js
