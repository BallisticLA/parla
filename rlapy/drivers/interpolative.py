import numpy as np
import scipy.linalg as la
from rlapy.comps.sketchers import RowSketcher
import rlapy.comps.interpolative as id_comps
import rlapy.utils.sketching as usk
import rlapy.utils.linalg_wrappers as ulaw
from rlapy.comps.sketchers import RS1

"""
Look at SciPy's 
    https://github.com/scipy/scipy/blob/master/scipy/linalg/_interpolative_backend.py
the underlying Fortran implementation accepts function handles for matvecs 
against LinearOperator objects. They branch on "dense or LinearOperator?" at 
the top-level of a given user-facing function. 
"""


class OneSidedID:
    """One-sided ID (row or column)"""

    def __call__(self, A, k, over, axis, gen):
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


class OSID3(OneSidedID):
    """
    Sketch + (QRCP skeleton) + (least squares) approach to ID

    See Dong & Martinsson, 2021.

    economic onesided ID that only selects the indices of col/row
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

class TwoSidedID:
    """Fixed rank Double ID"""

    def __call__(self, A, k, over, rng):
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

    def __call__(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        # TODO: start with col ID if A is tall
        X, Is = self.osid(A, k, over, axis=0, rng=rng)
        A = A[Is, :]
        Z, Js = id_comps.qrcp_osid(A, k, axis=1)
        return X, Is, Z, Js

class TSID2(TwoSidedID):
    """

    economic twosided ID that only selects the indices of col/row
    """

    def __init__(self, osid: OneSidedID):
        self.osid = osid

    def __call__(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        # TODO: start with col ID if A is tall
        Is = self.osid(A, k, over, axis=0, rng=rng)
        A = A[Is, :]
        Js = self.osid(A, k, over, axis=1, rng=rng)
        return Is, Js


class CURDecomposition:
    """Fixed rank CUR Decomposition"""

    def __call__(self, A, k, over, rng):
        """
        Return (C, R) where
            C is A.shape[0]-by-k,
            R is k-by-A.shape[1],
            Uinv is a function that takes B and outputs A[Is, Js]^{-1}B
        so that
            A \approx C @ U @ R.

        Use oversampling parameter "over" in the sketching step.
        """
        raise NotImplementedError()

class CURD1(CURDecomposition):
    """
    Obtain a two-sided ID by any means, then extend to a CUR decomposition

    """
    def __init__(self, tsid: TwoSidedID):
        self.tsid = tsid
        self.A_s = None
    
    def evaluate_inverse(self, B):
        if self.A_s is None:
            raise ValueError('U not initialized')
        return la.solve(self.A_s, B)
    
    def __call__(self, A, k, over, rng):
        rng = np.random.default_rng(rng)
        Is, Js = self.tsid(A, k, over, rng=rng)
        print("shape I: ", Is.shape)
        print("shape J: ", Js.shape)
        self.A_s = A[Is, :][:, Js]
        U = self.evaluate_inverse
        return Is, Js, U

def test_cur(m, rank, k):
    A = np.random.randn(m, rank).astype(np.float64)
    A = A.dot(A.T)[:, :(m // 2)]
    rng = 1
    num_pass = 4
    over = 3
    print(A.shape)
    # ------------------------------------------------------------------------
    # test index_set == False

    curd = CURD1(TSID2(OSID3(RS1(sketch_op_gen=usk.gaussian_operator,
            num_pass=num_pass,
            stabilizer=ulaw.orth,
            passes_per_stab=1))))
    Is, Js, U = curd(A, k, over, rng)
    A_id = A[:, Js] @ U(A[Is, :])
    err = la.norm(A - A_id) / la.norm(A)
    print("error: ", err)
    assert err < 1e-4

test_cur(1000, 300, 300)