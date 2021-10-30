import numpy as np
import scipy.linalg as la


def qrcp_osid(Y, k, axis):
    """One-sided rank-k ID of Y using QR with column pivoting"""
    if axis == 1:
        # Column ID
        Q, S, J = la.qr(Y, mode='economic', pivoting=True)
        S_trailing = la.solve_triangular(S[:k, :k], S[:k, k:],
                                         overwrite_b=True,
                                         lower=False)
        Z = np.zeros((k, Y.shape[1]))
        Z[:, J] = np.hstack((np.eye(k), S_trailing))
        Js = J[:k]
        # Y \approx C @ Z; C = Y[:, Js]
        return Z, Js
    elif axis == 0:
        # Row ID
        Z, Is = qrcp_osid(Y.T, k, axis=1)
        X = Z.T
        return X, Is
    else:
        raise ValueError()
