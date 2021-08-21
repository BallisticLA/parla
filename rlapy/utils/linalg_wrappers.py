import numpy as np
import warnings
import scipy.linalg as la


def orth(S):
    return la.qr(S, mode='economic')[0]


def lu_stabilize(S):
    L = la.lu(S, permute_l=True)[0]
    return L


def lupt(M):
    """Factor M = L @ U @ P.T. Equivalently, M @ P = L @ U."""
    P, L, U = la.lu(M)
    return U.T, L.T, P
