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
    P, L, U = la.lu(M.T)
    return U.T, L.T, P


def lup(M):
    """Factor M = L @ U @ P"""
    P, L, U = la.lu(M.T)
    return U.T, L.T, P.T


def apply_pinv_on_left(target, operator):
    """return res = pinv(operator) @ target"""
    res = la.lstsq(operator, target)[0]
    return res


def apply_pinv_on_right(target, operator):
    """return res = target @ pinv(operator)"""
    res = la.lstsq(operator.T, target.T)[0].T
    return res
