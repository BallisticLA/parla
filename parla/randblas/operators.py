from parla.randblas.enums import DenseDist
from typing import Optional
import numpy as np
import scipy.sparse as spar
import parla.utils.sketching as sk_utils


# Later:
#   smskf: [s]a[m]ple a [sk]etching operator that's FFT-based.
#   smsks: [s]a[m]ple a [sk]etching operator that's [s]parse.
#   smsk3: [s]a[m]ple a [sk]etching operator that takes BLAS [3] level work.


"""
The classes below will only be structs in the C++ implementation.
Any instance methods defined for the classes are only there to clarify semantics.

    There is some redundancy across the classes. This could be resolved by using
    inheritance, but I (Riley) think that would stray too far from the spirit of
    a struct.
"""


class SketchingBuffer:
    """The (n_rows, n_cols) here are somewhat artificial, but they're needed for consistency with SASOs
    if we are to avoid "lds" parameters in applier functions."""

    def __init__(self,
                 dist: DenseDist,
                 ctr_offset: int,
                 key: int,
                 n_rows: int,
                 n_cols: int,
                 buff: Optional[np.ndarray] = None,
                 populated: bool = False,
                 persistent: bool = False
                 ):
        """
        Roughly speaking, when S.buff is interpreted in column-major ...
            buff[i + n_rows * j] = cbrng(ctr_offset + i + j*n_rows),
        otherwise,
            buff[i * n_cols + j] = cbrng(ctr_offset + i*n_cols + j).

        The qualifier "rough" is needed in both cases because one call to
        the cbrng generates multiple numbers.
        """
        self.dist = dist
        self.ctr_offset = ctr_offset
        self.key = key
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.buff = buff
        self.populated = populated
        self.persistent = persistent
        self.state_check()
        pass

    def state_check(self):
        # Start with checks that apply regardless of whether
        # self.populated == True.
        if self.buff is None:
            assert not self.populated
            # Note: persistence is optional in this case.
        else:
            assert self.persistent
            # Note: populating buff is optional in this case.
        if self.populated:
            assert self.buff is not None
            assert self.persistent


def populated_dense_buff(S: SketchingBuffer, rng):
    size = S.n_rows * S.n_cols
    if S.dist == DenseDist.Uniform:
        buff = 2 * (rng.random(size=size) - 0.5)
    elif S.dist == DenseDist.Rademacher:
        buff = rng.choice([-1.0, 1.0], size=size)
    else:
        buff = rng.normal(size=size)
    return buff


def prep_dense_buff_sketching(S_struct):
    assert S_struct.dist != DenseDist.Haar
    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    S_ptr = S_struct.buff
    if S_ptr is None:
        S_ptr = populated_dense_buff(S_struct, rng)
        if S_struct.persistent:
            S_struct.buff = S_ptr
            S_struct.populated = True
    elif not S_struct.populated:
        pop_buff = populated_dense_buff(S_struct, rng)
        S_ptr[:] = pop_buff
        S_struct.populated = True
    S_struct.state_check()
    return S_ptr


class SASO:

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 vec_nnz: int,
                 key: int,
                 ctr_offset: int,
                 mat: Optional[spar.spmatrix] = None,
                 populated: bool = False,
                 persistent: bool = False):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vec_nnz = vec_nnz
        self.key = key
        self.ctr_offset = ctr_offset
        self.mat = mat
        self.persistent = persistent
        self.populated = populated
        self.state_check()
        pass

    def state_check(self):
        if self.mat is None:
            assert not self.populated
            # Note: persistence is optional in this case.
        else:
            assert self.persistent
            # Note: populating a pre-allocated "mat" will be
            # optional in C++, but we haven't implemented it here.
        if self.populated:
            assert self.mat is not None
            assert self.persistent
            assert self.mat.shape == (self.n_rows, self.n_cols)
            # ^ That last check is unique to this Python implementation.
        pass


def populated_saso(S: SASO, rng) -> spar.spmatrix:
    S_op = sk_utils.sjlt_operator(S.n_rows, S.n_cols, rng, S.vec_nnz)
    return S_op


def prep_saso_sketching(S_struct):
    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    S_mat = S_struct.mat
    if S_mat is None:
        S_mat = populated_saso(S_struct, rng)
        if S_struct.persistent:
            S_struct.mat = S_mat
            S_struct.populated = True
    elif not S_struct.populated:
        raise NotImplementedError()
    S_struct.state_check()
    return S_mat
