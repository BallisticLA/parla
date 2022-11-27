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


def state_check(S_struct):
    # Start with checks that apply regardless of whether
    # self.populated == True.
    if S_struct.op_data is None:
        assert not S_struct.populated
        # Note: persistence is optional in this case.
    else:
        assert S_struct.persistent
        # Note: populating op_data is optional in this case.
    if S_struct.populated:
        assert S_struct.op_data is not None
        assert S_struct.persistent


class SketchingBuffer:
    """The (n_rows, n_cols) here are somewhat artificial, but they're needed for consistency with SASOs
    if we are to avoid "lds" parameters in applier functions."""

    def __init__(self,
                 dist: DenseDist,
                 ctr_offset: int,
                 key: int,
                 n_rows: int,
                 n_cols: int,
                 op_data: Optional[np.ndarray] = None,
                 populated: bool = False,
                 persistent: bool = False
                 ):
        """
        Roughly speaking, when S.op_data is interpreted in column-major ...
            op_data[i + n_rows * j] = cbrng(ctr_offset + i + j*n_rows),
        otherwise,
            op_data[i * n_cols + j] = cbrng(ctr_offset + i*n_cols + j).

        The qualifier "rough" is needed in both cases because one call to
        the cbrng generates multiple numbers.
        """
        self.dist = dist
        self.ctr_offset = ctr_offset
        self.key = key
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.op_data = op_data
        self.populated = populated
        self.persistent = persistent
        state_check(self)
        pass


def populated_dense_buff(S: SketchingBuffer, rng):
    size = S.n_rows * S.n_cols
    if S.dist == DenseDist.Uniform:
        buff = 2 * (rng.random(size=size) - 0.5)
    elif S.dist == DenseDist.Rademacher:
        buff = rng.choice([-1.0, 1.0], size=size)
    else:
        buff = rng.normal(size=size)
    return buff


def prep_dense_buff_sketching(S_struct: SketchingBuffer):
    assert S_struct.dist != DenseDist.Haar
    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    S_ptr = S_struct.op_data
    if S_ptr is None:
        S_ptr = populated_dense_buff(S_struct, rng)
        if S_struct.persistent:
            S_struct.op_data = S_ptr
            S_struct.populated = True
    elif not S_struct.populated:
        pop_buff = populated_dense_buff(S_struct, rng)
        S_ptr[:] = pop_buff
        S_struct.populated = True
    state_check(S_struct)
    return S_ptr


class SASO:

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 vec_nnz: int,
                 key: int,
                 ctr_offset: int,
                 op_data: Optional[spar.spmatrix] = None,
                 populated: bool = False,
                 persistent: bool = False):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vec_nnz = vec_nnz
        self.key = key
        self.ctr_offset = ctr_offset
        self.op_data = op_data
        self.persistent = persistent
        self.populated = populated
        state_check(self)
        pass


def populated_saso(S: SASO, rng) -> spar.spmatrix:
    S_op = sk_utils.sjlt_operator(S.n_rows, S.n_cols, rng, S.vec_nnz)
    return S_op


def prep_saso_sketching(S_struct: SASO):
    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    S_mat = S_struct.op_data
    if S_mat is None:
        S_mat = populated_saso(S_struct, rng)
        if S_struct.persistent:
            S_struct.op_data = S_mat
            S_struct.populated = True
    elif not S_struct.populated:
        raise NotImplementedError()
    state_check(S_struct)
    assert S_mat.shape == (S_struct.n_rows, S_struct.n_cols)
    return S_mat
