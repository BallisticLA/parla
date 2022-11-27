from parla.randblas.enums import DenseDist
from typing import Optional
import numpy as np
import scipy.sparse as spar
import parla.utils.sketching as sk_utils


# Later:
#   smskf: [s]a[m]ple a [sk]etching operator that's FFT-based.
#   smsks: [s]a[m]ple a [sk]etching operator that's [s]parse.
#   smsk3: [s]a[m]ple a [sk]etching operator that takes BLAS [3] level work.


class SketchingBuffer:

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
        Qualitative conclusions:
            (*) For now, take an "all or nothing" approach to memory management.
                I.e., user gives us a full buffer, or user gives us nothing, and
                we maybe allocate a full buffer internally (depend on implementation
                and whether persistent=True).
            (*) For things like Haar sketching, workspace becomes more complicated
                because there's a block size to tune.
                    (*) If ORMQR is the only LAPACK function that the RandBLAS might
                        need then we can copy-paste an source code from an appropriate
                        implementation. But there will still be the matter of block size.
        """
        self.dist = dist
        self.ctr_offset = ctr_offset
        self.key = key
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.populated = populated
        self.buff = buff
        self.persistent = persistent
        if self.populated:
            assert self.buff is not None
        if self.buff is not None:
            assert self.persistent
        """
        Roughly speaking, when S.buff is interpeted in column-major ...
            buff[i + n_rows * j] = cbrng(ctr_offset + i + j*n_rows),
        otherwise,
            buff[i * n_cols + j] = cbrng(ctr_offset + i*n_cols + j).

        The qualifier "rough" is needed in both cases because one call to
        the cbrng generates multiple numbers.
        """


def populated_dense_buff(S: SketchingBuffer, rng):
    size = S.n_rows * S.n_cols
    if S.dist == DenseDist.Uniform:
        buff = 2 * (rng.random(size=size) - 0.5)
    elif S.dist == DenseDist.Rademacher:
        buff = rng.choice([-1.0, 1.0], size=size)
    else:
        buff = rng.normal(size=size)
    return buff


class SASO:

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 vec_nnz: int,
                 key: int,
                 ctr_offset: int,
                 mat: Optional[spar.spmatrix] = None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vec_nnz = vec_nnz
        self.key = key
        self.ctr_offset = ctr_offset
        self.mat = mat
        if mat is not None:
            assert mat.shape == (n_rows, n_cols)
        self.persistent = False

    @property
    def populated(self):
        return self.mat is not None


def populated_saso(S: SASO, rng) -> spar.spmatrix:
    S_op = sk_utils.sjlt_operator(S.n_rows, S.n_cols, rng, S.vec_nnz)
    return S_op
