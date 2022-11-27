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
        if buff is not None:
            assert persistent
            # We don't require that buff is populated.
        if populated:
            assert buff is not None
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
        if mat is not None:
            assert persistent
            assert populated
            assert mat.shape == (n_rows, n_cols)
        pass


def populated_saso(S: SASO, rng) -> spar.spmatrix:
    S_op = sk_utils.sjlt_operator(S.n_rows, S.n_cols, rng, S.vec_nnz)
    return S_op
