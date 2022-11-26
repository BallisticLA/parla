from ctypes import c_uint64, c_uint32
from enum import Enum
from typing import Optional, Union
import numpy as np


class Layout(Enum):
    ColMajor = 'C'
    RowMajor = 'R'


class Op(Enum):
    NoTrans = 'N'
    Trans = 'T'
    ConjTrans = 'C'


class Uplo(Enum):
    Upper = 'U'
    Lower = 'L'
    General = 'G'


class Diag(Enum):
    NonUnit = 'N'
    Unit = 'U'


class Side(Enum):
    Left = 'L'
    Right = 'R'


class DenseDist(Enum):
    Gaussian = 'G'
    Uniform = 'U'
    Rademacher = 'R'
    Haar = 'H'  # needs LAPACK-level routines (or custom implementation).


class RNGCounter:

    def __init__(self,
                 ctr: Union[c_uint32, c_uint64],
                 key: Union[c_uint32, c_uint64]):
        self.ctr = ctr
        self.key = key


class SketchingBuffer:

    def __init__(self,
                 dist: DenseDist,
                 ctr_offset: int,
                 key: int,
                 n_rows: int,
                 n_cols: int,
                 buff: Optional[np.ndarray] = None
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
        self.populated = False
        self.persistent = False
        """
        Roughly speaking, when S.buff is interpeted in column-major ...
            buff[i + n_rows * j] = cbrng(ctr_offset + i + j*n_rows),
        otherwise,
            buff[i * n_cols + j] = cbrng(ctr_offset + i*n_cols + j).

        The qualifier "rough" is needed in both cases because one call to
        the cbrng generates multiple numbers.
        """
        self.buff = buff


# lskgef: [L]eft [SK]etch a [GE]neral matrix with an [F]FT-like operator.
# lskges: [L]eft [SK]etch a [GE]neral matrix with a [S]parse operator.
# lskge3: [L]eft [SK]etch a [GE]neral matrix with a BLAS [3] operation (or operation of the same complexity).

#   smskf: [s]a[m]ple a [sk]etching operator that's FFT-based.
#   smsks: [s]a[m]ple a [sk]etching operator that's [s]parse.
#   smsk3: [s]a[m]ple a [sk]etching operator that takes BLAS [3] level work.


def populated_dense_buff(S: SketchingBuffer, rng):
    size = S.n_rows * S.n_cols
    if S.dist == DenseDist.Uniform:
        buff = 2 * (rng.random(size=size) - 0.5)
    elif S.dist == DenseDist.Rademacher:
        buff = rng.choice([-1.0, 1.0], size=size)
    else:
        buff = rng.normal(size=size)
    return buff


def lskge3(layout: Layout,
           transS: Op,
           transA: Op,
           d: int,  # B is d-by-n
           n: int,  # op(A) is m-by-n
           m: int,  # op(S) is d-by-m
           alpha: float,
           S_struct: SketchingBuffer,
           A_ptr: np.ndarray,
           lda: int,
           beta: float,
           B_ptr: np.ndarray,
           ldb: int):
    """
    [L]left-[SK]etch a [GE]neral matrix with a BLAS [3] operation,
    or an operation of the same complexity:

        B = alpha * op(S) @ op(A) + beta * B.

    Notes:
        layout specifies the layout for (A, B, S) as represented
        by vectors A_ptr, B_ptr, and S.buff.

        If transA == NoTrans and layout == ColMajor, then
        there are at least "lda * n" elements following A_ptr.

        The dimensions {d,m} of op(S) aren't necessarily equal to
        {S.n_rows, S.n_cols}; the former can be larger than the latter.
    """
    assert A_ptr.ndim == 1
    assert B_ptr.ndim == 1
    assert S_struct.dist != DenseDist.Haar
    # if S_struct.buff is None and S_struct.populated:
    #    raise ValueError('Data in S is inconsistent.')

    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    if S_struct.buff is None:
        buff = populated_dense_buff(S_struct, rng)
        if S_struct.persistent:
            S_struct.buff = buff
            S_struct.populated = True
    else:
        buff = S_struct.buff
        assert buff.ndim == 1
        if not S_struct.populated:
            pop_buff = populated_dense_buff(S_struct, rng)
            buff[:] = pop_buff
        S_struct.populated = True
        S_struct.persistent = True

    # The dimensions for A, rather than op(A).
    if transA == Op.NoTrans:
        rows_A, cols_A = m, n
    else:
        rows_A, cols_A = n, m
    # Dimensions for S, rather than op(S).
    if transS == Op.NoTrans:
        rows_S, cols_S = d, m
    else:
        rows_S, cols_S = m, d

    # Check that the dimensions for (A, B) are compatible with the
    # provided stride parameters (lda, ldb) we'll use for (A_ptr, B_ptr).
    if layout == Layout.ColMajor:
        lds = S_struct.n_rows
        assert lds >= rows_S
        assert lda >= rows_A
        assert d >= ldb
    else:
        lds = S_struct.n_cols
        assert lds >= cols_S
        assert lda >= cols_A
        assert n >= ldb

    # Convert to appropriate NumPy arrays, since we can't easily access BLAS directly.
    # This won't be needed in RandBLAS.
    if layout == Layout.ColMajor:
        S = buff[:lds * cols_S].reshape((lds, cols_S), order='F')
        S = S[:rows_S, :]
        A = A_ptr[:lda * cols_A].reshape((lda, cols_A), order='F')
        A = A[:rows_A, :]
        B = B_ptr[:ldb*n].reshape((ldb, n), order='F')
        B = B[:d, :]
    else:
        S = buff[:lds * rows_S].reshape((rows_S, lds), order='C')
        S = S[:, :cols_S]
        A = A_ptr[:lda * rows_A].reshape((rows_A, lda), order='C')
        A = A[:, :cols_A]
        B = B_ptr[:ldb * d].reshape((d, ldb), order='C')
        B = B[:, :n]

    # Perform the multiplication
    if transS == Op.NoTrans and transA == Op.NoTrans:
        C = S @ A
    elif transS == Op.NoTrans and transA == Op.Trans:
        C = S @ A.T
    elif transS == Op.Trans and transA == Op.Trans:
        C = S.T @ A.T
    else:
        C = S.T @ A
    C *= alpha
    B *= beta
    B += C

    # Write back to memory in B_ptr.
    #   This is mind-bogglingly slow.
    #   It's needed because
    #   This won't be needed in RandBLAS.
    if layout == Layout.ColMajor:
        for col in range(n):
            start = ldb * col
            stop = start + d
            B_ptr[start:stop] = B[:, col]
    else:
        for row in range(d):
            start = ldb * row
            stop = start + n
            B_ptr[start:stop] = B[row, :]
    pass


class SASO:

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 vec_nnz: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vec_nnz = vec_nnz
        self.rows = None
        self.cols = None
        self.vals = None


def lskges():
    """[L]ight-[SK]etch of a [GE]neral matrix with a [S]parse operator"""
    pass


if __name__ == '__main__':
    S = SketchingBuffer(DenseDist.Uniform, 0, 0, n_rows=10, n_cols=50)
    np.random.seed(0)
    d = 10  # <= S.n_rows
    m = 40  # <= S.n_cols
    n = 4
    A = np.random.randn(m * n)
    lda = m
    B = np.zeros(d * n)
    ldb = d
    alpha = 1.0
    beta = 0.0
    lskge3(Layout.ColMajor,
           Op.NoTrans,
           Op.NoTrans,
           d, n, m,
           alpha, S, A, lda,
           beta, B, ldb)
