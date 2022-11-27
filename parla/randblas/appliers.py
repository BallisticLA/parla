from parla.randblas.enums import DenseDist, Side, Op, Layout, Uplo, Diag
from parla.randblas.operators import SketchingBuffer, SASO, populated_saso, populated_dense_buff
from parla.randblas.python_specific_helpers import write_back, to_2d_array
import numpy as np


# lskge3: [L]eft [SK]etch a [GE]neral matrix with a BLAS [3] operation (or operation of the same complexity).
# lskges: [L]eft [SK]etch a [GE]neral matrix with a [S]parse operator.
# lskgef: [L]eft [SK]etch a [GE]neral matrix with an [F]FT-like operator.


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

    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    S_ptr = S_struct.buff
    if S_ptr is None:
        S_ptr = populated_dense_buff(S_struct, rng)
        if S_struct.persistent:
            S_struct.buff = S_ptr
            S_struct.populated = True
    elif not S_struct.populated:
        assert S_ptr.ndim == 1
        pop_buff = populated_dense_buff(S_struct, rng)
        S_ptr[:] = pop_buff
        S_struct.populated = True

    # The dimensions for (A, S), rather than (op(A), op(S)).
    rows_A, cols_A = (m, n) if transA == Op.NoTrans else (n, m)
    rows_S, cols_S = (d, m) if transS == Op.NoTrans else (m, d)

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
    #   This won't be needed in RandBLAS.
    S = to_2d_array(S_ptr, rows_S, cols_S, lds, layout)
    A = to_2d_array(A_ptr, rows_A, cols_A, lda, layout)
    B = to_2d_array(B_ptr, d, n, ldb, layout)

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
    write_back(B, B_ptr, n, d, ldb, layout)
    pass


def lskges(layout: Layout,
           transS: Op,
           transA: Op,
           d: int,  # B is d-by-n
           n: int,  # op(A) is m-by-n
           m: int,  # op(S) is d-by-m
           alpha: float,
           S_struct: SASO,
           A_ptr: np.ndarray,
           lda: int,
           beta: float,
           B_ptr: np.ndarray,
           ldb: int):
    """
    [L]left-[SK]etch a [GE]neral matrix with a short-axis-sparse operator.

        B = alpha * op(S) @ op(A) + beta * B.

    Notes:
        layout specifies the layout for (A, B) as represented
        by vectors A_ptr and B_ptr.

        If transA == NoTrans and layout == ColMajor, then
        there are at least "lda * n" elements following A_ptr.

        The unordered pair {d,m} giving the dimensions of op(S) must
        be equal to the unordered pair {S_struct.n_rows, S_struct.n_cols}.
    """
    assert A_ptr.ndim == 1
    assert B_ptr.ndim == 1
    cbrng = np.random.Philox(key=S_struct.key, counter=S_struct.ctr_offset)
    rng = np.random.Generator(cbrng)
    if S_struct.populated:
        S_mat = S_struct.mat
    else:
        S_mat = populated_saso(S_struct, rng)
        if S_struct.persistent:
            S_struct.mat = S_mat

    # The dimensions for (A, S), rather than (op(A), op(S)).
    rows_A, cols_A = (m, n) if transA == Op.NoTrans else (n, m)
    rows_S, cols_S = (d, m) if transS == Op.NoTrans else (m, d)
    if S_mat.shape != (rows_S, cols_S):
        msg = f"""
        Can't apply a portion of a SASO. The SASOs provided
        dimensions are {S_mat.shape}, but the dimensions
        declared for (A, B) say the SASO should have shape
        {(rows_S, cols_S)}.
        """
        raise NotImplementedError(msg)

    # Check that the dimensions for (A, B) are compatible with the
    # provided stride parameters (lda, ldb) we'll use for (A_ptr, B_ptr).
    if layout == Layout.ColMajor:
        assert lda >= rows_A
        assert d >= ldb
    else:
        assert lda >= cols_A
        assert n >= ldb

    # Convert to appropriate NumPy arrays, since we can't easily access BLAS directly.
    #   This won't be needed in RandBLAS.
    A = to_2d_array(A_ptr, rows_A, cols_A, lda, layout)
    B = to_2d_array(B_ptr, d, n, ldb, layout)

    # Perform the multiplication
    if transS == Op.NoTrans and transA == Op.NoTrans:
        C = S_mat @ A
    elif transS == Op.NoTrans and transA == Op.Trans:
        C = S_mat @ A.T
    elif transS == Op.Trans and transA == Op.Trans:
        C = S_mat.T @ A.T
    else:
        C = S_mat.T @ A
    C *= alpha
    B *= beta
    B += C
    write_back(B, B_ptr, n, d, ldb, layout)
    pass


if __name__ == '__main__':
    # BLAS lets you take contiguous row-slices or column-slices by changing
    # the location of the "starting" pointer in some array.
    #
    #   Problems:
    #
    #       1. our API doesn't let you set the starting pointer for the
    #          sketching operator, so it only lets you take leading slices.
    #
    #       2. there isn't a notion of a starting pointer for sparse sketching
    #          operators.
    #
    #       3. SASOs aren't closed under selecting a subset of their rows.
    #
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
