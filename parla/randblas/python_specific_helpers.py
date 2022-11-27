from parla.randblas.enums import Layout


def to_2d_array(A_ptr, rows_A, cols_A, lda, layout):
    if layout == Layout.ColMajor:
        A = A_ptr[:lda * cols_A].reshape((lda, cols_A), order='F')
        A = A[:rows_A, :]
    else:
        A = A_ptr[:lda * rows_A].reshape((rows_A, lda), order='C')
        A = A[:, :cols_A]
    return A


def write_back(B, B_ptr, n, d, ldb, layout):
    # Write back to memory in B_ptr.
    #   This is mind-bogglingly slow.
    #   This won't be needed in RandBLAS.
    #   It's needed here in case B doesn't share the same
    #       memory as B_ptr.
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

