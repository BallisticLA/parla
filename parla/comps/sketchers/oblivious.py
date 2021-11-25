"""
Data-oblivious sketching methods.
    Provide an OO interface around existing procedural implementations.

Names of implementation classes take the form "SkOp[XY]", where

        XY = ON: orthonormal

        XY = GA: Gaussian

        X = S: sparse (need a sparse matrix data structure)
            Y = J: sparse Johnson-Lindenstrauss
            Y = S: sparse sign operator
            Later, might do Y = L for LESS.

        X = T: trig transform
            Y = C: discrete cosine transform
            Later, might do Y = W for Walsh-Hadamard

        XY = IN: index into rows or columns
"""
import parla.utils.sketching as usk
import numpy as np


class SketchOpGen:

    def __call__(self, n_rows, n_cols, rng):
        raise NotImplementedError()


class SkOpON(SketchOpGen):
    """Generate an orthonormal sketching operator."""

    def __call__(self, n_rows, n_cols, rng):
        return usk.orthonormal_operator(n_rows, n_cols, rng)


class SkOpGA(SketchOpGen):
    """Generate a Gaussian sketching operator."""

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, n_rows, n_cols, rng):
        return usk.gaussian_operator(n_rows, n_cols, rng, self.normalize)


class SkOpSJ(SketchOpGen):
    """Generate a sparse sketching operator, based on SJLT."""

    def __init__(self, vec_nnz=8, complex=False):
        self.vec_nnz = vec_nnz
        self.complex = complex

    def __call__(self, n_rows, n_cols, rng):
        S = usk.sjlt_operator(n_rows, n_cols, rng, self.vec_nnz)
        if not self.complex:
            return S
        else:
            scales_re = rng.standard_normal((S.data.size,))
            scales_im = rng.standard_normal((S.data.size,))
            scales = scales_re + 1j*scales_im
            scales /= np.abs(scales)
            S.data = scales
            return S


class SkOpSS(SketchOpGen):
    """Generate a sparse-sign sketching operator."""

    def __init__(self, density=0.05):
        self.density = density

    def __call__(self, n_rows, n_cols, rng):
        usk.sparse_sign_operator(n_rows, n_cols, rng, self.density)


class SkOpTC(SketchOpGen):
    """Generate an SRTT sketching operator, based on the DCT-II."""

    def __call__(self, n_rows, n_cols, rng):
        return usk.srct_operator(n_rows, n_cols, rng)


class SkOpIN(SketchOpGen):
    """Generate a sketching operator which indexes into rows or columns."""

    def __init__(self, indices=None):
        self.indices = indices

    def __call__(self, n_rows, n_cols, rng):
        return usk.sampling_operator(n_rows, n_cols, rng, self.indices)
