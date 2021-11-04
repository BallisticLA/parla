"""
Data-aware sketching methods.
"""
import numpy as np
import parla.utils.sketching as usk
import parla.utils.linalg_wrappers as ulaw


def powered_range_sketch_op(num_pass, A, k, rng):
    """
    Return an n-by-k matrix S for use in sketching the rows of the m-by-n
    matrix A. (I.e., for computing a sketch Y = A @ S.) The qualitative goal
    is that the range of S should be well-aligned with the top-k right
    singular vectors of A.

    This function works by taking "num_pass" steps of a power method that
    starts with a random Gaussian matrix, and then makes alternating
    applications of A and A.T. We stabilize the power method with a QR
    factorization.

    Setting num_pass = 0 is a valid option.
    """
    assert num_pass >= 0
    assert k >= 1
    assert k <= min(A.shape)
    S = RS1(sketch_op_gen=usk.gaussian_operator,
            num_pass=num_pass,
            stabilizer=ulaw.orth,
            passes_per_stab=1)(A, k, rng)
    return S


class RowSketcher:
    """
    Given a matrix A and a positive integer k, generates a matrix S with k
    columns, for later use in sketching Y = A @ S. By virtue of taking
    linear combinations of the columns, the matrix S is essentially sketching
    the rows of A.
    """

    def __call__(self, A, k, rng):
        """
        Return a matrix S where range(S) is "reasonably" well aligned with
        the span of the top k right singular vectors of A.

        Do this while optionally incorporating information about A, e.g.,
        by subspace powering. It's possible that we construct the matrix S
        without accessing A, but in that situation the meaning of
        "reasonable" is very weak.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix we'll sketch later with Y = A @ S.

        k : int
            Number of columns of S.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        The simplest possible implementation of this function is

            S = np.random.standard_normal((A.shape[1], k))
            return S

        That is -- there is no requirement that implementations actually
        read the entries of A.
        """
        raise NotImplementedError()


class RS1(RowSketcher):
    """
    Powered Row Sketching Operator

    RS1 objects are used to create n-by-k matrices S for use in sketching
    the rows of an m-by-n matrix A. The qualitative goal is that the range
    of S should be well-aligned with the top-k right singular vectors of A.

    PRSO objects work by applying a power method that starts with an initial
    random matrix with k columns, and then makes alternating applications of
    A and A.T. The tuning parameters in this procedure are:

        How we generate the initial random matrix.
        The number of passes over A (or A.T).
        How we stabilize the power method. E.g., QR or LU factorization.
        How often we stabilize the power method.

    References
    ----------
    This implementation is inspired by [ZM:2020, Algorithm 3.3]. The most
    significant difference is that this function stops one step "early",
    so that it returns a matrix S for use in sketching Y = A @ S, rather than
    returning an orthonormal basis for a sketched matrix Y. Here are the
    differences between this implementation and [ZM:2020, Algorithm 3.3],
    assuming the latter algorithm was modified to stop "one step early" like
    this algorithm:

        (1) We make no assumptions on the distribution of the initial
            (oblivious) sketching matrix. [ZM:2020, Algorithm 3.3] uses
            a Gaussian distribution.

        (2) We allow any number of passes over A, including zero passes.
            [ZM2020: Algorithm 3.3] requires at least one pass over A.

        (3) We let the user provide the stabilization method. [ZM:2020,
            Algorithm 3.3] uses LU for stabilization.

        (4) We let the user decide how many applications of A or A.T
            can be made between calls to the stabilizer. We

    """

    def __init__(self, sketch_op_gen, num_pass, stabilizer, passes_per_stab):
        self.sketch_op_gen = sketch_op_gen
        self.num_pass = num_pass
        self.stabilizer = stabilizer
        self.passes_per_stab = passes_per_stab

    def __call__(self, A, k, rng):
        """
        Return a matrix S where range(S) is "reasonably" well aligned with
        the span of the top k right singular vectors of A.

        Do this with a subspace iteration approach that takes "self.num_pass"
        passes over A. The subspace iteration is initialized with a matrix
        from self.sketch_op_gen. The subspace iteration is stabilized by
        self.stabilizer, which is called after every "self.passes_per_stab"
        cumulative passes over A and A.T.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix we'll sketch later with Y = A @ S.

        k : int
            Number of columns of S.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        If computation was performed in exact arithmetic, then the range
        of the returned matrix S would match that of

            S = (A' A)^(self.num_pass/2) self.sketch_op_gen(n, k, rng)

        if self.num_pass is even, or that of

            S = (A' A)^((self.num_pass-1)/2) A' self.sketch_op_gen(m, k, rng)

        if self.num_pass is odd.
        """
        rng = np.random.default_rng(rng)
        passes_done = 0
        if self.num_pass % 2 == 0:
            S = self.sketch_op_gen(A.shape[1], k, rng)
        else:
            S = A.T @ self.sketch_op_gen(A.shape[0], k, rng)
            passes_done += 1
            if self.passes_per_stab == 1:
                S = self.stabilizer(S)
        q = (self.num_pass - passes_done) // 2
        # q is an even integer; need to compute
        #   S := (A' A)^q S
        # up to intermediate stabilization.
        while q > 0:
            S = A @ S
            passes_done += 1
            if passes_done % self.passes_per_stab == 0:
                S = self.stabilizer(S)
            S = A.T @ S
            passes_done += 1
            if passes_done % self.passes_per_stab == 0:
                S = self.stabilizer(S)
            q -= 1
        return S
