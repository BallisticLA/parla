import numpy as np


class SketchAndPrecondLog:
    """
    Log runtime and error metric information from a call to a sketch-and-precondition
    least squares / saddle point solver.

    Attributes
    ----------
    time_sketch : float
        time to build S and compute S @ A.

    time_factor : float
        time to factor S @ A.

    time_convert : float
        time (if any) spent converting a problem from one form into an equivalent
        form required by the iterative solver. This is usually zero.

    time_presolve: float
        time to compute the initialization point for the iterative solver

    time_iterate : float
        total time spent by the iterative solver

    errors : ndarray
        A vector of error metrics as the approximate solution "x" changes from
        one iteration to the next. errors[0] is the error when x=0. errors[i]
        for i >= 1 is the error at the start of iteration i. The value at iteration
        1 is based on the user-provided initialization point for the iterative solver.

    times : ndarray
        times[i] is the total time spent by the algorithm to produce
        the iterate with error equal to errors[i]. These times are computed
        produced under the model that each iteration of the iterative solver
        takes the same time as every other iteration. The iterative solver
        ran for times.size - 1 iterations.

    error_desc : str
        A message that explains the metric used to populate "self.errors".

    Notes
    -----
    Populating "times" by amortizing the time spent by the deterministic iterative
    solver isn't representative for some least squares / saddle point problems with
    regularization parameter delta > 0. Reason being: those iterative solvers
    explicitly augment the data matrix A by a scaled identity matrix, and that will
    cause the first iteration to take more time than subsequent iterations.
    """

    def __init__(self):
        self.time_sketch = 0.0
        self.time_factor = 0.0
        self.time_presolve = 0.0
        self.time_convert = 0.0
        self._time_setup = 0.0
        self.time_iterate = 0.0
        self.times = None
        self.errors = None
        self.error_desc = """Fill in."""
        self.condnum_precond = None

    @property
    def time_setup(self):
        """
        Total time spent sketching, factoring, and (if applicable) converting the
        problem from a user-specified form into a solver-compatible form.
        """
        self._time_setup = self.time_sketch + self.time_factor + self.time_convert
        return self._time_setup

    def wrap_up(self, iter_errors, init_error):
        """
        Populate self.times and self.errors.

        Parameters
        ----------
        iter_errors : ndarray
            iter_errors[i] is some error metric for the i-th iterate produced by
            the iterative solver.

        init_error : float
            this reports the error metric using the same convention as was used
            to compute iter_errors, but this value is based on setting the iterate
            equal to the zero vector.
        """
        iters = iter_errors.size
        time_setup = self.time_setup
        iterating = np.linspace(0, self.time_iterate, iters, endpoint=True)
        cumulative = time_setup + self.time_presolve + iterating
        times = np.concatenate(([time_setup], cumulative))
        self.times = times
        self.errors = np.concatenate(([init_error], iter_errors))
        pass
