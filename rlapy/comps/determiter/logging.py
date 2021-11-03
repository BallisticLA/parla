import numpy as np


class SketchAndPrecondLog:

    def __init__(self):
        self.time_sketch = 0.0
        self.time_factor = 0.0
        self.time_presolve = 0.0
        self.time_convert = 0.0
        self._time_setup = 0.0
        self.time_iterate = 0.0
        self.times: np.ndarray
        self.errors: np.ndarray
        self.error_desc = """Fill in."""

    @property
    def time_setup(self):
        self._time_setup = self.time_sketch + self.time_factor \
                           + self.time_convert
        return self._time_setup

