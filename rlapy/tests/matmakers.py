from typing import Union
import numpy as np
import rlapy.utils.sketching as sk


def rand_low_rank(n_rows, n_cols, spectrum: Union[int, np.ndarray], rng):
    rng = np.random.default_rng(rng)
    if isinstance(spectrum, int):
        spectrum = rng.random(size=(spectrum,))
    spectrum = np.sort(spectrum)
    spectrum = spectrum[::-1]  # reverse
    spectrum /= spectrum[0]
    rank = spectrum.size
    U = sk.orthonormal_operator(n_rows, rank, rng)
    V = sk.orthonormal_operator(rank, n_cols, rng)
    M = (U * spectrum) @ V
    return M
