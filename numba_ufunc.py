from numba import vectorize, float64, complex128, guvectorize, njit, prange
import numpy as np

@vectorize([complex128(float64, float64, float64, float64, float64, float64, float64)], nopython=True)
def compute_gaussian(z, E0, w0, r2, Rz, wz, phase):
    return E0 * (w0 / wz) * np.exp(-r2 / (wz**2)) * np.exp(1j * 2 * np.pi * z + 1j * 2 * np.pi * r2 / (2 * Rz) - 1j * phase)

@vectorize([complex128(float64)], nopython=True)
def expc(x):
    """ Computes exp(2 * pi * i * x) / (2 * pi * i * x)

    Args:
        x (float | np.ndarray): Input value or array of values.

    Returns:
        float | np.ndarray: Computed expc value(s).
    """
    if x == 0:
        return 1
    return np.exp(1j * 2 * np.pi * x) / (2j * np.pi * x)
