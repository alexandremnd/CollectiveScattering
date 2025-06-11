import numpy as np
from numba import vectorize, float64, complex128
from scipy.spatial.distance import cdist

from numba_ufunc import compute_gaussian, expc

__all__ = ['PlaneWave', 'GaussianBeam', 'ScatteredField']

class ScalarField(object):
    def __init__(self, E0, theta):
        self.E0 = E0
        self.theta = theta
        self.k = np.array([-np.sin(self.theta), 0, np.cos(self.theta)])

        self.u = self.k
        self.u2 = np.array([np.cos(self.theta), 0, np.sin(self.theta)])
        self.u3 = np.array([0, 1, 0])


class PlaneWave(ScalarField):
    def __call__(self, r):
        """Compute the electric field of a plane wave at a given position.

        Args:
            field (ScalarElectricField): The scalar electric field object containing E0, theta, and k.
            r (np.ndarray): The position vector where the field is computed.

        Returns:
            np.ndarray: The electric field at the given position.
        """
        return self.E0 * np.exp(1j * 2 * np.pi * np.dot(self.u, r.T))

class GaussianBeam(ScalarField):
    def __init__(self, E0, theta, w0):
        super().__init__(E0, theta)
        self.w0 = w0

    def __call__(self, r):
        """Compute the electric field of a Gaussian beam at a given position.

        Args:
            field (ScalarElectricField): The scalar electric field object containing E0, theta, and k.
            r (np.ndarray): The position vector where the field is computed.

        Returns:
            np.ndarray: The electric field at the given position.
        """
        z = np.dot(self.u, r.T)
        r2 = np.dot(self.u2, r.T)**2 + np.dot(self.u3, r.T)**2

        zR = np.pi * (self.w0**2) # Rayleigh range
        wz = self.w0 * np.sqrt(1 + (z / zR)**2) # Waist at z

        Rz = np.ones(z.shape) * np.inf
        Rz[z != 0] = z[z != 0] * (1 + (zR / z[z != 0])**2) # Radius of curvature of the wavefront

        phase = np.arctan(z / zR)
        return compute_gaussian(z, self.E0, self.w0, r2, Rz, wz, phase)

class ScatteredField(ScalarField):
    def __init__(self, scatterers: np.ndarray, amplitudes):
        self.scatterers = scatterers
        self.amplitudes = amplitudes

    def __call__(self, r, out=None):
        norm = cdist(r, self.scatterers, metric='euclidean')
        result = -1j * self.amplitudes.T * expc(norm)

        if out is None:
            return np.sum(result, axis=1)
        return np.sum(result, axis=1, out=out)