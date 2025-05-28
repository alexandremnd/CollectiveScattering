import numpy as np
from scipy.spatial.distance import cdist

from electric_field import *
from numba_ufunc import expc

def optical_lattice(Na, Nd, Rd, d, a, scatterers=None):
    """Generate a random distribution of scatterers in a disk.
    The scatterers are distributed in a disk of radius Rd, with Nd disks along the z-axis,

    Args:
        Na (int): Number of scatterers
        Nd (int): Number of disks along the z-axis
        Rd (float): Radius of the disk
        d (float): Distance between disks
        a (float): Thickness of the disks

    Returns:
        np.ndarray: Coordinates of the scatterers in the form of a 2D array with shape (Na, 3).
    """
    if scatterers is None:
        scatterers = np.empty((Na, 3), dtype=np.float64)

    u = np.random.uniform(low=0, high=1, size=(Na,))
    theta = np.random.uniform(low=0, high=2 * np.pi, size=(Na,))
    scatterers[:, 0] = Rd * np.sqrt(u) * np.cos(theta)
    scatterers[:, 1] = Rd * np.sqrt(u) * np.sin(theta)
    scatterers[:, 2] = np.random.randint(low=0, high=Nd, size=(Na,)) * d + np.random.uniform(low=-a / 2, high=a / 2)

    return scatterers

def centered_optical_lattice(Na, Nd, Rd, d, a, scatterers=None):
    """Generate a centered random distribution of scatterers in a disk.
    The scatterers are distributed in a disk of radius Rd, with Nd disks along the z-axis,
    and the center of the disk is at the origin.

    Args:
        Na (int): Number of scatterers
        Nd (int): Number of disks along the z-axis
        Rd (float): Radius of the disk
        d (float): Distance between disks
        a (float): Thickness of the disks

    Returns:
        np.ndarray: Coordinates of the scatterers in the form of a 2D array with shape (Na, 3).
    """
    scatterers = optical_lattice(Na, Nd, Rd, d, a, scatterers)
    scatterers -= np.array([0, 0, (Nd - 1) * d / 2])
    return scatterers

def excited_probabilities(scatterers: np.ndarray, incident_field: PlaneWave | GaussianBeam, detuning: float):
    """Compute the excited probabilities of the scatterers in the stationary regime.

    Args:
        scatterers (np.ndarray): Coordinates of the scatterers in the form of a 2D array with shape (Na, 3).
        incident_field (PlaneWave | GaussianBeam): Type of incident field.
        detuning (float): Detuning of the incident field.

    Returns:
        (np.ndarray): Excited probabilities of the scatterers in the form of a 2D array with shape (Na, 1).
    """
    Na = scatterers.shape[0]

    B = (1j / 2) * incident_field(scatterers)
    B = np.reshape(B, (Na, 1))

    norm = cdist(scatterers, scatterers, metric='euclidean')

    A = np.empty((Na, Na), dtype=np.complex64)

    A = -0.5 * expc(norm)
    A = A + np.diag(1j * detuning * np.ones(Na), 0)

    return np.linalg.solve(A, B)