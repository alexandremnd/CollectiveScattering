import numpy as np
import scipy.linalg

from electric_field import PlaneWave, GaussianBeam, ScatteredField

def optical_lattice(Na, Nd, Rd, d, a):
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
    scatterers = np.zeros((Na, 3))
    for i in range(Na):
        disk = np.random.randint(low=0, high=Nd)
        x, y = Rd, Rd
        while x**2 + y**2 > Rd**2:
            x = np.random.uniform(low=-Rd, high=Rd)
            y = np.random.uniform(low=-Rd, high=Rd)
        scatterers[i] = np.array([x, y, disk * d + np.random.uniform(low=-a / 2, high=a / 2)])
    return scatterers

def centered_optical_lattice(Na, Nd, Rd, d, a):
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
    scatterers = optical_lattice(Na, Nd, Rd, d, a)
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

    xi, xj = np.meshgrid(scatterers[:, 0], scatterers[:, 0].T)
    yi, yj = np.meshgrid(scatterers[:, 1], scatterers[:, 1].T)
    zi, zj = np.meshgrid(scatterers[:, 2], scatterers[:, 2].T)
    norm = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

    A = np.zeros((Na, Na), dtype=np.complex64)

    A[norm != 0] = -0.5 * np.exp(2j * np.pi * norm[norm != 0]) / (2j * np.pi * norm[norm != 0])
    A = A + np.diag((1j * detuning - 0.5) * np.ones(Na), 0)
    X = np.dot(np.linalg.inv(A), B)
    return X

def excited_probabilities_lstsq(scatterers: np.ndarray, incident_field: PlaneWave | GaussianBeam, detuning: float):
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

    xi, xj = np.meshgrid(scatterers[:, 0], scatterers[:, 0].T)
    yi, yj = np.meshgrid(scatterers[:, 1], scatterers[:, 1].T)
    zi, zj = np.meshgrid(scatterers[:, 2], scatterers[:, 2].T)
    norm = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)

    A = np.zeros((Na, Na), dtype=np.complex64)

    A[norm != 0] = -0.5 * np.exp(2j * np.pi * norm[norm != 0]) / (2j * np.pi * norm[norm != 0])
    A = A + np.diag((1j * detuning - 0.5) * np.ones(Na), 0)
    X, _, _, _ = scipy.linalg.lstsq(A, B, lapack_driver='gelsy', check_finite=False, overwrite_a=True, overwrite_b=True)
    return X