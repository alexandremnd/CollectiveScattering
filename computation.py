import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from lattice import centered_optical_lattice, excited_probabilities
from electric_field import PlaneWave, GaussianBeam, ScatteredField


def compute_gamma_matrix(scatterers, gamma_matrix=None):
    """ Computes the gamma_jm matrix for the given scatterers.
    The gamma_jm matrix is defined as:
        gamma_jm = exp(2j * pi * norm) / (2j * pi * norm)
    where norm is the distance between scatterers.

    (i Δ0 - Gamma / 2) beta_j - Gamma / 2 sum_{j!=m} gamma_jm beta_m = d(beta_j)/dt

    which can be written as:

        i Δ0 beta_j - Gamma / 2 sum_{j} gamma_jm beta_m = d(beta_j)/dt

    where :
        gamma_jm = exp(2j * pi * norm) / (2j * pi * norm) if j != m
        gamma_jm = 1 if j == m.

    Args:
        scatterers (np.ndarray): Scatterers positions in 3D space (shape (Na, 3)).

    Returns:
        np.ndarray: (Na, Na) matrix of gamma_jm values.
    """
    norm = cdist(scatterers, scatterers)

    if gamma_matrix is not None:
        gamma_matrix[norm != 0] = np.exp(2j * np.pi * norm[norm != 0]) / (2j * np.pi * norm [norm != 0])
        gamma_matrix[np.diag_indices_from(gamma_matrix)] = 1
    else:
        gamma_matrix = np.ones_like(norm, dtype=np.complex128)
        gamma_matrix[norm != 0] = np.exp(2j * np.pi * norm[norm != 0]) / (2j * np.pi * norm [norm != 0])

    # print(f"Gamma matrix condition number: {np.linalg.cond(gamma_matrix)} | High condition number may lead to numerical instability")
    return gamma_matrix

def compute_intensity_angular(incident_field: PlaneWave | GaussianBeam, scattered_field: ScatteredField, theta=None, phi=None, resolution=250, distance=10):
    return compute_mean_angular_intensity(1, incident_field, scattered_field, theta, phi, resolution, distance)

def compute_mean_angular_intensity(iterations: int, Na, Nd, Rd, d, a, detuning, incident_field: PlaneWave | GaussianBeam, theta=None, phi=None, resolution=300, distance=1000):
    if theta is None:
        theta = np.linspace(0, np.pi, resolution)
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, resolution)

    tt, pp = np.meshgrid(theta, phi)

    x = distance * np.sin(tt) * np.cos(pp)
    y = distance * np.sin(tt) * np.sin(pp)
    z = distance * np.cos(tt)

    r = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

    I = np.zeros((resolution * resolution), dtype=np.float64)

    for i in tqdm(range(iterations)):
        scatterers = centered_optical_lattice(Na, Nd, Rd, d, a)
        amplitudes = excited_probabilities(scatterers, incident_field, detuning)
        scattered_field = ScatteredField(scatterers, amplitudes)

        I += np.abs(scattered_field(r) + incident_field(r))**2

    I /= iterations
    return I.reshape((resolution, resolution)), x, y, z

def compute_mean_intensity(iterations: int, Na, Nd, Rd, d, a, incident_field: PlaneWave | GaussianBeam, detuning, resolution=250, zoom=7):
    """ Computes the mean intensity of the field.
    The mean intensity is defined as the average of the intensity over the iterations.

    Args:
        iterations (int): Number of iterations to compute the mean intensity.
        Na (int): Number of scatterers.
        Nd (int): Number of disks.
        Rd (float): Radius of the disks.
        d (float): Distance between the disks.
        a (float): Thickness of the disks.
        incident_field (PlaneWave | GaussianBeam): Incident field object.
        detuning (float): Detuning of the incident field with the atomic transition.
        resolution (int, optional): How much point per axis on which the total field is evaluated. Defaults to 250.
        zoom (int, optional): How far away are we going ?. Defaults to 7.

    Returns:
        _type_: _description_
    """
    lim = (Nd-1) * d
    extent = np.linspace(- zoom * lim, zoom * lim, resolution)

    X, Z = np.meshgrid(extent, extent)
    X = X.reshape((-1, 1))
    Z = Z.reshape((-1, 1))
    Y = np.zeros_like(X)
    r = np.hstack((X, Y, Z))

    I = np.zeros_like(X, dtype=np.float64).flatten()

    for i in tqdm(range(iterations)):
        scatterers = centered_optical_lattice(Na, Nd, Rd, d, a)
        amplitudes = excited_probabilities(scatterers, incident_field, detuning)
        scattered_field = ScatteredField(scatterers, amplitudes)
        I += np.abs(scattered_field(r) + incident_field(r))**2

    I /= iterations
    return I, X.flatten(), Z.flatten(), scatterers

def compute_time_intensity(scatterers, amplitudes, theta=None, phi=None, resolution=250, distance=10):
    """ Computes the time evolution of the intensity of the field.

    Args:
        scatterers (np.ndarray): Scatterers positions in 3D space (shape (Na, 3)).
        amplitudes (np.ndarray): Amplitudes of the scattered field (shape (Na,)).

    Returns:
        np.ndarray: Time evolution of the intensity of the field (shape (Na,)).
    """

    if theta is None:
        theta = np.linspace(0, np.pi, resolution)
    if phi is None:
        phi = np.linspace(0, 2 * np.pi, resolution)
    resolution = len(theta)

    tt, pp = np.meshgrid(theta, phi)
    x = distance * np.sin(tt) * np.cos(pp)
    y = distance * np.sin(tt) * np.sin(pp)
    z = distance * np.cos(tt)
    r = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

    Nt = amplitudes.shape[1]
    I = np.zeros((Nt), dtype=np.float64)

    for i in range(Nt):
        scattered_field = ScatteredField(scatterers, amplitudes[:, i])
        I[i] = np.mean(np.abs(scattered_field(r))**2)

    return I