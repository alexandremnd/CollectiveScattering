import numpy as np
from scipy.linalg import expm

def compute_time_evolution_euler(M: np.ndarray, initial_amplitudes, t = np.linspace(0, 20, 5000)):
    """ Compute the time evolution of the system using Euler's method.

    Args:
        M (np.ndarray): Evolution matrix of the system.
        initial_amplitudes (np.ndarray): Stationnary amplitudes of the system.

    Returns:
        _type_: _description_
    """

    Na = M.shape[0]
    Nt = t.shape[0]

    betat = np.zeros((Na, Nt), dtype=np.complex128)
    betat[:, 0] = initial_amplitudes.flatten()

    dt = t[1] - t[0]
    for i in range(1, Nt):
        betat[:, i] = (M * dt) @ betat[:, i-1] + betat[:, i-1]

    return betat

def compute_time_evolution_exp(M, initial_amplitudes, t = np.linspace(0, 20, 5000)):
    """ Compute the time evolution of the system using the exponential method which is:

            d(beta)/dt = - 0.5 gamma @ beta => beta(t) = exp(- 0.5 * gamma * t) @ beta(0)

        This method is expected to be more accurate than the Euler method, but less accurate than the exact diagonalization method.

    Args:
        M (np.ndarray): Evolution matrix of the system.
        initial_amplitudes (np.ndarray): Stationnary amplitudes of the system.

    Returns:
        np.ndarray, np.ndarray: time, beta(t)
    """
    Na = M.shape[0]
    Nt = t.shape[0]

    betat = np.zeros((Na, Nt), dtype=np.complex128)
    betat[:, 0] = initial_amplitudes.flatten()

    for i in range(1, Nt):
        betat[:, i] = expm(M * t[i]) @ betat[:, 0]

    return betat

def compute_time_evolution_exact_diagonalization(P, V, initial_amplitudes, t = np.linspace(0, 20, 500)):
    """ Compute the time evolution of the system using the exact diagonalization method.

    Args:
        P (np.ndarray): Matrix of eigenvectors of the system (j-th column is the j-th eigenvector).
        V (np.ndarray): Matrix of eigenvalues of the system (j-th eigenvalue should correspond to the j-th eigenvector).
        initial_amplitudes (np.ndarray): _description_
        Nt (int, optional): Number of time point. Defaults to 5000.

    Returns:
        _type_: _description_
    """
    Na = P.shape[0]
    P_inv = np.linalg.inv(P)

    t = t.reshape((1, -1))

    # V is a column vector
    # t is a row vector
    # (V @ t)_ij = V_i t_j such that row is for a beta_i at time t_j
    V = V.reshape((Na, 1))
    Dt = np.exp(V @ t)

    # Mathematically, D is a diagonal matrix, but to perform the time computation faster
    # Each diagonal of D at tj is in the j-th column
    # Thus, we cannot convert D(tj) into the non diagonal basis, but we can convert the initial condition into the diagonal basis
    # Compute at each time the new state in diagonal basis, and switch back to non diagonal basis.
    beta_d = P_inv @ initial_amplitudes
    beta_dt = Dt * beta_d
    beta_t = P @ beta_dt

    V = V.flatten()

    return  beta_t

def exact_solution_1scat(beta0, t):
    sol = beta0 * np.exp(- 0.5 * t)
    sol = np.abs(sol.flatten())**2
    return sol