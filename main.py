import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plotly.io as pio
pio.renderers.default = "browser"

from electric_field import PlaneWave, GaussianBeam, ScatteredField, ScalarField
from lattice import centered_optical_lattice, excited_probabilities
from dynamics import *
from computation import compute_mean_intensity, get_intensity_xOz, compute_gamma_matrix, compute_intensity_angular, compute_mean_angular_intensity, compute_time_intensity
from plot import show_intensity_xOz, show_intensity_3D, show_intensity_3D_plotly





    if False:
        I, x, z, scatterers = compute_mean_intensity(100, Na, Nd, Rd, d, a, incident_field, detuning)

        show_intensity_xOz(I/E0**2, x, z, scatterers,
                        incident_field.k,
                        get_bragg_direction(incident_field, d))

        # I, x, y, z = compute_mean_angular_intensity(200, Na, Nd, Rd, d, a, detuning, incident_field, resolution=200, distance=200)
        # show_intensity_3D_plotly(I/E0**2, x, y, z, 200)

    # =================== Calcul dynamique ================
    # plt.legend(fancybox=False, framealpha=0)
    # plt.xlabel('Time $\\Gamma t$')
    # plt.ylabel('Intensité $I(t)/I(0)$')
    # plt.show()


    # P1 = np.sum(np.abs(beta1)**2, axis=0)
    # P2 = np.sum(np.abs(beta2)**2, axis=0)

    # plt.plot(t1, P1, label=f'Exact diagonalization ({Na} scatterers)', c="blue")
    # plt.plot(t2, P2, label=f'Euler ({Na} scatterers)', c="green")

    # plt.legend(fancybox=False, framealpha=0)
    # plt.xlabel('Time $\\Gamma t$')
    # plt.ylabel('$\\sum_j |\\beta_j(t)|^2$')
    # plt.yscale("log")

    # plt.tight_layout()
    # plt.show()