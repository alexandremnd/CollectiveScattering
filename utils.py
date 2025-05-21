import numpy as np

from electric_field import ScalarField

__all__ = ["bragg_direction", "periodicity_for_bragg_condition",
           "print_param", "save_param", "show_param_from_file"]

def print_param(Na, Nd, Rd, a, d, theta, disk_density):
    print(f"============= Paramètres =============")
    print(f"Nombre de diffuseurs : {Na}, nombre de disques : {Nd}")
    print(f"Rayon des disques : {Rd}, epaisseur : {a:.3f}")
    print(f"Distance entre les disques : {d:.3f}, longueur du réseau : {(Nd-1)*d:.2f}")
    print(f"Angle incidence : {np.rad2deg(theta):.2f}°")
    print(f"Densieé sans dimension des disques : {disk_density:.2f}")
    print("======================================")

def save_param(filepath, Na, Nd, Rd, a, d, theta, disk_density, detuning, w0, E0):
    with open(filepath, "w+") as f:
        f.write(f"Nombre de diffuseurs : {Na}, nombre de disques : {Nd}\n")
        f.write(f"Rayon des disques : {Rd}, epaisseur : {a:.3f}\n")
        f.write(f"Distance entre les disques : {d:.3f}, longueur du reseau : {(Nd-1)*d:.2f}\n")
        f.write(f"Angle incidence : {np.rad2deg(theta):.2f}°\n")
        f.write(f"Densite sans dimension des disques : {disk_density:.2f}\n")

        f.write(f"Detuning: {detuning:.4f}\n")
        f.write(f"w0: {w0:.4f}\n")
        f.write(f"E0 = {E0:.4f}\n")

def show_param_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())


def bragg_direction(incident_field: ScalarField, lattice_periodicity, order=1):
    """ Computes the Bragg direction for a given incident field and lattice periodicity.

    Args:
        incident_field (ScalarField): _description_
        lattice_periodicity (_type_): _description_
        order (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    lattice_vector = np.array([0, 0, 1/lattice_periodicity])
    incident_wave_vector = incident_field.k
    bragg_wave_vector = incident_wave_vector - lattice_vector*order
    bragg_direction = bragg_wave_vector/np.linalg.norm(bragg_wave_vector)
    return bragg_direction


# def get_lattice_reflectivity(incident_field: IncidentField, scattered_field: Scattered_Field,
#                              reflexion_direction, distance=100_000):
#     incident_wave_vector = incident_field.wave_vector
#     Ei = incident_field(*(distance * -incident_wave_vector))
#     Er = scattered_field(*(distance * reflexion_direction))
#     R = (np.abs(Er) ** 2) / (np.abs(Ei) ** 2)
#     return R


# Condition de Bragg/Laue
def periodicity_for_bragg_condition(theta):
    d = 1/(2*np.cos(theta))
    return d