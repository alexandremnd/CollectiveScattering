import numpy as np
import matplotlib.pyplot as plt
from main import getBraggConditionLatticePeriodicity


def reflectivity_detuning(disk_density, d, theta_i, Na, Nd, Rd, a, show=True):
    detuning_list = np.linspace(-10, 10, 101)
    R_list = list()

    for detuning in detuning_list:
        zeta = -disk_density*a*(3/(4*np.pi))*1/(2*detuning + 1j)
        alpha = np.acos(np.cos(2*np.pi*d*np.cos(theta_i)) - zeta*np.sin(2*np.pi*d*np.cos(theta_i)))

        A = (1/np.sin(alpha))*np.array([[zeta*np.cos(2*np.pi*d*np.cos(theta_i)) + np.sin(2*np.pi*d*np.cos(theta_i)), zeta*np.exp(-1j*2*np.pi*d*np.cos(theta_i))],
                                        [-zeta*np.exp(1j*2*np.pi*d*np.cos(theta_i)), -zeta*np.cos(2*np.pi*d*np.cos(theta_i)) - np.sin(2*np.pi*d*np.cos(theta_i))]])

        M_Nd = np.cos(Nd*alpha)*np.array([[1, 0], [0, 1]]) + 1j*np.sin(Nd*alpha)*A

        R = np.abs(M_Nd[0, 1]/M_Nd[1, 1])**2
        R_list.append(R)

    if show:
        plt.figure()
        label = f"θ={theta_i*180/np.pi:.3}°\nNa={Na}\nd={d:.4}λ\na={a}λ\ndensity={disk_density}\nNd={Nd}"
        plt.plot(detuning_list, R_list, label=label)
        plt.axvline(x=0, label="Δ=0Γ", c='orange')
        plt.title("Coefficient de réflexion en intensité en fonction du désaccord")
        plt.xlabel("Δ/Γ")
        plt.ylabel("R")
        plt.legend()
        plt.show()
    return detuning_list, R_list


def detuning_of_max_reflecitivy_distance(disk_density, theta_i, Na, Nd, Rd, a):
    d_Bragg = getBraggConditionLatticePeriodicity(theta_i)
    d_list = np.linspace(d_Bragg - 0.002, d_Bragg + 0.002, 51)
    detuning_max = list()
    for d in d_list:
        detuning_list, R_list = reflectivity_detuning(disk_density, d, theta_i, Na, Nd, Rd, a, show=False)
        R_max_index = np.argmax(R_list)
        detuning_max.append(detuning_list[R_max_index])

    plt.figure()
    label = f"θ={theta_i*180/np.pi:.3}°\nNd={Nd}\na={a}λ\ndensity={disk_density}\nNa={Na}"
    plt.plot(d_list, detuning_max, label=label)
    plt.title("Désaccord au maximum de réflexion en fonction de la distance entre les disques")
    plt.axvline(x=d_Bragg, label="Condition de Bragg", c='orange')
    plt.xlabel("d/λ")
    plt.ylabel("Δ/Γ")
    plt.legend()
    plt.show()


disk_density = 3.25
Nd = 10
Rd = 9
a = 0.06
Na = int(disk_density * np.pi*(Rd**2)*a * Nd)
print(Na)
# Na = 10000
# disk_density = (Na / Nd) / (np.pi * (Rd ** 2) * a)
theta_i = np.pi * 2 / 180
d = getBraggConditionLatticePeriodicity(theta_i)

reflectivity_detuning(disk_density, d, theta_i, Na, Nd, Rd, a)
# detuning_of_max_reflecitivy_distance(disk_density, theta_i, Na, Nd, Rd, a)
