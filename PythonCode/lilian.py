import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


# Placement des diffuseurs
def optical_lattice(Na, Nd, Rd, d, a):
    scatterers = np.zeros((Na, 3))
    for i in range(Na):
        disk = np.random.randint(low=0, high=Nd)
        x, y = Rd, Rd
        while x**2 + y**2 > Rd**2:
            x = np.random.uniform(low=-Rd, high=Rd)
            y = np.random.uniform(low=-Rd, high=Rd)
        scatterers[i] = np.array([x, y, disk * d + np.random.uniform(low=-a / 2, high=a / 2)])
    return scatterers


def centred_optical_lattice(Na, Nd, Rd, d, a):
    scatteres = optical_lattice(Na, Nd, Rd, d, a)
    scatteres -= np.array([0, 0, (Nd - 1) * d / 2])
    return scatteres


# Champ incident (adimensioné) en fonction de la position (adimensionée)
class Field:
    def __call__(self, *r):
        return 0


class IncidentField(Field):
    def __init__(self, E0, theta):
        self.E0 = E0
        self.theta = theta
        # Vecteur d'onde adimensionné et normé
        self.wave_vector = np.array([-np.sin(self.theta), 0, np.cos(self.theta)])


class Plane_Wave(IncidentField):
    def __call__(self, *r):
        r = np.array(r)
        u = self.wave_vector
        E = self.E0 * np.exp(1j * 2 * np.pi * np.dot(u, r))
        return E


class Gaussian_Beam(IncidentField):
    def __init__(self, E0, theta, w0):
        super().__init__(E0, theta)
        self.w0 = w0

    def __call__(self, *r):
        r = np.array(r)
        u = self.wave_vector
        u2 = np.array([np.cos(self.theta), 0, np.sin(self.theta)])  # Vecteurs normés orthogaunaux au vecteur d'onde
        u3 = np.array([0, 1, 0])

        z = np.dot(r, u)
        r2 = (np.dot(r, u2)) ** 2 + (np.dot(r, u3)) ** 2

        z0 = np.pi * (self.w0 ** 2)
        wz = self.w0 * np.sqrt(1 + ((z / z0) ** 2))
        if z == 0:
            Rz = np.inf
        else:
            Rz = z * (1 + ((z0 / z) ** 2))
        phase = np.arctan(z / z0)
        E = self.E0 * (self.w0 / wz) * np.exp(-r2 / (wz ** 2)) * np.exp(
            1j * 2 * np.pi * z + 1j * 2 * np.pi * r2 / (2 * Rz) - 1j * phase)
        return E


class Scattered_Field(Field):
    def __init__(self, scatteres: np.array, amplitudes):
        self.scatteres = scatteres
        self.amplitudes = amplitudes

    def __call__(self, *r):
        r = np.array(r)
        norm = np.linalg.norm(self.scatteres - r, axis=1)
        result = -self.amplitudes * ((np.exp(1j * 2 * np.pi * norm)) / (2 * np.pi * norm))
        E_scattered = np.sum(result)
        return E_scattered


# Calcul des amplitudes d'excitations à partir des diffuseurs, du champ incident et du désaccord (tous adimensionnés)
def excitation_amplitudes(scatteres: np.array, incident_field: IncidentField, detuning):
    E = incident_field
    N = scatteres.shape[0]
    B = np.zeros(N, dtype=np.complex64)
    for j in range(N):
        B[j] = (1j / 2) * E(*scatteres[j])
    A = np.zeros((N, N), dtype=np.complex64)
    for j in range(N):
        for k in range(j + 1, N):
            rj = scatteres[j]
            rk = scatteres[k]
            norm = np.linalg.norm(rj - rk)
            A[j][k] = -(1 / 2) * ((np.exp(1j * 2 * np.pi * norm)) / (1j * 2 * np.pi * norm))
    A = A + A.T + np.diagflat((1j * detuning - 1 / 2) * np.ones(N), 0)
    X = np.dot(np.linalg.inv(A), B)

    print(B)
    print(A)
    return X


# Calcul du champ total
def get_total_field(field1: Field, field2: Field):
    return lambda *r: (field1(*r) + field2(*r))


# Affichage de la distribution de diffuseurs
def show_scatteres(scatteres: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(f"Distribution des {len(scatteres)} diffuseurs")
    ax.set_xlabel('x/λ')
    ax.set_ylabel('y/λ')
    ax.set_zlabel('z/λ')
    ax.scatter(scatteres[:, 0], scatteres[:, 1], scatteres[:, 2])
    plt.show()


# Affichage de l'intensité aux points à la distance d (sans dimension)
def get_intensity_3D(D, incident_field: IncidentField, scattered_field: Scattered_Field):
    r = D
    theta = np.linspace(0, np.pi, 80)
    phi = np.linspace(0, 2 * np.pi, 160)
    theta, phi = np.meshgrid(theta, phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Calcul de l'intensité adimensionnée du champ en plusieurs points d'une sphère de rayon d
    field = get_total_field(incident_field, scattered_field)
    I = np.abs(np.vectorize(field)(x, y, z)) ** 2
    return I, x, y, z


def show_intensity_3D(I, x, y, z, D, full_render=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Intensité sans dimension du champ à D={D}λ")
    ax.set_xlabel('x/λ')
    ax.set_ylabel('y/λ')
    ax.set_zlabel('z/λ')
    norm = colors.LogNorm(vmin=max(np.min(I), 1e-300), vmax=np.max(I))
    if full_render:
        surface = ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.jet(norm(I)), cmap=cm.jet)
    else:
        surface = ax.plot_surface(x, y, z, facecolors=cm.jet(norm(I)), cmap=cm.jet)
    cmap = fig.colorbar(surface, ax=ax)
    cmap.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    tick_labels = np.logspace(np.log(max(np.min(I), 1e-300)), np.log(np.max(I)), 6)
    cmap.set_ticklabels([f"{tick_labels[i]:.2e}" for i in range(len(tick_labels))])
    ax.view_init(elev=-9, azim=167, roll=-87)
    plt.show()


# Affichage de l'intensité dans le plan y = 0
def get_intensity_xOz(incident_field: IncidentField, scattered_field: Scattered_Field, scatteres):
    zoom = 7
    x_max = np.max(scatteres[:, 0])
    x_min = np.min(scatteres[:, 0])
    z_max = np.max(scatteres[:, 2])
    z_min = np.min(scatteres[:, 2])
    lim = max(np.abs(x_min), np.abs(x_max), np.abs(z_min), np.abs(z_max))
    x = np.linspace(-zoom * lim, zoom * lim, 200)
    z = np.linspace(-zoom * lim, zoom * lim, 200)

    x, z = np.meshgrid(x, z)

    total_field = get_total_field(scattered_field, incident_field)
    I = np.vectorize(lambda x, y, z: np.abs(total_field(x, y, z)) ** 2)(x, 0, z)
    return I, x, z


def show_intensity_xOz(I, x, z, scatteres,  *arrow_directions):
    x_max = np.max(scatteres[:, 0])
    x_min = np.min(scatteres[:, 0])
    z_max = np.max(scatteres[:, 2])
    z_min = np.min(scatteres[:, 2])

    fig = plt.figure()
    ax = fig.add_subplot()
    surface = ax.pcolormesh(z, x, I, cmap='jet', norm=LogNorm(vmin=np.min(I), vmax=np.max(I)))
    fig.colorbar(surface, ax=ax)
    rect = Rectangle((z_min, x_min), width=z_max-z_min, height=x_max-x_min, linewidth=1,
                     edgecolor='purple', facecolor='none')
    ax.add_patch(rect)
    for direction in arrow_directions:
        plt.quiver(0, 0, direction[2], direction[0], angles='xy', scale=8)
    plt.xlabel('z/λ')
    plt.ylabel('x/λ')
    ax.set_title(f"Intensité sans dimension |E|²")
    plt.show()


# Diffusion de Bragg
def get_bragg_direction(incident_field: IncidentField, lattice_periodicity, order=1):
    lattice_vector = np.array([0, 0, 1/lattice_periodicity])
    incident_wave_vector = incident_field.wave_vector
    bragg_wave_vector = incident_wave_vector - lattice_vector*order
    bragg_direction = bragg_wave_vector/np.linalg.norm(bragg_wave_vector)
    return bragg_direction


def get_lattice_reflectivity(incident_field: IncidentField, scattered_field: Scattered_Field,
                             reflexion_direction, distance=100_000):
    incident_wave_vector = incident_field.wave_vector
    Ei = incident_field(*(distance * -incident_wave_vector))
    Er = scattered_field(*(distance * reflexion_direction))
    R = (np.abs(Er) ** 2) / (np.abs(Ei) ** 2)
    return R


# Condition de Bragg/Laue
def getBraggConditionLatticePeriodicity(theta):
    d = 1/(2*np.cos(theta))
    return d


# Main Code
if __name__ == '__main__':
    np.set_printoptions(formatter={'all': lambda x: "{:.4e}".format(x)})
    Na = 5                                               # Nombre de diffuseurs
    Nd = 400                                                    # Nombre de disques
    Rd = 9                                                      # Rayon de chaque disque
    a = 0.25                                                    # Epaisseur des disques
    theta = np.deg2rad(15)                                # Angle d'incidence dans xOz par rapport à l'axe z
    E0 = 1e-3                                                      # Pulsation Rabi adimensionnée
    w0 = 10                                                     # Waist
    detuning = 1                                               # Ecart à la résonance

    d = getBraggConditionLatticePeriodicity(theta)
    # d = 0.501118
    disk_density = (Na/Nd)/(np.pi*(Rd**2)*a)
    print(f"Densité sans dimension des disques : {disk_density}")
    print(f"Distance entre les disques : {d}")
    print(f"Longueur du réseau : {(Nd-1)*d}")

    diffuseurs = centred_optical_lattice(Na, Nd, Rd, d, a)  # Calcul des positions des diffuseurs*
    diffuseurs = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    # show_scatteres(diffuseurs)  # Affichage des positions des diffuseurs
    incident_field = Gaussian_Beam(E0, theta, w0)  # Choix du champs incident


    amplitudes = excitation_amplitudes(diffuseurs, incident_field, detuning)  # Calcul des amplitudes d'excitation
    scattered_field = Scattered_Field(diffuseurs, amplitudes)  # Calcul du champ diffusé
    print(scattered_field((5, 5, 5)))
