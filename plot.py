import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go

def show_scatterers(scatteres: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(f"Distribution des {len(scatteres)} diffuseurs")
    ax.set_xlabel('$x/\\lambda$')
    ax.set_ylabel('$y/\\lambda$')
    ax.set_zlabel('$z/\\lambda$')
    ax.scatter(scatteres[:, 0], scatteres[:, 1], scatteres[:, 2])
    plt.show()

def show_intensity_xOz(I, x, z, scatterers,  *arrow_directions, cmap='binary'):
    x_max = np.max(scatterers[:, 0])
    x_min = np.min(scatterers[:, 0])
    z_max = np.max(scatterers[:, 2])
    z_min = np.min(scatterers[:, 2])

    fig = plt.figure()
    ax = fig.add_subplot()

    surface = ax.pcolormesh(z, x, I, cmap=cmap, norm=LogNorm(vmin=np.min(I), vmax=np.max(I)), shading='gouraud') # norm=LogNorm(vmin=np.min(I), vmax=np.max(I))
    fig.colorbar(surface, ax=ax)

    rect = Rectangle((z_min, x_min), width=z_max-z_min, height=x_max-x_min, linewidth=1.5, linestyle='--',
                     edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    for direction in arrow_directions:
        plt.quiver(0, 0, direction[2], direction[0], angles='xy', scale=8)

    plt.xlabel('$z/\\lambda$')
    plt.ylabel("$x/\\lambda$")
    ax.set_title(f"Intensité $|E|^2/E_0^2$")
    plt.show()

def show_intensity_3D(I, x, y, z, D, full_render=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Intensité à D={D}λ")
    ax.set_xlabel('$x/\\lambda$')
    ax.set_ylabel('$y/\\lambda$')
    ax.set_zlabel('$z/\\lambda$')

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

def show_intensity_3D_plotly(I, x, y, z, D):
    # Create logarithmic color values
    log_I = np.log10(np.maximum(I, 1e-300))
    vmin = np.min(log_I)
    vmax = np.max(log_I)

    # Create the surface plot with log-scaled colors
    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        surfacecolor=log_I,
        colorscale='Jet',
        colorbar=dict(
            title='Intensity',
            tickvals=np.linspace(vmin, vmax, 6),
            ticktext=[f"{10**val:.2e}" for val in np.linspace(vmin, vmax, 6)]
        )
    )])

    # Set the camera position to approximate the matplotlib view_init
    fig.update_layout(
        title=f"Intensité à D={D}λ",
        scene=dict(
            xaxis_title=r'$x/\lambda$',
            yaxis_title=r'$y/\lambda$',
            zaxis_title=r'$z/\lambda$',
            camera=dict(
                eye=dict(x=-0.5, y=-2, z=-0.15)
            )
        )
    )

    fig.show()

def plot_eigenvalues(eigenvalues, decomposition):
    ith_vector = np.arange(len(eigenvalues))

    gs = GridSpec(2, 1, hspace=0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)

    ax1.bar(ith_vector, decomposition.flatten(), color="tab:red")
    ax1.set_ylabel("$|\\langle u_i | \\Psi_{stat} \\rangle |^2$")

    ax2.plot(ith_vector, np.real(eigenvalues).flatten(), color="tab:red", marker="o")
    ax2.plot(ith_vector, np.imag(eigenvalues).flatten(), color="tab:red", marker="o")
    ax2.set_xlabel("Numéro de l'état propre")

    plt.tight_layout()
    plt.show()