"""Basic showcase of how to use `scipy` implementation of differential evolution."""  # noqa: E501

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import pathlib
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# Particle swarm
# ==============================================================================
# ------------------------------------------------------------------------------
# Basic implementation
# ------------------------------------------------------------------------------
def particle_swarm_optimization(  # noqa: PLR0913
    f: Callable[[float, float], float],
    bounds: tuple[float, float],
    w: float,
    c1: float,
    c2: float,
    num_particles: int = 10,
    seed: int = 42,
    max_iter: int = 50,
) -> list[np.typing.NDArray]:
    """Optimize function `f` through particle swarm.

    Args:
        f (Callable[[float, float], float]): function to optimize
        bounds (tuple[float, float]): particle bounds.
        w (float): _description_
        c1 (float): _description_
        c2 (float): _description_
        num_particles (int, optional): _description_. Defaults to 10.
        seed (int, optional): _description_. Defaults to 42.
        max_iter (int, optional): _description_. Defaults to 50.

    Returns:
        list[np.typing.NDArray]: particle position history through optimization

    """
    # For reproducibility
    rnd: np.random.Generator = np.random.default_rng(seed)

    # Initialize particle positions
    particles: np.typing.NDArray = rnd.uniform(
        bounds[0],
        bounds[1],
        (num_particles, 2),
    )

    # Initialize velocities
    velocities: np.typing.NDArray = np.zeros_like(particles)

    personal_best: np.typing.NDArray = particles.copy()
    personal_best_values: np.typing.NDArray = np.array(
        [f(x, y) for x, y in particles],
    )

    global_best: np.typing.NDArray = personal_best[
        np.argmin(personal_best_values)
    ]
    global_best_value: np.typing.NDArray = np.min(personal_best_values)

    history = [particles.copy()]

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = rnd.random(), rnd.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )
            particles[i] += velocities[i]

            value = f(particles[i, 0], particles[i, 1])
            if value < personal_best_values[i]:
                personal_best[i] = particles[i]
                personal_best_values[i] = value

            if value < global_best_value:
                global_best = particles[i]
                global_best_value = value

        history.append(particles.copy())

    return history


# ==============================================================================
# Paper Figure 4[c-d]
# ==============================================================================
# ------------------------------------------------------------------------------
# Function: f(x, y) = sin(PI * x) * cos(PI * y) + (x^2 + y^2)  # noqa: ERA001
# Used
# ------------------------------------------------------------------------------
def f(x: float, y: float) -> float:
    """Compute `f(x, y) = sin(PI * x) * cos(PI * y) + (x^2 + y^2)`.

    Args:
        x (float): _description_
        y (float): _description_

    Returns:
        float: sin(PI * x) * cos(PI * y) + (x^2 + y^2)

    """
    return np.sin(np.pi * x) * np.cos(np.pi * y) + (x**2 + y**2)


# ------------------------------------------------------------------------------
# ~ Plotting
def plot(  # noqa: PLR0913
    X: np.typing.NDArray,  # noqa: N803
    Y: np.typing.NDArray,  # noqa: N803
    Z: np.typing.NDArray,  # noqa: N803
    history: list[tuple[float, float]],
    display: bool = False,  # noqa: FBT001, FBT002
    export: None | str = None,
) -> None:
    """Plot the gradient descent.

    Args:
        X (np.typing.NDArray): X to evaluate
        Y (np.typing.NDArray): Y to evaluate
        Z (np.typing.NDArray): f(X, Y)
        history (list[tuple[float, float]]): Gradient descent history.
        display (bool, optional): Show the figure. Defaults to False.
        export (None | str, optional): Path of the pdf file to generate, None to not export. Defaults to None.

    """  # noqa: E501
    # ~ Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("f(x, y)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    for pn in range(len(history[0])):
        history_x = [history[i][pn][0] for i in range(len(history))]
        history_y = [history[i][pn][1] for i in range(len(history))]
        plt.plot(
            history_x,
            history_y,
            color="red",
            marker="o",
            linestyle="dashed",
        )

    # ~ Draw global minima
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]
    plt.plot([min_x], [min_y], color="orange", marker="o")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)

    if export is not None:
        # ~ Export
        plt.savefig(
            export,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
            dpi=600,
        )
    if display:
        # ~ Display
        plt.show()
    plt.close()


def plot_step(  # noqa: PLR0913
    X: np.typing.NDArray,  # noqa: N803
    Y: np.typing.NDArray,  # noqa: N803
    Z: np.typing.NDArray,  # noqa: N803
    history: list[tuple[float, float]],
    best_id: int,
    target_id: int,
    display: bool = False,  # noqa: FBT001, FBT002
    export: None | str = None,
) -> None:
    """Plot the gradient descent with 1 candidate update step (mutation and recombination assumed 1).

    Args:
        X (np.typing.NDArray): X to evaluate
        Y (np.typing.NDArray): Y to evaluate
        Z (np.typing.NDArray): f(X, Y)
        history (list[tuple[float, float]]): Gradient descent history.
        best_id (int): ID of the particle with optimum
        target_id (int): ID of the particle to highlight
        display (bool, optional): Show the figure. Defaults to False.
        export (None | str, optional): Path of the pdf file to generate, None to not export. Defaults to None.

    """  # noqa: E501
    # ~ Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("f(x, y)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.plot(
        [history[-2][best_id][0]],
        [history[-2][best_id][1]],
        color="blue",
        marker="*",
        markersize=24,
    )

    plt.plot(
        [history[-2][target_id][0]],
        [history[-2][target_id][1]],
        color="lightblue",
        marker="*",
        markersize=24,
    )

    for pn in range(len(history[0])):
        history_x = [history[i][pn][0] for i in range(len(history))]
        history_y = [history[i][pn][1] for i in range(len(history))]
        col = "#ffcccc" if pn != target_id else "red"
        plt.plot(
            history_x,
            history_y,
            color=col,
            marker="o",
            linestyle="dashed",
            zorder=2 if pn != target_id else 5,
        )

    # ~ Draw global minima
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]
    plt.plot([min_x], [min_y], color="orange", marker="o")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)

    if export is not None:
        # ~ Export
        plt.savefig(
            export,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
    if display:
        # ~ Display
        plt.show()


# ------------------------------------------------------------------------------
# ~ Main
def main(outdir: str, videodir: str) -> None:
    """_summary_.

    Args:
        outdir (str): _description_
        videodir (str): _description_

    """
    # --------------------------------------------------------------------------
    # Global figure settings
    # --------------------------------------------------------------------------
    # Variable bounds
    bounds: list[tuple[float, float], tuple[float, float]] = [(-2, 2), (-2, 2)]

    # Plot function landscape and descent path
    x_vals: np.typing.NDArray = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_vals: np.typing.NDArray = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
    Z: np.typing.NDArray = f(X, Y)  # noqa: N806

    # Find the indices of the minimum value
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]
    min_value = Z[min_index]
    print(  # noqa: T201
        "Particle swarm: "
        f"Minimum value of f(x, y) is {min_value} "
        f"at coordinates ({min_x}, {min_y})",
    )

    # --------------------------------------------------------------------------
    # Figure 4d
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list = particle_swarm_optimization(
        f,
        (-2, 2),
        1,
        1.5,
        0.6,
        num_particles=10,
        seed=42,
        max_iter=1,
    )
    best_id: int = np.argmin(f(pn[0], pn[1]) for pn in history[-2])
    target_id: int = 5

    # Plot
    plot_step(
        X,
        Y,
        Z,
        history,
        best_id,
        target_id,
        export=f"{outdir}/fig4c-particle_swarm_step.pdf",
    )

    # --------------------------------------------------------------------------
    # Figure 4d
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list = particle_swarm_optimization(
        f,
        (-2, 2),
        0.05,
        0.15,
        0.15,
        num_particles=10,
        seed=42,
        max_iter=50,
    )

    # Plot
    plot(X, Y, Z, history, export=f"{outdir}/fig4d-particle_swarm.pdf")

    # --------------------------------------------------------------------------
    # ~ Plot history for videos
    roundir: str = f"{videodir}/fig4d-particle_swarm/"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for i in range(len(history)):
        plot(
            X,
            Y,
            Z,
            history[:i+1],
            export=f"{roundir}/fig4d-particle_swarm_round_{i:03d}.png",
        )



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[2]}",
    )
