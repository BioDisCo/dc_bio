"""Comparative of Gradient Descent with momentum and with noise."""

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
# Gradient descent
# ==============================================================================
# ------------------------------------------------------------------------------
# Basic implementation
# ------------------------------------------------------------------------------
def gradient_descent(  # noqa: PLR0913
    df_dx: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    x0: float,
    y0: float,
    learning_rate: float = 0.05,
    max_iter: int = 50,
    epsilon: float = 1e-5,
) -> list[tuple[float, float]]:
    """Compute a basic gradient descent for a function `f` up to 2 variables.

    Args:
        df_dx (Callable[[float, float], float]): derivative w.r.t. `x`
        df_dy (Callable[[float, float], float]): derivative w.r.t. `y`
        x0 (float): init value for `x`
        y0 (float): init value for `y`
        learning_rate (float, optional): learning rate. Defaults to 0.05.
        max_iter (int, optional): Max number of iterations. Defaults to 50.
        epsilon (float, optional): Stopping threshold. Defaults to 1e-5.

    Returns:
        list[tuple[float, float]]: iterative values of `x` and `y` through the gradient descent

    """  # noqa: E501
    x, y = x0, y0
    history: list[tuple[float, float]] = [(x, y)]  # Store descent path

    for _ in range(max_iter):
        # Compute gradient
        grad_x: float = df_dx(x, y)
        grad_y: float = df_dy(x, y)

        # Update step
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        history.append((x, y))

        # Stop if gradient is small
        if np.sqrt(grad_x**2 + grad_y**2) < epsilon:
            break

    return history


# ------------------------------------------------------------------------------
# With momentum
# ------------------------------------------------------------------------------
def gradient_descent_momentum(  # noqa: PLR0913
    df_dx: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    x0: float,
    y0: float,
    learning_rate: float = 0.05,
    max_iter: int = 50,
    momentum: float = 0.8,
    epsilon: float = 1e-5,
) -> list[tuple[float, float]]:
    """Compute a gradient descent with momentum for a function `f` up to 2 variables.

    Args:
        df_dx (Callable[[float, float], float]): derivative w.r.t. `x`
        df_dy (Callable[[float, float], float]): derivative w.r.t. `y`
        x0 (float): init value for `x`
        y0 (float): init value for `y`
        learning_rate (float, optional): learning rate. Defaults to 0.05.
        max_iter (int, optional): Max number of iterations. Defaults to 50.
        momentum (float, optional): Momentum.
        epsilon (float, optional): Stopping threshold. Defaults to 1e-5.

    Returns:
        list[tuple[float, float]]: iterative values of `x` and `y` through the gradient descent

    """  # noqa: E501
    x, y = x0, y0
    velocity_x, velocity_y = 0.0, 0.0  # Initialize velocities
    history: list[tuple[float, float]] = [(x, y)]  # Store descent path

    for _ in range(max_iter):
        # Compute gradient
        grad_x: float = df_dx(x, y)
        grad_y: float = df_dy(x, y)

        # Update velocities with momentum
        velocity_x = momentum * velocity_x - learning_rate * grad_x
        velocity_y = momentum * velocity_y - learning_rate * grad_y

        # Update step
        x += velocity_x
        y += velocity_y
        history.append((x, y))

        # Stop if gradient is small
        if np.sqrt(grad_x**2 + grad_y**2) < epsilon:
            break

    return history


# ------------------------------------------------------------------------------
# With noise
# ------------------------------------------------------------------------------
def gradient_descent_noise(  # noqa: PLR0913
    df_dx: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    x0: float,
    y0: float,
    learning_rate: float = 0.05,
    max_iter: int = 50,
    seed: int = 42,
    noise_step: int = 2,
    epsilon: float = 1e-5,
) -> list[tuple[float, float]]:
    """Compute a noisy gradient descent for a function `f` up to 2 variables. Gaussian noise N(0, 1).

    Args:
        df_dx (Callable[[float, float], float]): derivative w.r.t. `x`
        df_dy (Callable[[float, float], float]): derivative w.r.t. `y`
        x0 (float): init value for `x`
        y0 (float): init value for `y`
        learning_rate (float, optional): learning rate. Defaults to 0.05.
        max_iter (int, optional): Max number of iterations. Defaults to 50.
        seed (int, optional): Random seed. Defaults to 42.
        noise_step (int, optional): Number of gradient descent steps between "noisy step". Defaults to 2.
        epsilon (float, optional): Stopping threshold. Defaults to 1e-5.

    Returns:
        list[tuple[float, float]]: iterative values of `x` and `y` through the gradient descent

    """  # noqa: E501
    rng: np.random.Generator = np.random.default_rng(seed)  # Set the rnd seed

    x, y = x0, y0
    history: list[tuple[float, float]] = [(x, y)]  # Store descent path

    for i in range(max_iter):
        # Compute gradient
        grad_x: float = df_dx(x, y)
        grad_y: float = df_dy(x, y)

        # Noise step
        if i % noise_step == 0:
            x_noise, y_noise = rng.normal(loc=0, scale=2, size=2)
            grad_x = x_noise
            grad_y = y_noise

        # Update step
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        history.append((x, y))

        # Stop if gradient is small
        if np.sqrt(grad_x**2 + grad_y**2) < epsilon:
            break

    return history


# ==============================================================================
# Paper Figure 2
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
# ~ Derivatives
def df_dx(x: float, y: float) -> float:
    """Compute the derivative of `f(x, y)` w.r.t. `x`.

    Args:
        x (float): _description_
        y (float): _description_

    Returns:
        float: df(x, y)/dx

    """
    return np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) + 2 * x


def df_dy(x: float, y: float) -> float:
    """Compute the derivative of `f(x, y)` w.r.t. `y`.

    Args:
        x (float): _description_
        y (float): _description_

    Returns:
        float: df(x, y)/dy

    """
    return -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) + 2 * y


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
    # ~ Process history
    history_x: list[float] = [point[0] for point in history]
    history_y: list[float] = [point[1] for point in history]

    # ~ Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("f(x, y)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.plot(
        history_x,
        history_y,
        color="red",
        marker="o",
        linestyle="dashed",
        label="Descent path",
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
def main(outdir: str) -> None:
    """_summary_.

    Args:
        outdir (str): _description_

    """
    # --------------------------------------------------------------------------
    # Global figure settings
    # --------------------------------------------------------------------------
    # Plot function landscape and descent path
    x_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    y_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
    Z: np.typing.NDArray = f(X, Y)  # noqa: N806

    # Find the indices of the minimum value
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]
    min_value = Z[min_index]
    print(  # noqa: T201
        "Gradient descent: "
        f"Minimum value of f(x, y) is {min_value} "
        f"at coordinates ({min_x}, {min_y})",
    )

    # --------------------------------------------------------------------------
    # Figure 2a
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list[tuple[float, float]] = gradient_descent(
        df_dx,
        df_dy,
        1.5,
        1.5,
        max_iter=500,
    )

    # Plot
    plot(X, Y, Z, history, export=f"{outdir}/fig2a-gradient.pdf")

    # --------------------------------------------------------------------------
    # Figure 2b
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list[tuple[float, float]] = gradient_descent_momentum(
        df_dx,
        df_dy,
        1.5,
        1.5,
        max_iter=500,
        momentum=0.8,
    )

    # Plot
    plot(X, Y, Z, history, export=f"{outdir}/fig2b-gradient_momentum.pdf")

    # --------------------------------------------------------------------------
    # Figure 2c
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list[tuple[float, float]] = gradient_descent_noise(
        df_dx,
        df_dy,
        1.5,
        1.5,
        max_iter=500,
        seed=42,
        noise_step=2,
    )

    # Plot
    plot(X, Y, Z, history, export=f"{outdir}/fig2c-gradient_noise.pdf")


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
    )
