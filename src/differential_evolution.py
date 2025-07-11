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
from scipy.optimize import differential_evolution as scipy_diff_evolution


# ==============================================================================
# Differential evolution
# ==============================================================================
# ------------------------------------------------------------------------------
# Basic usage
# ------------------------------------------------------------------------------
def differential_evolution(  # noqa: PLR0913
    f: Callable[[float, float], float],
    bounds: list[tuple[float, float], tuple[float, float]],
    strategy: Callable,
    popsize: int = 5,
    mutation: float = 0.7,
    recombination: float = 0.9,
    seed: int = 42,
    max_iter: int = 50,
    updates: None | list = None,
) -> list[np.typing.NDArray]:
    """_summary_.

    Args:
        f (Callable[[float, float], float]): Function to minimize
        bounds (list[tuple[float, float], tuple[float, float]]): Variables bounds
        strategy (Callable): Differential evolution strategy to use.
        popsize (int, optional): Initial population size. Defaults to 5.
        mutation: (float, optional): Mutation constant. Defaults to 0.7,
        recombination: (float): Recombination constant. Defaults to 0.9,
        seed (int, optional): Random seed. Defaults to 42.
        max_iter (int, optional): Max number of iterations. Defaults to 50.
        updates (int, optional): Keep track of update history, None to not keep it. Defaults to None.

    Returns:
        list[np.typing.NDArray]: iterative values of `x` and `y` through the gradient descent

    """  # noqa: E501
    rng: np.random.Generator = np.random.default_rng(seed)
    history: list[np.typing.NDArray] = []
    # Function results are not of interest for us as we keep track of
    # population history
    scipy_diff_evolution(
        (lambda args: f(*args)),
        bounds,
        strategy=(
            lambda *args, **kwargs: strategy(
                *args,
                **kwargs,
                history=history,
                updates=updates,
                mutation=mutation,
                recombination=recombination,
            )
        ),
        init="random",
        popsize=popsize,
        rng=rng,
        maxiter=max_iter,
        polish=False,
    )
    return history


# ------------------------------------------------------------------------------
# Custom strategy for keeping track of history
# ------------------------------------------------------------------------------
def custom_strategy_fn(  # noqa: ANN201, PLR0913
    candidate: int,
    population: np.typing.NDArray,
    rng: np.random.Generator = None,
    history: list | None = None,
    updates: list | None = None,
    mutation: float = 0.7,
    recombination: float = 0.9,
):
    """_summary_.

    Args:
        candidate (int): _description_
        population (np.typing.NDArray): _description_
        rng (np.random.Generator, optional): _description_. Defaults to None.
        history (list | None, optional): _description_. Defaults to None.
        updates (list | None, optional): _description_. Defaults to None.
        mutation (float, optional): _description_. Defaults to 0.7.
        recombination (float, optional): _description_. Defaults to 0.9.

    Returns:
        _type_: _description_

    """
    # Set default variable
    if rng is None:
        rng = np.random.default_rng()

    # Keep track of the history, i.e., population state
    if candidate == 0 and history is not None:
        history.append(population.transpose())
    if candidate == 0 and updates is not None:
        updates.append([])

    parameter_count = population.shape[-1]

    # evolve the candidate
    trial = np.copy(population[candidate])

    # choose a parameter dimension that will be always replaced
    fill_point = rng.choice(parameter_count)

    # random order of the population
    pool = np.arange(len(population))
    rng.shuffle(pool)

    # two unique random numbers that aren't the same, and
    # aren't equal to candidate.
    idxs = []
    while len(idxs) < 3 and len(pool) > 0:  # noqa: PLR2004
        # pop from head
        idx = pool[0]
        pool = pool[1:]
        if idx != candidate:
            idxs.append(idx)
    sa, sb, sc = idxs[:3]

    bprime = population[sa] + mutation * (population[sb] - population[sc])

    # for each parameter pick a uniform rnd number
    crossovers = rng.uniform(size=parameter_count)

    # check if this rnd value is < the recombination constant
    crossovers = crossovers < recombination

    # also one of them is the fill_point parameter that is always replaced
    # -> set it also to True
    crossovers[fill_point] = True

    # update the trial
    trial = np.copy(np.where(crossovers, bprime, trial))

    if updates is not None:
        updates[-1].append(
            (
                len(updates),
                candidate,
                population[sa].copy(),
                population[sb].copy(),
                population[sc].copy(),
                trial.copy(),
            ),
        )
    return trial


# ==============================================================================
# Paper Figure 4[a-b]
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


def plot_step(  # noqa: PLR0913
    X: np.typing.NDArray,  # noqa: N803
    Y: np.typing.NDArray,  # noqa: N803
    Z: np.typing.NDArray,  # noqa: N803
    history: list[tuple[float, float]],
    update: tuple[
        int,
        int,
        np.typing.NDArray,
        np.typing.NDArray,
        np.typing.NDArray,
        np.typing.NDArray,
    ],
    display: bool = False,  # noqa: FBT001, FBT002
    export: None | str = None,
) -> None:
    """Plot the gradient descent with 1 candidate update step (mutation and recombination assumed 1).

    Args:
        X (np.typing.NDArray): X to evaluate
        Y (np.typing.NDArray): Y to evaluate
        Z (np.typing.NDArray): f(X, Y)
        history (list[tuple[float, float]]): Gradient descent history.
        update (tuple[int, int, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]): Update to draw.
        display (bool, optional): Show the figure. Defaults to False.
        export (None | str, optional): Path of the pdf file to generate, None to not export. Defaults to None.

    """  # noqa: E501
    # ~ Process history
    history_x: list[float] = [point[0] for point in history]
    history_y: list[float] = [point[1] for point in history]

    # ~ Process update
    epoch, candidate, sa, sb, sc, trial = update

    # ~ Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label("f(x, y)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.plot(
        [sa[0]],
        [sa[1]],
        color="black",
        marker="o",
        markersize=20,
    )

    for pn in range(len(history_x[0])):
        p_x = [history_x[t][pn] for t in range(len(history))]
        p_y = [history_y[t][pn] for t in range(len(history))]
        col = "#ffcccc" if pn != candidate else "red"

        plt.plot(
            p_x,
            p_y,
            color=col,
            marker="o",
            linestyle="dashed",
            label="Evolution trajectory",
            zorder=2 if pn != candidate else 3,
        )

    plt.arrow(
        sc[0],
        sc[1],
        (sb[0] - sc[0]),
        (sb[1] - sc[1]),
        color="blue",
        length_includes_head=True,
        head_width=0.2,
        head_length=0.2,
        zorder=10,
    )

    plt.plot(
        [trial[0].copy()],
        [trial[1].copy()],
        color="red",
        marker="x",
        markersize=24,
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
        "Differential evolution: "
        f"Minimum value of f(x, y) is {min_value} "
        f"at coordinates ({min_x}, {min_y})",
    )

    # --------------------------------------------------------------------------
    # Figure 4a
    # --------------------------------------------------------------------------
    # Run gradient descent
    updates: list = []
    history: list = differential_evolution(
        f,
        bounds,
        custom_strategy_fn,
        popsize=5,
        mutation=1,
        recombination=1,
        seed=37,
        max_iter=2,
        updates=updates,
    )

    # Plot
    plot_step(
        X,
        Y,
        Z,
        history,
        updates[0][3],
        export=f"{outdir}/fig4a-differential_evolution_step.pdf",
    )

    # --------------------------------------------------------------------------
    # Figure 4b
    # --------------------------------------------------------------------------
    # Run gradient descent
    history: list = differential_evolution(
        f,
        bounds,
        custom_strategy_fn,
        popsize=5,
        mutation=0.7,
        recombination=0.9,
        seed=37,
        max_iter=10,
        updates=None,
    )

    # Plot
    plot(X, Y, Z, history, export=f"{outdir}/fig4b-differential_evolution.pdf")


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
    )
