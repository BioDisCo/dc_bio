"""Basic showcase of CRN simulations."""  # noqa: E501

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import pathlib
import random
import sys
from typing import Iterable, Literal

import matplotlib.pyplot as plt
from mobspy import BaseSpecies, Simulation, Zero


# ==============================================================================
# Paper Figure 7: Chemical Reaction Network
# ==============================================================================
# ------------------------------------------------------------------------------
# ABC Model
# ------------------------------------------------------------------------------
def simulate_abc(
    init_values: tuple[float, float],
    rates: tuple[float, float],
    duration: float,
    repeats: int = 1,
    seed: int | None = 42,
    simulation_method: Literal["stochastic", "deterministic"] = "deterministic",
) -> Iterable[dict[str, list[float]]]:
    """_summary_.

    Args:
        init_values (tuple[float, float]): _description_
        rates (tuple[float, float]): _description_
        duration (float): _description_
        repeats (int, optional): _description_. Defaults to 1.
        seed (int | None, optional): _description_. Defaults to 42.
        simulation_method (Literal[&quot;stochastic&quot;, &quot;deterministic&quot;], optional): _description_. Defaults to "deterministic".

    Returns:
        Iterable[dict[str, list[float]]]: _description_

    """
    random.seed(seed)

    A, B, C = BaseSpecies()
    A(init_values[0]) + B(init_values[1]) >> A + C[rates[0]]
    A >> Zero[rates[1]]

    MySim = Simulation(A | B | C)

    # Deterministics
    MySim.run(
        simulation_method=simulation_method,
        duration=duration,
        save_data=False,
        repetitions=repeats,
        plot_data=False,
        seeds=(None if seed is None else [i * seed for i in range(repeats)]),
    )

    return MySim.results


# ------------------------------------------------------------------------------
# Mutual annihilation Model
# ------------------------------------------------------------------------------
def simulate_mutual_annihilation(
    init_values: tuple[float, float, float],
    rates: tuple[float, float, float, float],
    duration: float,
    repeats: int = 1,
    seed: int | None = 42,
    simulation_method: Literal["stochastic", "deterministic"] = "deterministic",
) -> Iterable[dict[str, list[float]]]:
    """_summary_.

    Args:
        init_values (tuple[float, float, float]): _description_
        rates (tuple[float, float, float, float]): _description_
        duration (float): _description_
        repeats (int, optional): _description_. Defaults to 1.
        seed (int | None, optional): _description_. Defaults to 42.
        simulation_method (Literal[&quot;stochastic&quot;, &quot;deterministic&quot;], optional): _description_. Defaults to "deterministic".

    Returns:
        Iterable[dict[str, list[float]]]: _description_

    """
    random.seed(seed)

    A, B, R = BaseSpecies()
    A(init_values[0]) + B(init_values[1]) >> A[rates[0]]
    A + B >> B[rates[1]]
    A + R(init_values[2]) >> A + A[rates[2]]
    B + R >> B + B[rates[3]]

    MySim = Simulation(A | B | R)

    # stochastic
    MySim.run(
        simulation_method=simulation_method,
        duration=duration,
        save_data=False,
        repetitions=repeats,
        plot_data=False,
        seeds=(None if seed is None else [i * seed for i in range(repeats)]),
    )

    return MySim.results


# ==============================================================================
# Plot
# ==============================================================================
def plot_abc(
    outfile: str,
    sim: Iterable[dict[str, list[float]]],
    mode: Literal["deterministic", "stochastic"],
) -> None:
    """_summary_.

    Args:
        outfile (str): _description_
        sim (Iterable[dict[str, list[float]]]): _description_
        mode (Literal[&quot;deterministic&quot;, &quot;stochastic&quot;]): _description_

    """
    plt.figure(figsize=(4, 4))
    linewidths: tuple[float, float] = (
        (1.5, 1) if mode == "deterministic" else (2.5, 2)
    )
    for i, res in enumerate(sim):
        plt.plot(
            res["Time"],
            res["A"],
            label="A" if i == 0 else "_",
            color="red",
            alpha=1 if i == 0 else 0.3,
            linewidth=linewidths[0] if i == 0 else linewidths[1],
        )
        plt.plot(
            res["Time"],
            res["B"],
            label="B" if i == 0 else "_",
            color="blue",
            alpha=1 if i == 0 else 0.3,
            linewidth=linewidths[0] if i == 0 else linewidths[1],
        )
        plt.plot(
            res["Time"],
            res["C"],
            label="C" if i == 0 else "_",
            color="black",
            alpha=1 if i == 0 else 0.3,
            linewidth=linewidths[0] if i == 0 else linewidths[1],
        )
    plt.legend(frameon=False, loc=(0.8, 0.7))
    plt.ylabel("count/volume" if mode == "deterministic" else "count")
    plt.xlabel("time")
    plt.savefig(outfile, bbox_inches="tight", transparent=True, pad_inches=0.01)


def plot_mutual_annihilation(
    outfile: str,
    sim: Iterable[dict[str, list[float]]],
    mode: Literal["deterministic", "stochastic"],
) -> None:
    """_summary_.

    Args:
        outfile (str): _description_
        sim (Iterable[dict[str, list[float]]]): _description_
        mode (Literal[&quot;deterministic&quot;, &quot;stochastic&quot;]): _description_

    """
    plt.figure(figsize=(4, 4))
    for i, res in enumerate(sim):
        plt.plot(
            res["Time"],
            res["A"],
            label="A" if i == 0 else "_",
            color="red",
            alpha=1 if i == 0 else 0.3,
            linewidth=1.5 if i == 0 else 1,
        )
        plt.plot(
            res["Time"],
            res["B"],
            label="B" if i == 0 else "_",
            color="blue",
            alpha=1 if i == 0 else 0.3,
            linewidth=1.5 if i == 0 else 1,
        )
    plt.gca().set_yscale("log")
    plt.ylim([0.1, 100])
    plt.legend(frameon=False, loc=(0.8, 0.7))
    plt.ylabel("count/volume" if mode == "deterministic" else "count")
    plt.xlabel("time")
    plt.savefig(
        outfile,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.01,
    )


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
    # ~ Random
    rnd_seed: int = 142

    # ~ Simulation settings
    duration: float = 10.0
    stochastic_repeats: int = 10

    # --------------------------------------------------------------------------
    # ~ Initial values

    # ~ ABC
    init_values_abc: tuple[float, float] = (10, 10)
    rates_abc: tuple[float, float] = (0.1, 0.1)

    # ~ Mutual Annihilation
    init_values_mutual_annihilation: tuple[float, float] = (12, 8, 100)
    rates_mutual_annihilation: tuple[float, float] = (0.1, 0.1, 0.01, 0.01)

    # --------------------------------------------------------------------------
    # Figure 7[a/b] - ABC model
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 7a - deterministic
    sim_abc: list[dict[str, list[float]]] = simulate_abc(
        init_values_abc,
        rates_abc,
        duration,
        repeats=1,
        seed=None,
        simulation_method="deterministic",
    )
    plot_abc(
        f"{outdir}/fig7a-crn-abc-deterministic.pdf",
        sim_abc,
        "deterministic",
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 7b - stochastic
    sim_abc: list[dict[str, list[float]]] = simulate_abc(
        init_values_abc,
        rates_abc,
        duration,
        repeats=stochastic_repeats,
        seed=rnd_seed,
        simulation_method="stochastic",
    )
    plot_abc(
        f"{outdir}/fig7b-crn-abc-stochastic.pdf",
        sim_abc,
        "stochastic",
    )

    # --------------------------------------------------------------------------
    # Figure 7[c/d] - Mutual Annihilation model
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 7c - deterministic
    sim_mutual_annihilation: list[dict[str, list[float]]] = (
        simulate_mutual_annihilation(
            init_values_mutual_annihilation,
            rates_mutual_annihilation,
            duration,
            repeats=1,
            seed=None,
            simulation_method="deterministic",
        )
    )

    plot_mutual_annihilation(
        f"{outdir}/fig7c-crn-mutual_annihilation-deterministic.pdf",
        sim_mutual_annihilation,
        "deterministic",
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 7d - stochastic
    sim_mutual_annihilation: list[dict[str, list[float]]] = (
        simulate_mutual_annihilation(
            init_values_mutual_annihilation,
            rates_mutual_annihilation,
            duration,
            repeats=stochastic_repeats,
            seed=rnd_seed,
            simulation_method="stochastic",
        )
    )
    plot_mutual_annihilation(
        f"{outdir}/fig7d-crn-mutual_annihilation-stochastic.pdf",
        sim_mutual_annihilation,
        "stochastic",
    )


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
    )
