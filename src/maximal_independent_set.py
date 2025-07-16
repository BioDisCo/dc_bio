"""Basic showcase of for MIS algorithm."""  # noqa: E501

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import pathlib
import sys
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.classes.reportviews import NodeView


# ==============================================================================
# Maximal Independent Sets
# ==============================================================================
# ------------------------------------------------------------------------------
# Basic implementation
# ------------------------------------------------------------------------------
def maximal_independent_set(
    graph: nx.Graph,
    phases: int,
    rounds_per_phase: int,
    step_func: Callable[[int, int, set, list[set], int], set],
    neighbors_ub: int,
    seed: int = 42,
) -> tuple[dict[NodeView, set[str]], list[dict[NodeView, set[str]]]]:
    """Compute the graph MIS.

    Args:
        graph (nx.Graph): _description_
        phases (int): _description_
        rounds_per_phase (int): _description_
        step_func (Callable[[int, int, set, list[set], int], set]): _description_
        neighbors_ub (int): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        tuple[dict[NodeView, set[str]], list[dict[NodeView, set[str]]]]: MIS state for each node, node state history

    """
    rnd_gen: np.random.Generator = np.random.default_rng(seed)
    node_values: dict[str, set[float]] = {node: {0} for node in graph.nodes()}
    history: list[dict[str, set[float]]] = []
    for phase in range(phases):
        for round_nr in range(rounds_per_phase):
            new_values = {node: set(vals) for node, vals in node_values.items()}
            for node in graph.nodes():
                neighbor_values = [
                    node_values[neighbor] for neighbor in graph.neighbors(node)
                ]
                computation = step_func(
                    phase,
                    round_nr,
                    node_values[node],
                    neighbor_values,
                    neighbors_ub,
                    rnd_gen=rnd_gen,
                )
                new_values[node] = computation.copy()
            node_values = new_values
            history.append(node_values)
    return node_values, history


def graph_step(
    phase: int,
    round_nr: int,
    own: set,
    in_values: list[set],
    neighbors_ub: int,
    rnd_gen: np.random.Generator | None = None,
) -> set[str]:
    """Phase: 'i' in https://www.science.org/doi/pdf/10.1126/science.1193210.

    Args:
        phase (int): _description_
        round_nr (int): _description_
        own (set): _description_
        in_values (list[set]): _description_
        neighbors_ub (int): _description_
        rnd_gen (np.random.Generator | None, optional): _description_. Defaults to None.

    Returns:
        set[str]: _description_

    """
    if rnd_gen is None:
        rnd_gen = np.random.default_rng(42)

    if "Term-MIS" in own:
        # already in the MIS -> remain in there
        return {"Term-MIS"}

    others = set()
    for s in in_values:
        others = others | s

    if "Term-nonMIS" in own or "Term-MIS" in others:
        # not in MIS -> remain
        return {"Term-nonMIS"}

    # else
    if round_nr % 2 == 0:
        # proposition round
        p = 1 / 2 ** (np.log(neighbors_ub) - phase)
        return {int(np.random.rand() < p)}

    else:
        # check round
        if 1 in own:
            if 1 in others:
                # do not join
                return {0}
            else:
                # join
                return {"Term-MIS"}

    return set()


# ==============================================================================
# Plot
# ==============================================================================
def plot_graph(
    fname: str,
    G: nx.Graph,
    node_values: dict[NodeView, set[str]],
) -> None:
    node_colors = []
    for node in G.nodes():
        if "Term-MIS" in node_values[node]:
            node_colors.append("lightblue")
        elif "Term-nonMIS" in node_values[node]:
            node_colors.append("white")
        else:
            node_colors.append("gray")

    # draw
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, edgecolors="black", node_size=500
    )
    nx.draw_networkx_edges(G, pos, edge_color="black")
    plt.axis("off")
    plt.savefig(
        fname, bbox_inches="tight", transparent=True, pad_inches=0.01, dpi=300
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
    rnd_seed: int = 42
    np.random.seed(42)

    # ~ Graph descriptions
    graph_width: int = 4
    graph_height: int = 4

    # --------------------------------------------------------------------------
    # Compute MIS
    # --------------------------------------------------------------------------
    G: nx.Graph = nx.hexagonal_lattice_graph(
        graph_width, graph_height, periodic=False, with_positions=True
    )

    max_degree: int = max(dict(G.degree()).values())
    mis_results, history = maximal_independent_set(
        G,
        phases=int(
            np.log(max_degree)
        ),  # See https://www.science.org/doi/pdf/10.1126/science.1193210
        rounds_per_phase=20,
        step_func=graph_step,
        neighbors_ub=max_degree,
        seed=rnd_seed,
    )

    # --------------------------------------------------------------------------
    # Figure 8b + intermediary
    # --------------------------------------------------------------------------
    roundir: str = f"{outdir}/fig8-mis-rounds"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for i, mis_state in enumerate(history):
        plot_graph(f"{roundir}/fig8-mis_round_{i:03d}.png", G, mis_state)

    # --------------------------------------------------------------------------
    # Figure 8c
    # --------------------------------------------------------------------------
    plot_graph(f"{outdir}/fig8c-mis_final.pdf", G, mis_results)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
    )
