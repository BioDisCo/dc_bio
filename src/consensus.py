"""Basic showcase of consensus algorithms."""

# ==============================================================================
# IMPORTS
# ==============================================================================
from __future__ import annotations

import pathlib
import random
import sys
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


# ==============================================================================
# Consensus
# ==============================================================================
# ------------------------------------------------------------------------------
# Basic implementation
# ------------------------------------------------------------------------------
def consensus(
    func: Callable[[set, list[set]], set],
    sequence: list[nx.DiGraph],
    init_values_1D: dict[str, set[float | tuple[float, float]]],
    propagate_frequency: int | None = None,
) -> list[list[float | tuple[float, float]]]:
    """_summary_.

    Args:
        func (Callable[[set, list[set]], set]): _description_
        sequence (list[nx.DiGraph]): _description_
        init_values_1D (dict[str, set[float, tuple[float, float]]]): _description_
        propagate_frequency (int | None, optional): _description_. Defaults to None.

    Returns:
        list[list[float | tuple[float, float]]]: _description_

    """
    history: list[list[float | tuple[float, float]]] = []
    graph_state: dict[str, set[float | tuple[float, float]]] = init_values_1D
    outputs: list[list[float | tuple[float, float]]] = [
        list(graph_state[node])[0] for node in sorted(graph_state.keys())
    ]
    for round, graph_i in enumerate(sequence, start=1):
        history.append(outputs.copy())

        round_func: Callable = func
        update_output: bool = True
        if propagate_frequency is not None and propagate_frequency > 0:
            if round % propagate_frequency != 0:
                round_func = graph_flood
                update_output = False

        next_graph_state: dict[str, set[float | tuple[float, float]]] = {
            node: set(vals) for node, vals in graph_state.items()
        }
        for node in graph_i.nodes():
            neighbor_values: list[set[float | tuple[float, float]]] = [
                graph_state[neighbor] for neighbor in sorted(graph_i.predecessors(node))
            ]
            next_graph_state[node] = round_func(
                graph_state[node],
                neighbor_values,
            ).copy()
        graph_state = next_graph_state

        if update_output:
            outputs = [list(graph_state[node])[0] for node in sorted(graph_state.keys())]

    history.append(outputs)
    return history


# ------------------------------------------------------------------------------
# Communication functions
# ------------------------------------------------------------------------------
def graph_flood(own: set, in_values: list[set]) -> set:
    """_summary_.

    Args:
        own (set): _description_
        in_values (list[set]): _description_

    Returns:
        set: _description_

    """
    ret = own
    for s in in_values:
        ret = ret | s
    return ret


def graph_mean(own: set, in_values: list[set]) -> set:
    """_summary_.

    Args:
        own (set): _description_
        in_values (list[set]): _description_

    Returns:
        set: _description_

    """
    ret = own
    for s in in_values:
        ret = ret | s
    return {sum(ret) / len(ret)}


def graph_midpoint(own: set, in_values: list[set]) -> set:
    """_summary_.

    Args:
        own (set): _description_
        in_values (list[set]): _description_

    Returns:
        set: _description_

    """
    ret = own
    for s in in_values:
        ret = ret | s
    return {(max(ret) + min(ret)) / 2.0}


def graph_midextremes(
    own: set[tuple[float]],
    in_values: list[set[tuple[float]]],
) -> set[tuple[float]]:
    """_summary_.

    Args:
        own (set[tuple[float]]): _description_
        in_values (list[set[tuple[float]]]): _description_

    Returns:
        set[tuple[float]]: _description_

    """
    ret = own
    for s in in_values:
        ret = ret | s
    points: list[tuple] = list(ret)

    # Compute pairwise distances
    distances = squareform(pdist(points))

    # Find indices of the two farthest points
    idx = np.unravel_index(np.argmax(distances), distances.shape)

    # Get the two farthest points
    point1, point2 = np.array(points[idx[0]]), np.array(points[idx[1]])

    # Compute the midpoint
    midpoint: np.ndarray = (point1 + point2) / 2.0

    return {tuple(midpoint.tolist())}


def graph_approachextreme(own: set, in_values: list[set]) -> set:
    """_summary_.

    Args:
        own (set): _description_
        in_values (list[set]): _description_

    Returns:
        set: _description_

    """
    ret = own
    for s in in_values:
        ret = ret | s
    points = np.array(list(ret))
    own_point: np.ndarray = np.array(list(own)[0])

    # Compute distances from x to all points
    distances = np.linalg.norm(points - own_point, axis=1)

    # Find the index of the farthest point
    idx = np.argmax(distances)

    # Get the farthest point
    farthest_point: np.ndarray = np.array(points[idx])

    # Compute the midpoint
    midpoint: np.ndarray = (own_point + farthest_point) / 2.0

    return {tuple(midpoint.tolist())}


# ==============================================================================
# Paper Figure 5
# ==============================================================================
# ------------------------------------------------------------------------------
# Graph generation
# ------------------------------------------------------------------------------
def add_random_edges(G: nx.DiGraph, num_edges: int = 10) -> nx.DiGraph:
    """Adds random edges to an DiGraph.

    Parameters
    ----------
    G (nx.DiGraph): The directed graph.
    num_edges (int): Number of random edges to add.

    Returns:
    -------
    nx.DiGraph: The graph with added edges.

    """
    nodes = list(G.nodes)
    nodes.sort()
    added_edges: int = 0

    while added_edges < num_edges:
        source, target = random.sample(nodes, k=2)
        if source != target and not G.has_edge(source, target):
            G.add_edge(source, target)
            added_edges += 1

    return G


def generate_trees(num_nodes: int, num: int) -> list[nx.DiGraph]:
    """_summary_.

    Args:
        num_nodes (int): _description_
        num (int): _description_

    Returns:
        list[nx.DiGraph]: _description_

    """
    assert num <= num_nodes, "at most one per root at the moment"
    trees: list[nx.DiGraph] = []
    nodes = list(range(num_nodes))
    for root in nodes:
        # Create a complete graph with random weights
        complete_graph = nx.complete_graph(num_nodes)
        for u, v in complete_graph.edges():
            complete_graph.edges[u, v]["weight"] = random.random()

        # Generate a random spanning tree using minimum spanning tree
        tree = nx.minimum_spanning_tree(complete_graph)

        # Relabel nodes to ensure the desired root
        mapping = {list(tree.nodes())[0]: root}
        for node in tree.nodes():
            if node not in mapping:
                mapping[node] = [n for n in nodes if n not in mapping.values()][0]
        tree = nx.relabel_nodes(tree, mapping)

        # Convert to directed tree rooted at 'root'
        directed_tree = nx.DiGraph()
        for parent, child in nx.bfs_edges(tree, source=root):
            directed_tree.add_edge(parent, child)

        trees.append(directed_tree)

        # done
        if len(trees) >= num:
            break
    return trees


def generate_graphs(
    num_nodes: int,
    num: int,
    seed: int = 42,
) -> list[nx.DiGraph]:
    """_summary_.

    Args:
        num_nodes (int): _description_
        num (int): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        list[nx.DiGraph]: _description_

    """
    random.seed(seed)
    trees: list[nx.DiGraph] = generate_trees(num_nodes, num=num)
    graphs: list[nx.DiGraph] = []
    for tree in trees:
        # Add edges and plot and save each graph to PDF
        graph: nx.DiGraph = add_random_edges(
            tree,
            num_edges=int(np.log(num_nodes)) * num_nodes,
        )
        graphs.append(graph)
    return graphs


def generate_graphs_sequence(
    graphs: list[nx.DiGraph],
    num: int,
    seed: int = 42,
) -> list[nx.DiGraph]:
    """_summary_.

    Args:
        graphs (list[nx.DiGraph]): _description_
        num (int): _description_
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        list[nx.DiGraph]: _description_

    """
    random.seed(seed)
    graphs_sequence: list[nx.DiGraph] = []
    for _ in range(num):
        graph_round_id: int = random.sample(range(len(graphs)), k=1)[0]
        graph_round_i: nx.DiGraph = graphs[graph_round_id]
        graphs_sequence.append(graph_round_i)
    return graphs_sequence


def generate_butterfly_graph(num_nodes: int) -> nx.DiGraph:
    """Generate a buterfly graph.

    Args:
        num_nodes: the number of nodes, must by even and positive

    Returns:
        the butterfly graph with `num_nodes` nodes

    Raises:
        ValueError: if num_nodes is odd or nonpositive
    """
    if num_nodes % 2 != 0 or num_nodes <= 0:
        msg = "Number of nodes of a butterfly graph must be even and positive."
        raise ValueError(msg)

    m = num_nodes // 2
    G = nx.DiGraph()
    G.add_edge(m, m + m)
    G.add_edge(m + m, m)

    for i in range(1, m + 1):
        G.add_edge(i, i)
        G.add_edge(1, i)
        G.add_edge(i, ((i - 2) % m) + 1)

        G.add_edge(i + m, i + m)
        G.add_edge(1 + m, i + m)
        G.add_edge(i + m, ((i - 2) % m) + 1 + m)

    return G


# ------------------------------------------------------------------------------
# ~ Plotting
def export_graph(
    filename: str,
    graph: nx.DiGraph,
    pos: dict | None = None,
    seed: int | None = None,
) -> dict:
    """_summary_.

    Args:
        graph (nx.DiGraph): _description_
        filename (str): _description_
        pos (dict | None, optional): _description_. Defaults to None.
        seed (dict | None, optional): _description_. Defaults to None.

    Returns:
        dict: _description_

    """
    plt.figure(figsize=(3, 3))
    if pos is None:
        pos = nx.spring_layout(graph, k=2, seed=seed)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
    )
    plt.savefig(filename, format="pdf")
    plt.close()
    return pos


def plot_trace_1D(fname: str, history: list[list[float]]) -> None:
    """_summary_.

    Args:
        fname (str): _description_
        history (list): _description_

    """
    plt.figure(figsize=(8, 3))
    plt.plot(
        range(len(history)),
        history,
        "-",
        color="gray",
        marker="o",
        alpha=0.6,
    )
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Round", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.savefig(fname, bbox_inches="tight", transparent=True, pad_inches=0.01)


def plot_trace_2D(fname: str, history: list[list[float]]) -> None:
    """_summary_.

    Args:
        fname (str): _description_
        history (list): _description_

    """
    plt.figure(figsize=(4, 4))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for pn in range(len(history[0])):
        # plot history
        history_x = [history[i][pn][0] for i in range(len(history))]
        history_y = [history[i][pn][1] for i in range(len(history))]
        plt.plot(
            history_x,
            history_y,
            color="gray",
            marker="o",
            linestyle="dashed",
        )
    for pn in range(len(history[0])):
        # plot initial and final values over the rest
        history_x = [history[i][pn][0] for i in range(len(history))]
        history_y = [history[i][pn][1] for i in range(len(history))]
        plt.plot(
            history_x[0:1],
            history_y[0:1],
            color="red",
            marker="o",
            linestyle="dashed",
        )
        plt.plot(
            history_x[-2:-1],
            history_y[-2:-1],
            color="blue",
            marker="o",
            linestyle="dashed",
        )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.savefig(fname, bbox_inches="tight", transparent=True, pad_inches=0.01)
    plt.close()


def plot_rate_1D(fname: str, history: list[list[float]]) -> None:
    """_summary_.

    Args:
        fname (str): _description_
        history (list): _description_

    """
    rates: list[float] = [max(abs(i - j) for i in state for j in state) for state in history]
    plt.figure(figsize=(8, 3))
    plt.plot(
        range(len(history)),
        rates,
        "-",
        color="gray",
        marker="o",
    )
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Round", fontsize=16)
    plt.ylabel("Rate", fontsize=16)
    plt.savefig(fname, bbox_inches="tight", transparent=True, pad_inches=0.01)


def build_psi_graph(n: int, i: int) -> nx.DiGraph:
    """Construct the Ψ_i worst-case graph as per Section 6 of the paper.

    > Matthias F¨ugger, Thomas Nowak, and Manfred Schwarz. Tight bounds for asymptotic and approximate consensus. Journal of the ACM (JACM), 68(6):1-35, 2021.

    Args:
        n (int): Number of agents (n >= 3).
        i (int): Index of Ψ graph to generate (0, 1, or 2).

    Returns:
        nx.DiGraph: The Ψ_i graph.

    """
    if not (0 <= i <= 2):  # noqa: PLR2004
        raise ValueError("i must be 0, 1, or 2")  # noqa: EM101, TRY003
    if n < 3:  # noqa: PLR2004
        raise ValueError("n must be at least 3")  # noqa: EM101, TRY003

    G: nx.DiGraph = nx.DiGraph()

    # Add all nodes
    G.add_nodes_from(range(1, n))

    # Path: nodes 3 → 4 → ... → n-1
    for j in range(3, n - 1):
        G.add_edge(j, j + 1)

    # Special case: i sends to 4
    G.add_edge(i, 3)

    # Remaining of {0,1,2} \ {i} → 3, with in-neighbor n
    for j in {0, 1, 2} - {i}:
        G.add_edge(n - 1, j)  # n → j
        G.add_edge(j, 3)  # j → 4

    return G


# ------------------------------------------------------------------------------
# ~ Main
def main(outdir: str, videodir: str) -> None:
    """_summary_.

    Args:
        outdir (str): _description_

    """
    # --------------------------------------------------------------------------
    # Global figure settings
    # --------------------------------------------------------------------------
    # ~ Graph generation
    num_nodes: int = 10
    tree_num: int = 3
    pos: dict | None = None

    # ~ Algorithm settings
    num_rounds: int = 10
    num_rounds_to_draw: int = 5

    # ~ Random
    rnd_seed: int = 8

    # --------------------------------------------------------------------------
    # ~ Generate communication graphs

    # ~ Init export directory
    graphs_dir: str = f"{outdir}/fig5a-graphs"
    pathlib.Path(graphs_dir).mkdir(parents=True, exist_ok=True)

    # ~ Build random communication graphs
    graphs: list[nx.DiGraph] = generate_graphs(num_nodes, tree_num, seed=rnd_seed)
    for i, G in enumerate(graphs):
        pos: dict = export_graph(
            f"{graphs_dir}/graph_{i}.pdf",
            G,
            pos,
            seed=rnd_seed,
        )

    # --------------------------------------------------------------------------
    # ~ Generate the communication graph sequence

    # ~ Init export directory
    rounds_dir: str = f"{outdir}/fig5a-rounds"
    pathlib.Path(rounds_dir).mkdir(parents=True, exist_ok=True)

    # ~ Generate a random communication graph sequence
    graphs_sequence: list[nx.DiGraph] = generate_graphs_sequence(
        graphs,
        num_rounds,
        seed=rnd_seed,
    )
    for i in range(min(num_rounds, num_rounds_to_draw)):
        Gi: nx.DiGraph = graphs_sequence[i]
        if i < num_rounds_to_draw:
            pos: dict = export_graph(
                f"{rounds_dir}/graph_round_{i}.pdf",
                Gi,
                pos,
                seed=rnd_seed,
            )

    # --------------------------------------------------------------------------
    # ~ Initial values

    random.seed(rnd_seed * 11)
    # ~ 1D
    init_values_1D: dict[str, set[float]] = {node: {random.uniform(0, 1)} for node in graphs_sequence[0].nodes()}

    random.seed(rnd_seed * 42)
    # ~ 2D
    init_values_2D: dict[str, set[tuple[float, float]]] = {
        node: {(random.uniform(0, 1), random.uniform(0, 1))} for node in graphs_sequence[0].nodes()
    }

    # --------------------------------------------------------------------------
    # Figure 5[b/c] - Equal-weight algorithm
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 5b - without flooding
    history_1D: list[list[float]] = consensus(
        graph_mean,
        graphs_sequence,
        init_values_1D,
    )
    plot_trace_1D(
        f"{outdir}/fig5b-consensus_mean.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 5b - with flooding
    history_1D: list[list[float]] = consensus(
        graph_mean,
        graphs_sequence,
        init_values_1D,
        propagate_frequency=2,
    )
    plot_trace_1D(
        f"{outdir}/fig5c-consensus_mean_propagate.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # Figure 5[d/e] - Midpoint algorithm
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 5c - without flooding
    history_1D: list[list[float]] = consensus(
        graph_midpoint,
        graphs_sequence,
        init_values_1D,
    )
    plot_trace_1D(
        f"{outdir}/fig5d-consensus_midpoint.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 5e - with flooding
    history_1D: list[list[float]] = consensus(
        graph_midpoint,
        graphs_sequence,
        init_values_1D,
        propagate_frequency=2,
    )
    plot_trace_1D(
        f"{outdir}/fig5e-consensus_midpoint_propagate.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 5f - equal-weight butterfly

    butterfly_graph = generate_butterfly_graph(num_nodes=10)
    butterfly_sequence = generate_graphs_sequence(graphs=[butterfly_graph], num=20)
    butterfly_init_values_1D = {node: {0 if node <= num_nodes // 2 else 1} for node in butterfly_graph.nodes()}
    history_1D: list[list[float]] = consensus(
        graph_mean,
        sequence=butterfly_sequence,
        init_values_1D=butterfly_init_values_1D,
        propagate_frequency=None,
    )
    plot_trace_1D(
        f"{outdir}/fig5f-consensus_butterfly_mean.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # ~ Fig. 5g - amortized midpoint butterfly

    history_1D: list[list[float]] = consensus(
        graph_midpoint,
        sequence=butterfly_sequence,
        init_values_1D=butterfly_init_values_1D,
        propagate_frequency=9,
    )
    plot_trace_1D(
        f"{outdir}/fig5g-consensus_butterfly_amortized_midpoint.pdf",
        history_1D,
    )

    # --------------------------------------------------------------------------
    # Figure 6[a/b] - MidExtremes algorithm
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 6a - without flooding
    history_2D: list[list[float]] = consensus(
        graph_midextremes,
        graphs_sequence,
        init_values_2D,
    )
    plot_trace_2D(
        f"{outdir}/fig6a-consensus_midextremes.pdf",
        history_2D,
    )

    # --------------------------------------------------------------------------
    # ~ Plot for videos
    roundir: str = f"{videodir}/fig6a-midextremes/"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(0, len(history_2D), 1)):
        plot_trace_2D(
            f"{roundir}/round_{j:03d}.png",
            history_2D[: i + 1],
        )

    # --------------------------------------------------------------------------
    # ~ Fig. 6b - with flooding
    history_2D: list[list[float]] = consensus(
        graph_midextremes,
        graphs_sequence,
        init_values_2D,
        propagate_frequency=2,
    )
    plot_trace_2D(
        f"{outdir}/fig6b-consensus_midextremes_propagate.pdf",
        history_2D,
    )

    # --------------------------------------------------------------------------
    # ~ Plot for videos
    roundir: str = f"{videodir}/fig6b-midextremes_propagate/"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(0, len(history_2D), 1)):
        plot_trace_2D(
            f"{roundir}/round_{j:03d}.png",
            history_2D[: i + 1],
        )

    # --------------------------------------------------------------------------
    # Figure 6[c/d] - ApproachExtreme algorithm
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # ~ Fig. 6c - without flooding
    history_2D: list[list[float]] = consensus(
        graph_approachextreme,
        graphs_sequence,
        init_values_2D,
    )
    plot_trace_2D(
        f"{outdir}/fig6c-consensus_approachextreme.pdf",
        history_2D,
    )

    # --------------------------------------------------------------------------
    # ~ Plot for videos
    roundir: str = f"{videodir}/fig6c-approachextreme/"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(0, len(history_2D), 1)):
        plot_trace_2D(
            f"{roundir}/round_{j:03d}.png",
            history_2D[: i + 1],
        )

    # --------------------------------------------------------------------------
    # ~ Fig. 6d - with flooding
    history_2D: list[list[float]] = consensus(
        graph_approachextreme,
        graphs_sequence,
        init_values_2D,
        propagate_frequency=2,
    )
    plot_trace_2D(
        f"{outdir}/fig6d-consensus_approachextreme_propagate.pdf",
        history_2D,
    )

    # --------------------------------------------------------------------------
    # ~ Plot for videos
    roundir: str = f"{videodir}/fig6d-approachextreme_propagate/"
    pathlib.Path(roundir).mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(range(0, len(history_2D), 1)):
        plot_trace_2D(
            f"{roundir}/round_{j:03d}.png",
            history_2D[: i + 1],
        )


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[2]}",
    )
