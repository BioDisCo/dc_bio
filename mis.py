import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable
from networkx.classes.reportviews import NodeView

np.random.seed(42)


def run_alg(
    graph: nx.Graph,
    phases: int,
    rounds_per_phase: int,
    f: Callable[[int, int, set, list[set], int], set],
    D: int
) -> dict[NodeView, set]:
    """
    execution of f on graph
    """
    node_values = {node: {0} for node in graph.nodes()}
    for phase in range(phases):
        for round_nr in range(rounds_per_phase):
            new_values = {node: set(vals) for node, vals in node_values.items()}
            for node in graph.nodes():
                neighbor_values = [
                    node_values[neighbor] for neighbor in graph.neighbors(node)
                ]
                computation = f(phase, round_nr, node_values[node], neighbor_values, D)
                new_values[node] = computation.copy()
            node_values = new_values
            plot_graph(G, node_values, f"mis-{phase*rounds_per_phase + round_nr:03d}.png")
    return node_values


def graph_step(phase: int, round_nr: int, own: set, in_values: list[set], D: int) -> set:
    """
    phase: 'i' in https://www.science.org/doi/pdf/10.1126/science.1193210
    """
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
        p = 1 / 2**(np.log(D) - phase)
        return { int(np.random.rand() < p) }
    
    else:
        # check round
        if 1 in own:
            if 1 in others:
                # do not join
                return { 0 }
            else:
                # join
                return {"Term-MIS"}
    
    return set()


def plot_graph(G: nx.Graph, node_values: dict[NodeView, set], fname: str) -> None:
    node_colors = []
    for i, node in enumerate(G.nodes()):
        if "Term-MIS" in node_values[node]:
            node_colors.append('blue')
        elif "Term-nonMIS" in node_values[node]:
            node_colors.append('white')
        else:
            node_colors.append('gray')

    # draw
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='black')
    plt.axis('off')
    plt.savefig(fname, bbox_inches="tight", transparent=True, pad_inches=0.01, dpi=300)


if __name__ == "__main__":
    G = nx.hexagonal_lattice_graph(4, 4, periodic=False, with_positions=True)

    max_degree = max(dict(G.degree()).values())
    print("Maximum degree in G is", max_degree)

    values = run_alg(G, phases=int(np.log(max_degree)), rounds_per_phase=20, f=graph_step, D=max_degree)
    plot_graph(G, values, "mis-final.pdf")
