import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Any, Callable
from networkx.classes.reportviews import NodeView

random.seed(42)
np.random.seed(42)


def plot_graph_to_pdf(graph: nx.DiGraph, filename: str) -> None:
    """
    Plot and save a tree to a PDF file
    """
    plt.figure(figsize=(3, 3))
    pos = nx.spring_layout(tree, k=2)
    nx.draw(
        tree,
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


def add_random_edges(G: nx.DiGraph, num_edges: int = 10) -> nx.DiGraph:
    """
    Adds random edges to an DiGraph.

    Parameters:
    G (nx.DiGraph): The directed graph.
    num_edges (int): Number of random edges to add.

    Returns:
    nx.DiGraph: The graph with added edges.
    """
    nodes = list(G.nodes)
    added_edges: int = 0

    while added_edges < num_edges:
        source, target = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(source, target):
            G.add_edge(source, target)
            added_edges += 1

    return G


def generate_trees(num_nodes: int, num: int) -> list[nx.DiGraph]:
    """
    Generate a sequence of directed trees with each node being the root once
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


def graph_flood(own: set, in_values: list[set]) -> set:
    ret = own
    for s in in_values:
        ret = ret | s
    return ret


def graph_mean(own: set, in_values: list[set]) -> set:
    ret = own
    for s in in_values:
        ret = ret | s
    return {sum(ret) / len(ret)}


def graph_midpoint(own: set, in_values: list[set]) -> set:
    ret = own
    for s in in_values:
        ret = ret | s
    return {(max(ret) + min(ret)) / 2}


def execute_fun(
    graph: nx.DiGraph,
    num_rounds: int,
    node_values: dict[NodeView, set],
    f: Callable[[set, list[set]], set],
) -> dict[NodeView, set]:
    """
    execution of f on graph for num_rounds with initial values node_values
    """
    for _ in range(num_rounds):
        new_values = {node: set(vals) for node, vals in node_values.items()}
        for node in graph.nodes():
            neighbor_values = [
                node_values[neighbor] for neighbor in graph.predecessors(node)
            ]
            computation = f(node_values[node], neighbor_values)
            new_values[node] = computation.copy()
        node_values = new_values
    return node_values


def plot_trace(node_values: list, fname: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.ylim(0, 1)
    plt.plot(
        range(len(node_values)),
        node_values,
        "-",
        color="gray",
        marker="o",
    )
    plt.xlabel("round")
    plt.ylabel("value")
    plt.savefig(fname, bbox_inches="tight", transparent=True, pad_inches=0.01)


def run_alg(graphs, f, fname: str, propagate_for_rounds: int = 0) -> None:
    random.seed(42)
    to_plot = []
    values = {}
    outputs = []
    for round in range(num_rounds):
        # pick a random graph
        graph = random.sample(trees, k=1)[0]
        plot_graph_to_pdf(graph, f"graph_round_{round}.pdf")
        if round == 0:
            # init
            values = {node: {random.uniform(0, 1)} for node in graph.nodes()}
            outputs = [list(values[node])[0] for node in sorted(graph.nodes().keys())]

        # for plotting
        to_plot.append(outputs)

        if propagate_for_rounds > 0:
            # there is a propagation phase
            if round % propagate_for_rounds == 0:
                # apply f
                values = execute_fun(graph, num_rounds=1, node_values=values, f=f)
                outputs = [
                    list(values[node])[0] for node in sorted(graph.nodes().keys())
                ]
            else:
                # propagate
                values = execute_fun(
                    graph, num_rounds=1, node_values=values, f=graph_flood
                )
                # not output change
        else:
            # apply f
            values = execute_fun(graph, num_rounds=1, node_values=values, f=f)
            outputs = [list(values[node])[0] for node in sorted(graph.nodes().keys())]
    plot_trace(to_plot, fname)


if __name__ == "__main__":
    num_nodes = 10
    tree_num = 3
    num_rounds = 7

    # Generate directed, rooted graphs
    trees = generate_trees(num_nodes, num=tree_num)
    graphs: list[nx.DiGraph] = []
    for i, tree in enumerate(trees):
        # Add edges and plot and save each graph to PDF
        graph = add_random_edges(tree, num_edges=int(np.log(num_nodes)) * num_nodes)
        graphs.append(graph)
        plot_graph_to_pdf(graph, f"graph_{i}.pdf")
        # only for strongly connected digraphs:
        # print(f"diameter of graph {i} is {nx.diameter(graph)}")

    # execute mean
    run_alg(graphs, graph_midpoint, "midpoint.pdf")
    run_alg(graphs, graph_midpoint, f"midpoint-propagate.pdf", propagate_for_rounds=2)
    run_alg(graphs, graph_mean, "mean.pdf")
    run_alg(graphs, graph_mean, f"mean-propagate.pdf", propagate_for_rounds=2)
