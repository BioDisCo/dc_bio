import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Any, Callable
from networkx.classes.reportviews import NodeView

random.seed(42)

def plot_graph_to_pdf(graph: nx.DiGraph, filename: str) -> None:
    """
    Plot and save a tree to a PDF file
    """
    plt.figure(figsize=(3, 3))
    pos = nx.spring_layout(tree, k=2)
    nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
            node_size=500, font_size=10, arrows=True, arrowstyle='->', arrowsize=15)
    plt.savefig(filename, format='pdf')
    plt.close()


def generate_trees(num_nodes: int) -> list[nx.DiGraph]:
    """
    Generate a sequence of directed trees with each node being the root once
    """
    trees: list[nx.DiGraph] = []
    nodes = list(range(num_nodes))
    for root in nodes:
        # Create a complete graph with random weights
        complete_graph = nx.complete_graph(num_nodes)
        for (u, v) in complete_graph.edges():
            complete_graph.edges[u, v]['weight'] = random.random()

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
    return {sum(ret)/len(ret)}

def graph_midpoint(own: set, in_values: list[set]) -> set:
    ret = own
    for s in in_values:
        ret = ret | s
    return {(max(ret) + min(ret))/2}


def execute_fun(graph: nx.DiGraph, num_rounds: int, node_values: dict[NodeView, set], f: Callable[[set, list[set]], set]) -> dict[NodeView, set]:
    """
    execution of f on graph for num_rounds with initial values node_values
    """
    for _ in range(num_rounds):
        new_values = {node: set(vals) for node, vals in node_values.items()}
        for node in graph.nodes():
            neighbor_values = [node_values[neighbor] for neighbor in graph.predecessors(node)]
            computation = f(node_values[node], neighbor_values)
            new_values[node] = computation.copy()
        node_values = new_values
    return node_values


def plot_trace(node_values: list, fname: str):
    plt.figure(figsize=(5, 5))
    plt.ylim(0,1)
    plt.plot(range(len(node_values)), node_values, '-', color='gray', marker='o',)
    plt.xlabel('round')
    plt.ylabel('value')
    plt.savefig(fname,
                bbox_inches='tight', 
                transparent=True,
                pad_inches=0)


if __name__ == "__main__":
    num_nodes = 10
    num_rounds = 10

    # Generate directed trees
    trees = generate_trees(num_nodes)
    for i, tree in enumerate(trees):
        # Plot and save each tree to PDF
        plot_graph_to_pdf(tree, f"tree_{i}.pdf")

        # # test flood
        # print(f"\nTree {i} with root {i}")
        # node_values = {node: {node} for node in tree.nodes()}
        # final_values = execute_fun(tree, num_rounds, node_values, f=graph_flood)
        # for node in sorted(final_values.keys()):
        #     print(f"Node {node}: {final_values[node]}")


    # execute mean
    random.seed(42)
    to_plot = []
    values = {}
    for round in range(num_rounds):
        # print("round", round)
        # pick a random graph
        graph = random.sample(trees, k=1)[0]
        if round == 0:
            # init
            values = {node: {random.uniform(0,1)} for node in graph.nodes()}
        
        current_vals = [list(values[node])[0] for node in sorted(values.keys())]
        to_plot.append(current_vals)

        values = execute_fun(graph, num_rounds=10, node_values=values, f=graph_flood)
        values = execute_fun(graph, num_rounds=1, node_values=values, f=graph_mean)

    plot_trace(to_plot, "mean.pdf")

    # execute midpoint
    random.seed(42)
    to_plot = []
    values = {}
    for round in range(num_rounds):
        # print("round", round)
        # pick a random graph
        graph = random.sample(trees, k=1)[0]
        if round == 0:
            # init
            values = {node: {random.uniform(0,1)} for node in graph.nodes()}
        
        current_vals = [list(values[node])[0] for node in sorted(values.keys())]
        to_plot.append(current_vals)

        values = execute_fun(graph, num_rounds=1, node_values=values, f=graph_midpoint)

    plot_trace(to_plot, "midpoint.pdf")


        
