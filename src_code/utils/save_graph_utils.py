import networkx as nx
import pickle
import json
from pathlib import Path


def save_graph(graph, output_path, format='pickle'):
    """
    Save the network graph to a file.

    Parameters:
    -----------
    graph : networkx.Graph
        The network graph to save
    output_path : str
        Path where the graph should be saved
    format : str
        Format to save the graph in ('pickle' or 'json')
    """
    # Create directory if it doesn't exist
    if format == 'pickle':
        # Save as pickle (preserves all graph attributes and structure)
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)

    elif format == 'json':
        # Convert to dict format that can be JSON serialized
        graph_data = {
            'nodes': [[n, graph.nodes[n]] for n in graph.nodes()],
            'edges': [[u, v, graph.edges[u, v]] for u, v in graph.edges()]
        }
        with open(output_path, 'w') as f:
            json.dump(graph_data, f)


def load_graph(input_path, format='pickle'):
    """
    Load a previously saved network graph.

    Parameters:
    -----------
    input_path : str
        Path to the saved graph file
    format : str
        Format the graph was saved in ('pickle' or 'json')

    Returns:
    --------
    networkx.Graph
        The loaded network graph
    """
    if format == 'pickle':
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    elif format == 'json':
        with open(input_path, 'r') as f:
            graph_data = json.load(f)

        # Create new graph and populate it
        G = nx.Graph()

        # Add nodes with their attributes
        for node, attrs in graph_data['nodes']:
            G.add_node(node, **attrs)

        # Add edges with their attributes
        for u, v, attrs in graph_data['edges']:
            G.add_edge(u, v, **attrs)

        return G