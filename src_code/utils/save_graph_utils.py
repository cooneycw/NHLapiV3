from datetime import datetime
import copy
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


def load_filtered_graph(graph_path, cutoff_date=None):
    """
    Load a NetworkX graph and optionally filter out games before a cutoff date.

    Args:
        graph_path: Path to the saved graph file
        cutoff_date: Optional datetime.date - if provided, only games on or after this date
                    will be included in the returned graph

    Returns:
        filtered_graph: NetworkX graph with only games after the cutoff date
    """

    print(f"Loading graph from {graph_path}")
    original_graph = load_graph(graph_path)

    if cutoff_date is None:
        print(f"Loaded full graph with {len(original_graph.nodes)} nodes and {len(original_graph.edges)} edges")
        return original_graph

    # Make a deep copy of the graph to avoid modifying the original
    filtered_graph = copy.deepcopy(original_graph)

    # Get all game nodes
    game_nodes = [node for node, data in filtered_graph.nodes(data=True)
                  if data.get('type') == 'game']

    # Track nodes and edges to remove
    nodes_to_remove = []

    # Filter games by date
    excluded_games = 0
    for game_id in game_nodes:
        game_data = filtered_graph.nodes[game_id]
        game_date = game_data.get('game_date')

        exclude_game = False
        if game_date:
            # Convert string dates to datetime.date objects if needed
            if isinstance(game_date, str):
                try:
                    from datetime import datetime
                    game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                except:
                    # If date parsing fails, keep the game (can't determine if it should be excluded)
                    continue

            if isinstance(game_date, datetime):
                game_date = game_date.date()

            # Check if game is before cutoff date
            if game_date < cutoff_date:
                exclude_game = True
                excluded_games += 1

        # If game should be excluded, mark it and its related nodes for removal
        if exclude_game:
            nodes_to_remove.append(game_id)

            # Also remove team game performance nodes related to this game
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            if home_team:
                home_tgp = f"{game_id}_{home_team}"
                if home_tgp in filtered_graph.nodes:
                    nodes_to_remove.append(home_tgp)

            if away_team:
                away_tgp = f"{game_id}_{away_team}"
                if away_tgp in filtered_graph.nodes:
                    nodes_to_remove.append(away_tgp)

            # Find and mark player game performance nodes for removal
            for node in list(filtered_graph.nodes()):
                if isinstance(node, str) and node.startswith(f"{game_id}_") and node not in nodes_to_remove:
                    nodes_to_remove.append(node)

    # Remove the nodes
    for node in nodes_to_remove:
        if node in filtered_graph:
            filtered_graph.remove_node(node)

    total_games = len(game_nodes)
    remaining_games = total_games - excluded_games

    print(f"Date filtering: Excluded {excluded_games} games before {cutoff_date}")
    print(f"Retained {remaining_games} games ({(remaining_games / total_games) * 100:.1f}% of total)")
    print(f"Filtered graph has {len(filtered_graph.nodes)} nodes and {len(filtered_graph.edges)} edges")

    return filtered_graph
