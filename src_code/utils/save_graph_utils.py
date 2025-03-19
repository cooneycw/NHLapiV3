from datetime import datetime
import copy
import networkx as nx
import pickle
import json
from pathlib import Path


def save_graph(graph, output_path, format='json'):
    """
    Save the network graph to a file.

    Parameters:
    -----------
    graph : networkx.Graph
        The network graph to save
    output_path : str
        Path where the graph should be saved
    format : str
        Format to save the graph in ('json' or 'pickle')
        JSON format is recommended for better filtering capabilities and readability.
    """
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        # Convert to dict format that can be JSON serialized
        try:
            graph_data = {
                'nodes': [[n, _make_json_serializable(graph.nodes[n])] for n in graph.nodes()],
                'edges': [[u, v, _make_json_serializable(graph.edges[u, v])] for u, v in graph.edges()]
            }
            with open(output_path, 'w') as f:
                json.dump(graph_data, f)
            print(f"Graph saved in JSON format to {output_path}")
            print(f"JSON format allows for efficient filtering during load")
        except (TypeError, ValueError) as e:
            print(f"Error saving as JSON: {e}")
            print("Some objects in the graph may not be JSON serializable")
            print("Falling back to pickle format...")
            format = 'pickle'

    if format == 'pickle':
        # Save as pickle (preserves all graph attributes and structure)
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Graph saved in pickle format to {output_path}")


def _make_json_serializable(obj):
    """Helper function to convert common non-JSON-serializable objects to serializable forms."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):  # Convert custom objects to dicts
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith('_')}
    return obj


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
    Load a NetworkX graph and filter out games before a cutoff date during the loading process.

    Args:
        graph_path: Path to the saved graph file
        cutoff_date: Optional datetime.date - if provided, only games on or after this date
                    will be included in the returned graph

    Returns:
        filtered_graph: NetworkX graph with only games after the cutoff date
    """
    print(f"Loading graph from {graph_path}")

    # Initialize counters
    total_games = 0
    excluded_games = 0

    # Determine file format based on extension and fallback to content inspection
    path_str = str(graph_path)
    if path_str.endswith('.pkl') or path_str.endswith('.pickle'):
        file_format = 'pickle'
    elif path_str.endswith('.json'):
        file_format = 'json'
    else:
        # Try to detect format by looking at the first few bytes
        try:
            with open(graph_path, 'rb') as f:
                first_bytes = f.read(4)
                # JSON files typically start with { which is 0x7B in ASCII/UTF-8
                if first_bytes.startswith(b'{') or first_bytes.startswith(b'['):
                    file_format = 'json'
                # Otherwise assume it's pickle
                else:
                    file_format = 'pickle'
                    print(f"No recognized extension, guessing format as pickle based on content")
        except Exception as e:
            print(f"Error inspecting file: {e}")
            print(f"Defaulting to pickle format")
            file_format = 'pickle'

    # Handle pickle format - can't easily filter during load
    if file_format == 'pickle':
        with open(graph_path, 'rb') as f:
            original_graph = pickle.load(f)

        # If no cutoff date, return the full graph
        if cutoff_date is None:
            print(f"Loaded full graph with {len(original_graph.nodes)} nodes and {len(original_graph.edges)} edges")
            return original_graph

        # Otherwise need to filter the loaded graph
        filtered_graph = nx.Graph()

        # First identify all game nodes to keep
        kept_game_ids = set()
        for node, data in original_graph.nodes(data=True):
            if data.get('type') == 'game':
                total_games += 1
                game_date = data.get('game_date')

                # Parse date if it's a string
                if isinstance(game_date, str):
                    try:
                        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                    except:
                        # If date parsing fails, keep the game
                        kept_game_ids.add(node)
                        continue

                if isinstance(game_date, datetime):
                    game_date = game_date.date()

                # Keep game if it's on or after cutoff date
                if cutoff_date is None or game_date is None or game_date >= cutoff_date:
                    kept_game_ids.add(node)
                else:
                    excluded_games += 1

        # Add nodes to keep
        for node, data in original_graph.nodes(data=True):
            # Keep all game nodes that passed the date filter
            if data.get('type') == 'game' and node in kept_game_ids:
                filtered_graph.add_node(node, **data)
            # For performance nodes, check if they belong to a kept game
            elif isinstance(node, str) and '_' in node:
                # Try to extract game_id from node name
                parts = node.split('_', 1)
                if len(parts) >= 1 and parts[0] in kept_game_ids:
                    filtered_graph.add_node(node, **data)
            # Keep all non-game, non-performance nodes
            elif data.get('type') != 'game' and not any(isinstance(node, str) and node.startswith(f"{game_id}_")
                                                        for game_id in original_graph.nodes
                                                        if original_graph.nodes.get(game_id, {}).get('type') == 'game'):
                filtered_graph.add_node(node, **data)

        # Only add edges where both endpoints exist in filtered graph
        for u, v, data in original_graph.edges(data=True):
            if u in filtered_graph.nodes and v in filtered_graph.nodes:
                filtered_graph.add_edge(u, v, **data)

    # Handle JSON format - can filter during load
    else:  # json format
        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
        except UnicodeDecodeError:
            # If we get a unicode error, try again as pickle
            print("Error: Failed to decode as JSON, attempting to load as pickle instead")
            file_format = 'pickle'
            return load_filtered_graph(graph_path, cutoff_date)  # Recursive call with corrected format

        # Create new graph
        filtered_graph = nx.Graph()

        # Filter nodes during loading
        kept_game_ids = set()

        # First pass: identify game nodes to keep based on date
        for node_data in graph_data['nodes']:
            node, attrs = node_data

            if attrs.get('type') == 'game':
                total_games += 1
                game_date = attrs.get('game_date')

                # Parse date if it's a string
                if isinstance(game_date, str):
                    try:
                        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                    except:
                        # If date parsing fails, keep the game
                        kept_game_ids.add(node)
                        continue

                if isinstance(game_date, datetime):
                    game_date = game_date.date()

                # If no cutoff or game meets criteria, add to graph
                if cutoff_date is None or game_date is None or game_date >= cutoff_date:
                    kept_game_ids.add(node)
                else:
                    excluded_games += 1

        # Second pass: add filtered nodes to graph
        for node_data in graph_data['nodes']:
            node, attrs = node_data

            # Game nodes - only add if they passed date filter
            if attrs.get('type') == 'game':
                if node in kept_game_ids:
                    filtered_graph.add_node(node, **attrs)
            # Performance nodes - check if they belong to a kept game
            elif isinstance(node, str) and '_' in node:
                # Try to extract game_id from node name
                parts = node.split('_', 1)
                if len(parts) >= 1 and parts[0] in kept_game_ids:
                    filtered_graph.add_node(node, **attrs)
            # Other nodes - add all
            else:
                filtered_graph.add_node(node, **attrs)

        # Add edges where both endpoints exist in filtered graph
        for edge_data in graph_data['edges']:
            u, v, attrs = edge_data
            if u in filtered_graph.nodes and v in filtered_graph.nodes:
                filtered_graph.add_edge(u, v, **attrs)

    # Print statistics
    remaining_games = total_games - excluded_games

    print(f"Date filtering: Excluded {excluded_games} games before {cutoff_date}")

    # Avoid division by zero
    if total_games > 0:
        percentage = (remaining_games / total_games) * 100
        print(f"Retained {remaining_games} games ({percentage:.1f}% of total)")
    else:
        print(f"No game nodes found in the graph (0 total games)")

    print(f"Filtered graph has {len(filtered_graph.nodes)} nodes and {len(filtered_graph.edges)} edges")

    return filtered_graph
