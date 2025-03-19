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


def load_graph(input_path, format=None):
    """
    Load a previously saved network graph with automatic format detection.

    Parameters:
    -----------
    input_path : str
        Path to the saved graph file
    format : str, optional
        Format the graph was saved in ('pickle' or 'json')
        If None, will attempt to auto-detect format

    Returns:
    --------
    networkx.Graph
        The loaded network graph
    """
    import pickle
    import json
    import networkx as nx

    # Auto-detect format if not specified
    if format is None:
        # Check file extension
        path_str = str(input_path)
        if path_str.endswith('.pkl') or path_str.endswith('.pickle'):
            format = 'pickle'
        elif path_str.endswith('.json'):
            format = 'json'
        else:
            # Try to detect format by looking at the first few bytes
            try:
                with open(input_path, 'rb') as f:
                    first_bytes = f.read(4)
                    # JSON files typically start with { which is 0x7B in ASCII/UTF-8
                    if first_bytes.startswith(b'{') or first_bytes.startswith(b'['):
                        format = 'json'
                        print(f"Auto-detected format as JSON based on file content")
                    else:
                        format = 'pickle'
                        print(f"Auto-detected format as pickle based on file content")
            except Exception as e:
                print(f"Error during format auto-detection: {e}")
                print(f"Defaulting to JSON format (the save_graph default)")
                format = 'json'

    print(f"Loading graph in {format} format from {input_path}")

    if format == 'pickle':
        try:
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading as pickle: {e}")
            print("Attempting to load as JSON instead...")
            format = 'json'

    if format == 'json':
        try:
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

            print(f"Successfully loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return G

        except UnicodeDecodeError:
            # If we get a unicode error, try again as binary JSON
            print("Error: Failed as text JSON, attempting binary JSON load")
            try:
                with open(input_path, 'rb') as f:
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
            except Exception as e:
                print(f"Error during binary JSON load: {e}")
                raise ValueError(f"Unable to load graph from {input_path} in any supported format")

    raise ValueError(f"Unsupported format: {format}")


def load_filtered_graph(input_path, cutoff_date=None, format=None):
    """
    Load a network graph with filtering based on game dates.

    Parameters:
    -----------
    input_path : str
        Path to the saved graph file
    cutoff_date : datetime.date or datetime.datetime, optional
        Filter games to include only those on or after this date
        If None, returns the full graph without filtering
    format : str, optional
        Format the graph was saved in ('pickle' or 'json')
        If None, will attempt to auto-detect format

    Returns:
    --------
    networkx.Graph
        The loaded and filtered network graph
    """
    # First load the full graph
    graph = load_graph(input_path, format)

    # If no cutoff date specified, return the full graph
    if cutoff_date is None:
        return graph

    # Convert datetime to date if needed
    if isinstance(cutoff_date, datetime):
        cutoff_date = cutoff_date.date()

    print(f"Filtering graph to include only games on or after {cutoff_date}")

    # Find game nodes to filter
    game_nodes = [node for node, data in graph.nodes(data=True)
                  if data.get('type') == 'game']

    # Track nodes and edges to remove
    nodes_to_remove = []
    excluded_count = 0
    pgp_edge_count = 0

    # Process each game node
    for game_id in game_nodes:
        game_data = graph.nodes[game_id]
        game_date = game_data.get('game_date')

        if game_date:
            # Convert string dates to datetime.date objects if needed
            if isinstance(game_date, str):
                try:
                    # Try standard format first
                    game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                except ValueError:
                    try:
                        # Try ISO format (used in JSON serialization)
                        game_date = datetime.fromisoformat(game_date).date()
                    except:
                        # If date parsing fails, include the game
                        continue

            # Convert datetime to date if needed
            if isinstance(game_date, datetime):
                game_date = game_date.date()

            # Exclude game if it's before the cutoff date
            if game_date < cutoff_date:
                nodes_to_remove.append(game_id)
                excluded_count += 1

    # Now identify all nodes and edges to be removed
    all_nodes_to_remove = set()
    pgp_nodes_by_game = {}

    for game_id in nodes_to_remove:
        # Find team game performance nodes
        if game_id in graph.nodes:
            game_data = graph.nodes[game_id]
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            home_tgp = f"{game_id}_{home_team}"
            away_tgp = f"{game_id}_{away_team}"

            # Find player game performance nodes
            pgp_nodes = [node for node in graph.nodes()
                         if isinstance(node, str) and node.startswith(f"{game_id}_")
                         and graph.nodes[node].get('type') == 'player_game_performance']

            # Store PGP nodes by game for edge checking
            pgp_nodes_by_game[game_id] = pgp_nodes

            # Add all related nodes to removal set
            all_nodes_to_remove.add(game_id)
            all_nodes_to_remove.add(home_tgp)
            all_nodes_to_remove.add(away_tgp)
            all_nodes_to_remove.update(pgp_nodes)

    # Count and track PGP-to-PGP edges that will be removed
    for game_id, pgp_nodes in pgp_nodes_by_game.items():
        # Check for PGP-to-PGP edges
        for i, node1 in enumerate(pgp_nodes):
            for node2 in pgp_nodes[i + 1:]:
                if graph.has_edge(node1, node2):
                    pgp_edge_count += 1

    # Remove all identified nodes (edges will be removed automatically)
    for node in all_nodes_to_remove:
        if node in graph.nodes:
            graph.remove_node(node)

    # Update graph indices if they exist
    if 'game_to_pgp' in graph.graph:
        # Convert game_id strings to integers if needed
        ids_to_remove = set()
        for game_id in nodes_to_remove:
            try:
                ids_to_remove.add(int(game_id))
            except:
                ids_to_remove.add(game_id)

        # Filter the game_to_pgp index
        graph.graph['game_to_pgp'] = {
            game_id: pgps for game_id, pgps in graph.graph['game_to_pgp'].items()
            if game_id not in ids_to_remove
        }

    if 'game_to_pgp_edges' in graph.graph:
        # Convert game_id strings to integers if needed
        ids_to_remove = set()
        for game_id in nodes_to_remove:
            try:
                ids_to_remove.add(int(game_id))
            except:
                ids_to_remove.add(game_id)

        # Filter the game_to_pgp_edges index
        graph.graph['game_to_pgp_edges'] = {
            game_id: edges for game_id, edges in graph.graph['game_to_pgp_edges'].items()
            if game_id not in ids_to_remove
        }

    print(f"Filtered out {excluded_count} games before {cutoff_date}")
    print(f"Removed {len(all_nodes_to_remove)} nodes and {pgp_edge_count} PGP-to-PGP edges")
    print(f"Filtered graph now has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    return graph
