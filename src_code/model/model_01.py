from src_code.utils.utils import load_game_data, create_player_dict
from src_code.utils.graph_utils import (
    create_graph, add_team_node, add_player_node, add_game, process_games_chronologically,
    add_player_game_performance, update_tgp_stats, update_pgp_stats, update_pgp_edge_stats,
    update_game_outcome,
    calculate_historical_stats,
    build_game_to_pgp_index, build_game_to_pgp_edges_index)
from src_code.utils.display_graph_utils import visualize_game_graph
from src_code.utils.save_graph_utils import save_graph, load_filtered_graph
import copy
import gc
import matplotlib.pyplot as plt
import traceback
from datetime import datetime


def model_data(config):
    """Process curated game data and build a graph model for analysis."""
    print("Starting model data processing...")

    # Define all data dimensions
    dimension_names = "all_names"
    dimension_teams = "all_teams"
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    dimension_curated = "all_curated"
    dimension_curated_data = "all_curated_data"

    # Load required datasets
    print("Loading datasets...")
    data_names = config.load_data(dimension_names)
    data_games = config.load_data(dimension_games)
    data_teams = config.load_data(dimension_teams)
    data_game_roster = config.load_data(dimension_game_rosters)
    data_curated, all_curated_data = config.load_curated_data()

    # Clear variables that aren't needed immediately

    # Process curated game IDs
    processed_game_ids = set()
    if data_curated:
        processed_game_ids = set(data_curated)
        print(f"Loaded {len(processed_game_ids)} curated game IDs.")
    else:
        print("Warning: No curated games data found. Processing will continue with no games.")

    # Check curated game data
    if not all_curated_data:
        print("Warning: No curated game data found. Graph will be incomplete.")
        all_curated_data = {}
    else:
        print(f"Loaded curated data for {len(all_curated_data)} games.")

    # Filter games to only include those that have been curated
    filtered_games = []
    for game in data_games:
        if 'id' in game and game['id'] in processed_game_ids:
            filtered_games.append(game)

    print(f"Using {len(filtered_games)} out of {len(data_games)} games that have been curated.")

    # Clear data_games as we now have filtered_games
    del data_games
    del data_curated  # Already converted to processed_game_ids set
    gc.collect()

    # Sort filtered games chronologically
    filtered_games.sort(key=lambda x: x['game_date'] if isinstance(x['game_date'], datetime)
    else datetime.strptime(x['game_date'], '%Y-%m-%d'))

    if filtered_games:
        print(
            f"Games sorted chronologically from {filtered_games[0]['game_date']} to {filtered_games[-1]['game_date']}")

    # Create player dictionary
    player_list, player_dict = create_player_dict(data_names)
    del data_names  # No longer needed after creating player_dict
    gc.collect()

    # Initialize the graph
    data_graph = create_graph()

    # Add team nodes
    print("Adding team nodes to graph...")
    for i, team in enumerate(data_teams):
        add_team_node(data_graph, team)

    # Clear teams data after adding to graph
    del data_teams
    gc.collect()

    # Add player nodes
    print("Adding player nodes to graph...")
    for j, player in enumerate(player_list):
        add_player_node(data_graph, player, player_dict)

    # Clear player data after adding to graph
    del player_list
    del player_dict
    gc.collect()

    # Process games chronologically
    print("Processing games chronologically...")
    process_games_chronologically(data_graph, filtered_games)

    # Create mapping for efficient lookups
    game_id_to_index = {game['id']: i for i, game in enumerate(filtered_games)}

    # Filter and sort game rosters
    filtered_game_rosters = []
    for roster in data_game_roster:
        if roster and len(roster) > 0 and 'game_id' in roster[0] and roster[0]['game_id'] in processed_game_ids:
            filtered_game_rosters.append(roster)

    # Clear original roster data
    del data_game_roster
    gc.collect()

    # Create sorted game rosters
    game_id_to_roster = {roster[0]['game_id']: roster for roster in filtered_game_rosters if roster and len(roster) > 0}
    sorted_game_rosters = []
    for game in filtered_games:
        if game['id'] in game_id_to_roster:
            sorted_game_rosters.append(game_id_to_roster[game['id']])

    # Clear intermediate data
    del filtered_game_rosters
    del game_id_to_roster
    gc.collect()

    print(f"Matched {len(sorted_game_rosters)} game rosters to filtered games.")

    # Add player game performances
    print("Adding player game performances...")
    team_game_maps = []
    for l, roster in enumerate(sorted_game_rosters):
        if l % 50 == 0:
            print(f"Processing roster {l} of {len(sorted_game_rosters)}")
            gc.collect()  # Periodic garbage collection
        team_game_map = add_player_game_performance(data_graph, roster)
        team_game_maps.append(team_game_map)

    print(f"Graph now has {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Create mapping from game ID to team_game_map index
    game_id_to_map_index = {}
    for i, roster in enumerate(sorted_game_rosters):
        if roster and len(roster) > 0 and 'game_id' in roster[0]:
            game_id_to_map_index[roster[0]['game_id']] = i

    # Process shift data
    print("Processing shift data for each game...")
    shifts = []
    processed_count = 0
    error_count = 0

    for m, game in enumerate(filtered_games):
        verbose = False
        if (m % 40 == 0) and (m != 0):
            print(f'Game {m} of {len(filtered_games)} ({(m / len(filtered_games)) * 100:.1f}%)')
            verbose = True
            gc.collect()  # Periodic garbage collection

        try:
            # Get game ID
            game_id = game['id']

            # Get shift data
            if game_id not in all_curated_data:
                print(f"Warning: No curated data found for game {game_id}")
                error_count += 1
                continue

            shift_data = all_curated_data[game_id]

            # Process shift data
            if game_id in game_id_to_map_index:
                map_index = game_id_to_map_index[game_id]
                process_shift_data(data_graph, verbose, team_game_maps[map_index], shift_data)
                shifts.append(shift_data)
                update_game_outcome(data_graph, game_id, game)
                processed_count += 1

                # Clear shift_data after processing
                shift_data = None
            else:
                print(f"Warning: Could not find matching team_game_map for game {game_id}")
                error_count += 1
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error processing game {game['id']}: {str(e)}\n{error_trace}")
            error_count += 1

    print(f"Processed shift data for {processed_count} games successfully. {error_count} games had errors.")

    # Clear large data structures no longer needed
    del filtered_games
    del sorted_game_rosters
    del team_game_maps
    del all_curated_data
    del game_id_to_map_index
    del shifts
    gc.collect()

    # Build indexes for efficient lookups
    print("Building performance mapping indexes...")
    build_game_to_pgp_index(data_graph)
    build_game_to_pgp_edges_index(data_graph)

    # Calculate historical statistics
    print("Processing historical game stats...")
    data_graph = calculate_historical_stats(config, data_graph)

    # Save the completed graph
    print("Saving graph to disk...")
    save_graph(data_graph, config.file_paths["graph"])

    print("Model data processing complete!")
    return data_graph


def diagnose_graph(graph, sample_game_id=None):
    """
    Comprehensive diagnostic function to analyze the structure of the graph.

    Args:
        graph: NetworkX graph to analyze
        sample_game_id: Optional specific game ID to analyze in detail
    """
    print("\n===== GRAPH STRUCTURE DIAGNOSIS =====")

    # 1. Basic graph statistics
    print(f"\nTotal nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")

    # 2. Count nodes by type
    node_types = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")

    # 3. Count edges by type
    edge_types = {}
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print("\nEdge types:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")

    # 4. If sample game provided, analyze in detail
    if sample_game_id and sample_game_id in graph.nodes:
        print(f"\nDetailed analysis for game {sample_game_id}:")

        # Get game data
        game_data = graph.nodes[sample_game_id]
        home_team = game_data.get('home_team', 'unknown')
        away_team = game_data.get('away_team', 'unknown')
        print(f"  Home team: {home_team}")
        print(f"  Away team: {away_team}")

        # Find TGP nodes for this game
        home_tgp = f"{sample_game_id}_{home_team}"
        away_tgp = f"{sample_game_id}_{away_team}"
        tgp_exists = []
        if home_tgp in graph.nodes:
            tgp_exists.append(home_tgp)
        if away_tgp in graph.nodes:
            tgp_exists.append(away_tgp)

        print(f"  TGP nodes found: {len(tgp_exists)} of 2 expected")

        # Find PGP nodes for this game
        pgp_nodes = []
        for node in graph.nodes():
            if isinstance(node, str) and node.startswith(f"{sample_game_id}_") and graph.nodes[node].get(
                    'type') == 'player_game_performance':
                pgp_nodes.append(node)

        print(f"  PGP nodes found: {len(pgp_nodes)}")
        print(f"  PGP nodes by team:")

        home_pgp = [n for n in pgp_nodes if graph.nodes[n].get('player_team') == home_team]
        away_pgp = [n for n in pgp_nodes if graph.nodes[n].get('player_team') == away_team]
        print(f"    {home_team}: {len(home_pgp)}")
        print(f"    {away_team}: {len(away_pgp)}")

        # Check if PGP nodes have data
        print("\n  PGP nodes statistics (first 3 from each team):")
        for team, pgp_list in [(home_team, home_pgp), (away_team, away_pgp)]:
            print(f"    {team} players:")
            for i, pgp in enumerate(pgp_list[:3]):
                node_data = graph.nodes[pgp]
                player_name = node_data.get('player_name', 'Unknown')
                stats = {
                    'toi': node_data.get('toi', [0, 0, 0]),
                    'goal': node_data.get('goal', [0, 0, 0]),
                    'assist': node_data.get('assist', [0, 0, 0]),
                    'shot_on_goal': node_data.get('shot_on_goal', [0, 0, 0])
                }
                print(f"      {i + 1}. {player_name} ({pgp}):")
                print(f"         TOI: {stats['toi']}")
                print(f"         Goals: {stats['goal']}")
                print(f"         Assists: {stats['assist']}")
                print(f"         Shots: {stats['shot_on_goal']}")

        # Check PGP-to-PGP edges
        pgp_pgp_edges = []
        for u, v, data in graph.edges(data=True):
            if (isinstance(u, str) and isinstance(v, str) and
                    u.startswith(f"{sample_game_id}_") and v.startswith(f"{sample_game_id}_") and
                    graph.nodes[u].get('type') == 'player_game_performance' and
                    graph.nodes[v].get('type') == 'player_game_performance'):
                pgp_pgp_edges.append((u, v))

        print(f"\n  PGP-to-PGP edges found: {len(pgp_pgp_edges)}")

        # Check a sample of PGP-to-PGP edges
        if pgp_pgp_edges:
            print("  Sample PGP-to-PGP edge data (first 3):")
            for i, (u, v) in enumerate(pgp_pgp_edges[:3]):
                edge_data = graph.get_edge_data(u, v)
                u_name = graph.nodes[u].get('player_name', 'Unknown')
                v_name = graph.nodes[v].get('player_name', 'Unknown')
                stats = {
                    'toi': edge_data.get('toi', [0, 0, 0]),
                    'goal': edge_data.get('goal', [0, 0, 0]),
                }
                print(f"    {i + 1}. {u_name} <-> {v_name}:")
                print(f"       TOI: {stats['toi']}")
                print(f"       Goals: {stats['goal']}")

                # Check for historical data
                has_hist = False
                for key in edge_data:
                    if key.startswith('hist_'):
                        has_hist = True
                        break
                print(f"       Has historical data: {has_hist}")

        # Check for 'game_to_pgp' index
        print("\n  Graph indices:")
        has_pgp_index = 'game_to_pgp' in graph.graph
        has_edge_index = 'game_to_pgp_edges' in graph.graph
        print(f"    Has game_to_pgp index: {has_pgp_index}")
        print(f"    Has game_to_pgp_edges index: {has_edge_index}")

        # If indices exist, check their content for this game
        if has_pgp_index:
            pgp_index_count = len(graph.graph['game_to_pgp'].get(int(sample_game_id), []))
            print(f"    PGP entries in index for this game: {pgp_index_count}")

        if has_edge_index:
            edge_index_count = len(graph.graph['game_to_pgp_edges'].get(int(sample_game_id), []))
            print(f"    PGP edge entries in index for this game: {edge_index_count}")

    print("\n===== END OF DIAGNOSIS =====\n")


def model_visualization(config):
    """
    Generate visualizations for games in the graph, with optional date filtering.

    Args:
        config: A Config object containing settings and paths
    """
    # Check if we should apply date filtering
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    print(f"Starting visualization process...")
    data_graph = load_filtered_graph(config.file_paths["graph"], cutoff_date=training_cutoff_date)

    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Rebuild the missing indices if they don't exist
    print("Checking and rebuilding indices if needed...")
    if 'game_to_pgp' not in data_graph.graph:
        print("Rebuilding game_to_pgp index...")
        build_game_to_pgp_index(data_graph)

    if 'game_to_pgp_edges' not in data_graph.graph:
        print("Rebuilding game_to_pgp_edges index...")
        build_game_to_pgp_edges_index(data_graph)

    # Get games and apply filtering if needed
    dimension_games = "all_boxscores"
    data_games = config.load_data(dimension_games)

    # If we have date filtering, filter the games list too
    if training_cutoff_date:
        filtered_games = []
        excluded_count = 0

        for game in data_games:
            game_date = game.get('game_date')
            if game_date:
                # Convert string dates to datetime.date objects if needed
                if isinstance(game_date, str):
                    try:
                        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                    except:
                        # If date parsing fails, include the game
                        filtered_games.append(game)
                        continue

                if isinstance(game_date, datetime):
                    game_date = game_date.date()

                # Include game if it's on or after the cutoff date
                if game_date >= training_cutoff_date:
                    filtered_games.append(game)
                else:
                    excluded_count += 1
            else:
                # Include games with no date information
                filtered_games.append(game)

        print(f"Date filtering: Excluded {excluded_count} games before {training_cutoff_date}")
        print(f"Generating visualizations for {len(filtered_games)} games...")
        visualization_games = filtered_games
    else:
        print(f"Generating visualizations for all {len(data_games)} games...")
        visualization_games = data_games

    # Run a quick diagnostic on the graph
    print("Running graph diagnostics...")
    sample_game_id = visualization_games[0]['id'] if visualization_games else None
    if sample_game_id:
        diagnose_graph(data_graph, sample_game_id)

    # Generate visualizations
    for m, game in enumerate(visualization_games):
        # Check if game exists in the filtered graph
        game_id = game.get('id')
        if game_id not in data_graph.nodes:
            print(f"Skipping game {game_id} - not found in filtered graph")
            continue

        # Create visualization for every 10th game or the last game
        if m % 10 == 0 or m == len(visualization_games) - 1:
            print(f'Generating visualization for game {game_id}')

            # Generate visualizations for different window sizes
            for window_size in config.stat_window_sizes:
                output_path = (f"{config.file_paths['game_output_jpg']}/game_{game_id}_"
                               f"network_{game['game_date']}_window{window_size}.jpg")

                try:
                    visualize_game_graph(
                        data_graph,
                        game_id,
                        window_size=window_size,
                        output_path=output_path,
                        edge_sample_rate=0.05,
                    )
                    plt.close('all')
                except Exception as e:
                    print(f"Error visualizing game {game_id}: {str(e)}")
                    print(traceback.format_exc())

                # Force garbage collection
                gc.collect()


def process_shift_data(data_graph, verbose, team_game_map, shift_data):
    # Extract only needed data to reduce memory duplication
    game_id = shift_data["game_id"]
    period_code = shift_data["period_code"]
    player_data = shift_data["player_data"]

    for i, _ in enumerate(shift_data["shift_id"]):
        team_map = {}
        line_player_team_map = {}

        # Build player-team mapping
        for player_dat in player_data[i]:
            player_team = player_dat['player_team']
            if player_team not in team_map:
                team_map[player_team] = (game_id[i], player_team)

            game_team = (game_id[i], player_team)
            if game_team not in line_player_team_map:
                line_player_team_map[game_team] = []
            line_player_team_map[game_team].append(player_dat['player_id'])

        # Process each team
        for team in line_player_team_map:
            other_players = copy.deepcopy(line_player_team_map[team])

            for player_dat in player_data[i]:
                if player_dat['player_id'] not in line_player_team_map[team]:
                    continue

                other_players.remove(player_dat['player_id'])
                team_tgp = str(team[0]) + '_' + team[1]
                player_pgp = str(team[0]) + '_' + str(player_dat['player_id'])

                # Update team and player stats
                update_tgp_stats(data_graph, team_tgp, period_code[i], player_dat)
                update_pgp_stats(data_graph, player_pgp, period_code[i], player_dat)

                # Update player-player edges
                for other in other_players:
                    other_pgp = str(team[0]) + '_' + str(other)
                    update_pgp_edge_stats(data_graph, player_pgp, other_pgp, period_code[i], player_dat)

            # Clean up after processing each team
            other_players = None

        # Clean up after each shift
        team_map = None
        line_player_team_map = None

    # Encourage garbage collection after processing complex structures
    if verbose:
        gc.collect()


def find_goals(player_data):
    goals = []
    for i, shift_player in enumerate(player_data):
        # Check if any period has a goal
        if 1 in shift_player['goal']:
            goals.append({
                'player_index': i,
                'player_id': shift_player['player_id'],
                'player_team': shift_player['player_team'],
                'goal_periods': [p + 1 for p, goal in enumerate(shift_player['goal']) if goal == 1]
            })
    return goals

