from src_code.utils.utils import load_game_data, create_player_dict
from src_code.utils.graph_utils import (
    create_graph, add_team_node, add_player_node, add_game, process_games_chronologically,
    add_player_game_performance, update_tgp_stats, update_pgp_stats, update_pgp_edge_stats,
    update_game_outcome, get_historical_tgp_stats, get_historical_pgp_stats,
    calculate_historical_stats,
    build_game_to_pgp_index, build_game_to_pgp_edges_index)
from src_code.utils.display_graph_utils import visualize_game_graph
from src_code.utils.save_graph_utils import save_graph, load_graph
import copy
import gc
import matplotlib.pyplot as plt
from datetime import datetime


def model_data(config):
    """
    Process curated game data and build a graph model for analysis.

    This function:
    1. Loads all required datasets including the curated games list and consolidated game data
    2. Filters games to only include those that have been curated
    3. Processes the filtered games in chronological order
    4. Builds a graph model with nodes for teams, players, games and performances
    5. Calculates historical statistics for analysis

    Args:
        config: A Config object containing settings and paths

    Returns:
        NetworkX graph object - The processed graph
    """
    print("Starting model data processing...")

    # Define all data dimensions
    dimension_names = "all_names"
    dimension_teams = "all_teams"
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    dimension_curated = "all_curated"  # IDs of curated games
    dimension_curated_data = "all_curated_data"  # Combined data for all curated games

    # Load all required datasets
    print("Loading datasets...")
    data_names = config.load_data(dimension_names)
    data_games = config.load_data(dimension_games)
    data_teams = config.load_data(dimension_teams)
    data_players = config.load_data(dimension_players)
    data_shifts = config.load_data(dimension_shifts)
    data_plays = config.load_data(dimension_plays)
    data_game_roster = config.load_data(dimension_game_rosters)
    data_curated = config.load_data(dimension_curated)  # Load the curated games list
    all_curated_data = config.load_data(dimension_curated_data)  # Load the combined game data

    # Check if curated data exists and convert to set for efficient lookups
    processed_game_ids = set()
    if data_curated:
        processed_game_ids = set(data_curated)
        print(f"Loaded {len(processed_game_ids)} curated game IDs.")
    else:
        print("Warning: No curated games data found. Processing will continue with no games.")

    # Check if we have the combined curated game data
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

    # Sort filtered games by game_date for chronological processing
    filtered_games.sort(key=lambda x: x['game_date'] if isinstance(x['game_date'], datetime)
    else datetime.strptime(x['game_date'], '%Y-%m-%d'))

    if filtered_games:
        print(
            f"Games sorted chronologically from {filtered_games[0]['game_date']} to {filtered_games[-1]['game_date']}")

    # Create player dictionary from names data
    player_list, player_dict = create_player_dict(data_names)

    # Initialize the graph
    data_graph = create_graph()

    # Add team nodes
    print("Adding team nodes to graph...")
    for i, team in enumerate(data_teams):
        add_team_node(data_graph, team)

    # Add player nodes
    print("Adding player nodes to graph...")
    for j, player in enumerate(player_list):
        add_player_node(data_graph, player, player_dict)

    # Process games chronologically to build basic graph structure
    print("Processing games chronologically...")
    process_games_chronologically(data_graph, filtered_games)

    # Create a mapping from game ID to index for efficient lookups
    game_id_to_index = {game['id']: i for i, game in enumerate(filtered_games)}

    # Filter and sort game rosters to match filtered games
    filtered_game_rosters = []
    for roster in data_game_roster:
        if roster and len(roster) > 0 and 'game_id' in roster[0] and roster[0]['game_id'] in processed_game_ids:
            filtered_game_rosters.append(roster)

    # Sort rosters to match the order of filtered_games
    game_id_to_roster = {roster[0]['game_id']: roster for roster in filtered_game_rosters if roster and len(roster) > 0}
    sorted_game_rosters = []
    for game in filtered_games:
        if game['id'] in game_id_to_roster:
            sorted_game_rosters.append(game_id_to_roster[game['id']])

    print(f"Matched {len(sorted_game_rosters)} game rosters to filtered games.")

    # Add player game performances
    print("Adding player game performances...")
    team_game_maps = []
    for l, roster in enumerate(sorted_game_rosters):
        if l % 50 == 0:
            print(f"Processing roster {l} of {len(sorted_game_rosters)}")
        team_game_map = add_player_game_performance(data_graph, roster)
        team_game_maps.append(team_game_map)

    print(f"Graph now has {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Create mapping from game ID to team_game_map index
    game_id_to_map_index = {}
    for i, roster in enumerate(sorted_game_rosters):
        if roster and len(roster) > 0 and 'game_id' in roster[0]:
            game_id_to_map_index[roster[0]['game_id']] = i

    # Process shift data for each game
    print("Processing shift data for each game...")
    shifts = []
    processed_count = 0
    error_count = 0

    for m, game in enumerate(filtered_games):
        verbose = False
        if (m % 40 == 0) and (m != 0):
            print(f'Game {m} of {len(filtered_games)} ({(m / len(filtered_games)) * 100:.1f}%)')
            verbose = True

        try:
            # Get game ID
            game_id = game['id']

            # Get the shift data from the consolidated data dictionary
            if game_id not in all_curated_data:
                print(f"Warning: No curated data found for game {game_id}")
                error_count += 1
                continue

            shift_data = all_curated_data[game_id]

            # Get the correct team_game_map index for this game
            if game_id in game_id_to_map_index:
                map_index = game_id_to_map_index[game_id]
                process_shift_data(data_graph, verbose, team_game_maps[map_index], shift_data)
                shifts.append(shift_data)
                update_game_outcome(data_graph, game_id, game)
                processed_count += 1
            else:
                print(f"Warning: Could not find matching team_game_map for game {game_id}")
                error_count += 1
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error processing game {game['id']}: {str(e)}\n{error_trace}")
            error_count += 1

    print(f"Processed shift data for {processed_count} games successfully. {error_count} games had errors.")

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


def model_visualization(config):
    """
    Generate visualizations for games in the graph, with optional date filtering.

    Args:
        config: A Config object containing settings and paths
    """
    from src_code.utils.save_graph_utils import load_graph, load_filtered_graph

    # Check if we should apply date filtering
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    # Load graph with optional date filtering
    if training_cutoff_date:
        print(f"Loading graph with date filtering (games on or after {training_cutoff_date})...")
        data_graph = load_filtered_graph(config.file_paths["graph"], training_cutoff_date)
    else:
        print("Loading complete graph (no date filtering)...")
        data_graph = load_graph(config.file_paths["graph"])

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
                        from datetime import datetime
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
        data_games = filtered_games
    else:
        print(f"Generating visualizations for all {len(data_games)} games...")

    # Generate visualizations
    for m, game in enumerate(data_games):
        # Check if game exists in the filtered graph
        game_id = game.get('id')
        if game_id not in data_graph.nodes:
            continue

        # Create visualization for every 10th game or the last game
        if m % 10 == 0 or m == len(data_games) - 1:
            print(f'Generating visualization for game {game["id"]}')

            # Generate visualizations for different window sizes
            for window_size in config.stat_window_sizes:
                output_path = (f"{config.file_paths['game_output_jpg']}/game_{game['id']}_"
                               f"network_{game['game_date']}_window{window_size}.jpg")

                visualize_game_graph(
                    data_graph,
                    game['id'],
                    window_size=window_size,
                    output_path=output_path,
                    edge_sample_rate=0.05,
                )
                plt.close('all')

                # Force garbage collection
                gc.collect()

def process_shift_data(data_graph, verbose, team_game_map, shift_data):
    # called on a per-game basis
    game_id = shift_data["game_id"]
    game_date = shift_data["game_date"]
    away_team = shift_data["away_teams"]
    home_team = shift_data["home_teams"]
    period_id = shift_data["period_id"]
    period_code = shift_data["period_code"]
    time_index = shift_data["time_index"]
    time_on_ice = shift_data["time_on_ice"]
    event_id = shift_data["event_id"]
    shift_id = shift_data["shift_id"]
    away_empty_net = shift_data["away_empty_net"]
    home_empty_net = shift_data["home_empty_net"]
    away_skaters = shift_data["away_skaters"]
    home_skaters = shift_data["home_skaters"]
    player_data = shift_data["player_data"]


    for i, shift in enumerate(shift_id):
        # one per shift
        team_map = {}
        line_player_team_map = {}
        for player_dat in player_data[i]:
            if player_dat['player_team'] not in team_map:
                team_map[player_dat['player_team']] = (game_id[i], player_dat['player_team'])
            game_team = (game_id[i], player_dat['player_team'])
            if game_team not in line_player_team_map:
                line_player_team_map[game_team] = []
            line_player_team_map[game_team].append(player_dat['player_id'])

        for team in line_player_team_map:
            other_players = copy.deepcopy(line_player_team_map[team])
            for j, player_dat in enumerate(player_data[i]):
                if player_dat['player_id'] not in line_player_team_map[team]:
                    continue
                other_players.remove(player_dat['player_id'])
                team_tgp = str(team[0]) + '_' + team[1]
                player_pgp = str(team[0]) + '_' + str(player_dat['player_id'])
                # if player_data[i][j]['goal'][period_code[i]] == 1:
                #     print(j, player_dat['player_id'])
                #     print(line_player_team_map[team])
                #     print(f"{i}:{j}:{player_data[i][j]['player_team']}:{player_data[i][j]['goal']}")
                #     print('\n')
                update_tgp_stats(data_graph, team_tgp, period_code[i], player_dat)
                update_pgp_stats(data_graph, player_pgp, period_code[i], player_dat)
                for k, other in  enumerate(other_players):
                    other_pgp = str(team[0]) + '_' + str(other)
                    update_pgp_edge_stats(data_graph, player_pgp, other_pgp, period_code[i], player_dat)


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

