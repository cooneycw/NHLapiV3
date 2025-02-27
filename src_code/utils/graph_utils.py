from multiprocessing import Pool
from functools import partial
from collections import defaultdict
from datetime import datetime
import copy
import gc
import networkx as nx


def create_graph():
    """
    Create a NetworkX graph based on dictionary/list structures:
      - Add each team node from 'team_info'
      - Add each player node for each team in 'team_players'
      - Add edges (team->player) for every team
      - Add a single team-to-team edge with match details
    """
    return nx.Graph()


def add_team_node(graph, team):
    graph.add_node(team, type = 'team')


def add_player_node(graph, player, player_dict):
    player_dict_details = player_dict[player]
    player_dict_details['games_played'] = 0
    player_dict_details['toi'] = [0, 0, 0]
    player_dict_details['faceoff_taken'] = [0, 0, 0]
    player_dict_details['faceoff_won'] = [0, 0, 0]
    player_dict_details['shot_attempt'] = [0, 0, 0]
    player_dict_details['shot_missed'] = [0, 0, 0]
    player_dict_details['shot_blocked'] = [0, 0, 0]
    player_dict_details['shot_on_goal'] = [0, 0, 0]
    player_dict_details['shot_saved'] = [0, 0, 0]
    player_dict_details['shot_missed_shootout'] = [0, 0, 0]
    player_dict_details['goal'] = [0, 0, 0]
    player_dict_details['goal_against'] = [0, 0, 0]
    player_dict_details['giveaways'] = [0, 0, 0]
    player_dict_details['takeaway'] = [0, 0, 0]
    player_dict_details['hit_another_player'] = [0, 0, 0]
    player_dict_details['hit_by_player'] = [0, 0, 0]
    player_dict_details['penalties'] = [0, 0, 0]
    player_dict_details['penalties_served'] = [0, 0, 0]
    player_dict_details['penalties_drawn'] = [0, 0, 0]
    player_dict_details['penalty_shot'] = [0, 0, 0]
    player_dict_details['penalty_shot_goal'] = [0, 0, 0]
    player_dict_details['penalty_shot_saved'] = [0, 0, 0]
    player_dict_details['penalties_duration'] = [0, 0, 0]

    graph.add_node(player, type = 'player', **player_dict_details)


def add_game(graph, game):
    """
    Create game nodes and edges, maintaining indices for quick lookups.
    """
    default_stats = {
        'win': [0, 0, 0],
        'loss': [0, 0, 0],
        'faceoff_taken': [0, 0, 0],
        'faceoff_won': [0, 0, 0],
        'shot_attempt': [0, 0, 0],
        'shot_missed': [0, 0, 0],
        'shot_blocked': [0, 0, 0],
        'shot_on_goal': [0, 0, 0],
        'shot_saved': [0, 0, 0],
        'shot_missed_shootout': [0, 0, 0],
        'goal': [0, 0, 0],
        'goal_against': [0, 0, 0],
        'giveaways': [0, 0, 0],
        'takeaways': [0, 0, 0],
        'hit_another_player': [0, 0, 0],
        'hit_by_player': [0, 0, 0],
        'penalties': [0, 0, 0],
        'penalties_served': [0, 0, 0],
        'penalties_drawn': [0, 0, 0],
        'penalty_shot': [0, 0, 0],
        'penalty_shot_goal': [0, 0, 0],
        'penalty_shot_saved': [0, 0, 0],
        'penalties_duration': [0, 0, 0],
    }

    # Initialize indices if they don't exist
    if 'team_games' not in graph.graph:
        graph.graph['team_games'] = defaultdict(list)
    if 'sorted_games' not in graph.graph:
        graph.graph['sorted_games'] = []

    game_id = game['id']
    game_date = game['game_date']
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%Y-%m-%d')

    # Add game node
    graph.add_node(game_id, type='game', game_date=game_date,
                   home_team=game['homeTeam'], away_team=game['awayTeam'])

    # Add team-game edges
    graph.add_edge(game['awayTeam'], game_id, home=0)
    graph.add_edge(game['homeTeam'], game_id, home=1)

    # Create TGP nodes
    away_tgp = f"{game_id}_{game['awayTeam']}"
    home_tgp = f"{game_id}_{game['homeTeam']}"

    away_stats = copy.deepcopy(default_stats)
    home_stats = copy.deepcopy(default_stats)

    graph.add_node(away_tgp, type='team_game_performance', home=0, **away_stats)
    graph.add_node(home_tgp, type='team_game_performance', home=1, **home_stats)
    graph.add_edge(away_tgp, game_id)
    graph.add_edge(home_tgp, game_id)

    # Update indices
    graph.graph['team_games'][game['awayTeam']].append({
        'game_id': game_id,
        'tgp_node': away_tgp,
        'date': game_date,
        'home': 0
    })

    graph.graph['team_games'][game['homeTeam']].append({
        'game_id': game_id,
        'tgp_node': home_tgp,
        'date': game_date,
        'home': 1
    })

    graph.graph['sorted_games'].append({
        'game_id': game_id,
        'date': game_date,
        'home_team': game['homeTeam'],
        'away_team': game['awayTeam']
    })
    graph.graph['sorted_games'].sort(key=lambda x: x['date'])


def process_games_chronologically(data_graph, data_games):
    """
    Process games in chronological order to ensure accurate days_since_last_game calculations.

    Args:
        data_graph: NetworkX graph to update
        data_games: List of game dictionaries

    Returns:
        bool: True if processing was successful
    """
    # First validate the dates
    dates_valid, issues = validate_game_dates(data_games)
    if not dates_valid:
        print("Found issues with game dates:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nProceeding with chronological processing anyway...")

    # Create a sorted copy of games
    sorted_games = sorted(data_games,
                          key=lambda x: datetime.strptime(x['game_date'], '%Y-%m-%d')
                          if isinstance(x['game_date'], str)
                          else x['game_date'])

    # Process games in chronological order
    for game in sorted_games:
        add_game(data_graph, game)
        update_days_since_last_game(data_graph, game['id'])

    return True


def validate_game_dates(data_games):
    """
    Validate that all game dates can be parsed and identify any ordering issues.

    Args:
        data_games: List of game dictionaries

    Returns:
        tuple: (bool indicating if dates are valid, list of any parsing/ordering issues found)
    """
    issues = []
    parsed_dates = []

    for game in data_games:
        game_id = game['id']
        game_date = game['game_date']

        try:
            if isinstance(game_date, str):
                parsed_date = datetime.strptime(game_date, '%Y-%m-%d')
            else:
                parsed_date = game_date
            parsed_dates.append((game_id, parsed_date))
        except ValueError as e:
            issues.append(f"Game {game_id}: Unable to parse date '{game_date}' - {str(e)}")

    # Check if games are in chronological order in the input
    sorted_dates = sorted(parsed_dates, key=lambda x: x[1])
    if parsed_dates != sorted_dates:
        issues.append("Warning: Games in data_games are not in chronological order")
        # Find specific out-of-order examples
        for i in range(len(parsed_dates) - 1):
            if parsed_dates[i][1] > parsed_dates[i + 1][1]:
                issues.append(f"Game {parsed_dates[i][0]} ({parsed_dates[i][1]}) comes before "
                              f"Game {parsed_dates[i + 1][0]} ({parsed_dates[i + 1][1]}) but has a later date")

    return len(issues) == 0, issues


def update_days_since_last_game(graph, game_id):
    """
    Update the days since last game for Team Game Performance (TGP) nodes,
    considering all previous games chronologically.

    Args:
        graph: NetworkX graph containing game and TGP data
        game_id: ID of the current game
    """
    game_node = graph.nodes[game_id]
    game_date = game_node['game_date']
    home_team = game_node['home_team']
    away_team = game_node['away_team']

    # Get TGP nodes for current game
    home_tgp = f"{game_id}_{home_team}"
    away_tgp = f"{game_id}_{away_team}"

    # Convert game_date to datetime if it's a string
    if isinstance(game_date, str):
        current_game_date = datetime.strptime(game_date, '%Y-%m-%d')
    else:
        current_game_date = game_date

    # Find last game for each team
    for team, tgp in [(home_team, home_tgp), (away_team, away_tgp)]:
        last_game_date = None

        # Look through all TGP nodes
        team_games = []
        for node in graph.nodes():
            # Check if it's a TGP node
            if isinstance(node, str) and graph.nodes[node].get('type') == 'team_game_performance':
                node_parts = node.split('_')
                # Verify it's for this team and not the current game
                if len(node_parts) == 2 and node_parts[1] == team and node != tgp:
                    try:
                        node_game_id = int(node_parts[0])
                        if node_game_id != game_id:
                            game_date = graph.nodes[node_game_id]['game_date']
                            if isinstance(game_date, str):
                                game_date = datetime.strptime(game_date, '%Y-%m-%d')
                            team_games.append((node_game_id, game_date))
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Error processing node {node}: {str(e)}")

        # Sort games by date to find the most recent one before current game
        team_games.sort(key=lambda x: x[1])
        previous_games = [g for g in team_games if g[1] < current_game_date]

        if previous_games:
            last_game_date = previous_games[-1][1]
            days_diff = min((current_game_date - last_game_date).days, 30)
        else:
            days_diff = 30  # Maximum value for first game

        # Add the days_since_last_game to the TGP node
        graph.nodes[tgp]['days_since_last_game'] = days_diff


def add_player_game_performance(graph, roster):
    default_stats = {
        'toi': [0, 0, 0],
        'faceoff_taken': [0, 0, 0],
        'faceoff_won': [0, 0, 0],
        'shot_attempt': [0, 0, 0],
        'shot_missed': [0, 0, 0],
        'shot_blocked': [0, 0, 0],
        'shot_on_goal': [0, 0, 0],
        'shot_saved': [0, 0, 0],
        'shot_missed_shootout': [0, 0, 0],
        'goal': [0, 0, 0],
        'assist': [0, 0, 0],
        'point': [0, 0, 0],
        'goal_against': [0, 0, 0],
        'giveaways': [0, 0, 0],
        'takeaways': [0, 0, 0],
        'hit_another_player': [0, 0, 0],
        'hit_by_player': [0, 0, 0],
        'penalties': [0, 0, 0],
        'penalties_served': [0, 0, 0],
        'penalties_drawn': [0, 0, 0],
        'penalty_shot': [0, 0, 0],
        'penalty_shot_goal': [0, 0, 0],
        'penalty_shot_saved': [0, 0, 0],
        'penalties_duration': [0, 0, 0],
    }

    team_game_map = {}

    for player in roster:
        game_id = player['game_id']
        player_id = player['player_id']
        player_team = player['player_team']
        player_position = player['player_position']
        player_pgp = str(game_id) + '_' + str(player_id)
        team_tgp = str(game_id) + '_' + player_team
        stat_node_copy = copy.deepcopy(default_stats)
        edge_node_copy = copy.deepcopy(default_stats)
        graph.add_node(player_pgp, type = 'player_game_performance', player_position=player_position, **stat_node_copy)
        graph.add_edge(player_pgp, team_tgp, type = 'player_game_performance_team_game_performance_edge')
        graph.add_edge(player_pgp, player_id, type= 'player_game_performance_player_edge', **edge_node_copy)

        team_key = (game_id, player_team)
        if team_key not in team_game_map:
            team_game_map[team_key] = []
        team_game_map[team_key].append(player_pgp)

    # Add edges between players on the same team
    for team_key, player_pgps in team_game_map.items():
        for i in range(len(player_pgps)):
            for j in range(i + 1, len(player_pgps)):
                stat_copy = copy.deepcopy(default_stats)
                graph.add_edge(player_pgps[i], player_pgps[j], **stat_copy)

    return team_game_map


def update_tgp_stats(data_graph, team_tgp, period_code, player_dat):
    """
    Update team game performance statistics in the graph.

    Args:
        data_graph: The NetworkX graph containing all game data
        team_tgp: The ID of the team's game performance node
        period_code: Period code (0 for regulation, 1 for OT, 2 for shootout)
        player_dat: Dictionary containing player statistics
    """
    if 'hist_game_count' not in data_graph.nodes[team_tgp]:
        data_graph.nodes[team_tgp]['hist_game_count'] = [0, 0, 0]

    # Increment game count only once per period type
    # Only set to 1 if not already set (avoid counting the same period multiple times)
    if data_graph.nodes[team_tgp]['hist_game_count'][period_code] == 0:
        data_graph.nodes[team_tgp]['hist_game_count'][period_code] = 1

    # Update team statistics for the current period
    for stat in ['goal', 'goal_against', 'faceoff_taken', 'faceoff_won',
                 'shot_attempt', 'shot_missed', 'shot_blocked', 'shot_on_goal',
                 'shot_saved', 'hit_another_player', 'hit_by_player',
                 'giveaways', 'takeaways', 'penalties_duration']:
        if stat in player_dat:
            if stat not in data_graph.nodes[team_tgp]:
                data_graph.nodes[team_tgp][stat] = [0, 0, 0]

            data_graph.nodes[team_tgp][stat][period_code] += player_dat[stat][period_code]

    # Update time on ice separately
    if 'toi' in player_dat:
        if 'toi' not in data_graph.nodes[team_tgp]:
            data_graph.nodes[team_tgp]['toi'] = [0, 0, 0]

        data_graph.nodes[team_tgp]['toi'][period_code] += player_dat['toi'][period_code]


def update_pgp_stats(data_graph, player_pgp, period_code, player_dat):
    """
    Update player game performance statistics in the graph.

    Args:
        data_graph: The NetworkX graph containing all game data
        player_pgp: The ID of the player's game performance node
        period_code: Period code (0 for regulation, 1 for OT, 2 for shootout)
        player_dat: Dictionary containing player statistics
    """
    # Initialize hist_game_count if it doesn't exist
    if 'hist_game_count' not in data_graph.nodes[player_pgp]:
        data_graph.nodes[player_pgp]['hist_game_count'] = [0, 0, 0]

    # Increment game count only once per period type
    # Only set to 1 if not already set (avoid counting the same period multiple times)
    if data_graph.nodes[player_pgp]['hist_game_count'][period_code] == 0:
        data_graph.nodes[player_pgp]['hist_game_count'][period_code] = 1

    # Copy player position if available
    if 'player_position' in player_dat and 'player_position' not in data_graph.nodes[player_pgp]:
        data_graph.nodes[player_pgp]['player_position'] = player_dat['player_position']

    # Update player statistics for the current period
    for stat in ['toi', 'goal', 'assist', 'point', 'goal_against', 'faceoff_taken',
                 'faceoff_won', 'shot_attempt', 'shot_missed', 'shot_blocked',
                 'shot_on_goal', 'shot_saved', 'hit_another_player', 'hit_by_player',
                 'giveaways', 'takeaways', 'penalties', 'penalties_served',
                 'penalties_drawn', 'penalties_duration', 'penalty_shot',
                 'penalty_shot_goal', 'penalty_shot_saved']:
        if stat in player_dat:
            if stat not in data_graph.nodes[player_pgp]:
                data_graph.nodes[player_pgp][stat] = [0, 0, 0]

            data_graph.nodes[player_pgp][stat][period_code] += player_dat[stat][period_code]


def update_pgp_edge_stats(data_graph, player1_pgp, player2_pgp, period_code, player_dat):
    """
    Update player-to-player edge statistics in the graph for a specific game period.

    Args:
        data_graph: The NetworkX graph containing all game data
        player1_pgp: The ID of the first player's game performance node
        player2_pgp: The ID of the second player's game performance node
        period_code: Period code (0 for regulation, 1 for OT, 2 for shootout)
        player_dat: Dictionary containing player statistics
    """
    if not data_graph.has_edge(player1_pgp, player2_pgp):
        data_graph.add_edge(player1_pgp, player2_pgp, type='player_game_performance_edge')

        # Initialize statistics arrays [reg, ot, so]
        for stat in ['toi', 'goal', 'faceoff_taken', 'faceoff_won', 'shot_on_goal',
                     'shot_saved', 'hit_another_player', 'hit_by_player', 'penalties_duration']:
            data_graph.edges[player1_pgp, player2_pgp][stat] = [0, 0, 0]

        # Initialize historical game count as [reg, ot, so] array
        data_graph.edges[player1_pgp, player2_pgp]['hist_games_played'] = [0, 0, 0]

    # Update edge statistics for the current period
    if 'toi' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['toi'][period_code] += player_dat['toi'][period_code]

    if 'goal' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['goal'][period_code] += player_dat['goal'][period_code]

    if 'faceoff_taken' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['faceoff_taken'][period_code] += player_dat['faceoff_taken'][
            period_code]

    if 'faceoff_won' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['faceoff_won'][period_code] += player_dat['faceoff_won'][period_code]

    if 'shot_on_goal' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['shot_on_goal'][period_code] += player_dat['shot_on_goal'][
            period_code]

    if 'shot_saved' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['shot_saved'][period_code] += player_dat['shot_saved'][period_code]

    if 'hit_another_player' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['hit_another_player'][period_code] += \
        player_dat['hit_another_player'][period_code]

    if 'hit_by_player' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['hit_by_player'][period_code] += player_dat['hit_by_player'][
            period_code]

    if 'penalties_duration' in player_dat:
        data_graph.edges[player1_pgp, player2_pgp]['penalties_duration'][period_code] += \
        player_dat['penalties_duration'][period_code]

    # Track game participation for this period
    if 'hist_games_played' not in data_graph.edges[player1_pgp, player2_pgp]:
        data_graph.edges[player1_pgp, player2_pgp]['hist_games_played'] = [0, 0, 0]

    # Increment game count only once per period type
    # Only set to 1 if not already set (avoid counting the same period multiple times)
    if data_graph.edges[player1_pgp, player2_pgp]['hist_games_played'][period_code] == 0:
        data_graph.edges[player1_pgp, player2_pgp]['hist_games_played'][period_code] = 1



def create_pgp_edges(graph, team_tgp):
    """
    Create edges between PGP nodes of players on the same team
    with their aggregate stats.
    """
    pgp_nodes = [node for node in graph.neighbors(team_tgp)
                 if graph.nodes[node]['type'] == 'player_game_performance']

    default_stats = {
        'toi': [0, 0, 0],
        'faceoff_taken': [0, 0, 0],
        'faceoff_won': [0, 0, 0],
        'shot_on_goal': [0, 0, 0],
        'shot_saved': [0, 0, 0],
        'goal': [0, 0, 0],
        'hit_another_player': [0, 0, 0],
        'hit_by_player': [0, 0, 0],
        'penalties_duration': [0, 0, 0],
    }
    for i, pgp1 in enumerate(pgp_nodes):
        for pgp2 in pgp_nodes[i+1:]:
            stat_copy = copy.deepcopy(default_stats)
            # Aggregate stats for the edge
            # Add edge with aggregate stats
            graph.add_edge(pgp1, pgp2, type='pgp_pgp_edge', **stat_copy)


def update_game_outcome(graph, game_id, game):
    """
    Update win/loss stats for both teams after game is complete.
    Called once per game after all periods are processed.

    Args:
        graph: NetworkX graph containing game data
        game_id: ID of the current game
        game: Game data dictionary containing period information
    """
    # Get both teams' TGP nodes
    home_team = game['homeTeam']
    away_team = game['awayTeam']
    home_tgp = f"{game_id}_{home_team}"
    away_tgp = f"{game_id}_{away_team}"

    home_node = graph.nodes[home_tgp]
    away_node = graph.nodes[away_tgp]

    # Sum goals across all periods
    home_goals = sum(home_node['goal'])
    away_goals = sum(away_node['goal'])

    # Determine which period the game ended in
    # Index 0: regulation, 1: overtime, 2: shootout
    period_index = 0
    if (home_node['goal'][1] > 0) or (away_node['goal'][1] > 0):
        period_index = 1
    elif (home_node['goal'][2] > 0) or (away_node['goal'][2] > 0):
        period_index = 2

    # Update win/loss arrays - only one element will be 1
    if home_goals > away_goals:
        home_node['win'] = [0] * 3
        home_node['loss'] = [0] * 3
        away_node['win'] = [0] * 3
        away_node['loss'] = [0] * 3
        home_node['win'][period_index] = 1
        away_node['loss'][period_index] = 1
        home_node['valid'] = True
        away_node['valid'] = True
    elif home_goals < away_goals:
        home_node['win'] = [0] * 3
        home_node['loss'] = [0] * 3
        away_node['win'] = [0] * 3
        away_node['loss'] = [0] * 3
        away_node['win'][period_index] = 1
        home_node['loss'][period_index] = 1
        home_node['valid'] = True
        away_node['valid'] = True
    else:
        home_node['win'] = [0] * 3
        home_node['loss'] = [0] * 3
        away_node['win'] = [0] * 3
        away_node['loss'] = [0] * 3
        home_node['valid'] = False
        away_node['valid'] = False


def get_historical_tgp_stats(data_graph, team_tgp, window_size, config):
    """
    Get historical statistics for a team game performance node.

    Args:
        data_graph: The NetworkX graph containing all game data
        team_tgp: The ID of the team's game performance node
        window_size: Size of the historical window to consider
        config: Configuration object with stat attributes

    Returns:
        Dictionary containing historical statistics
    """
    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_game_count': [0, 0, 0]
    }

    # Initialize stat arrays
    for stat in config.stat_attributes['team_stats']:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Get the game ID and team from the team_tgp ID
    current_game_id, team = team_tgp.split('_', 1)

    # Find historical team game performances for this team
    historical_tgps = []
    for node in data_graph.nodes():
        if data_graph.nodes[node].get('type') == 'team_game_performance':
            node_game_id, node_team = node.split('_', 1)
            if node_team == team and node != team_tgp:
                historical_tgps.append(node)

    # Sort by game date
    historical_tgps.sort(key=lambda x: data_graph.nodes[x].get('game_date', ''))

    # Get the most recent games up to the window size
    recent_tgps = historical_tgps[-window_size:] if len(historical_tgps) > window_size else historical_tgps

    # Aggregate statistics
    for tgp in recent_tgps:
        node_data = data_graph.nodes[tgp]

        # Add game counts for each period type
        if 'hist_game_count' in node_data:
            for i in range(3):  # For each period type (reg, ot, so)
                historical_stats[f'hist_{window_size}_game_count'][i] += node_data['hist_game_count'][i]

        # Add stats for each type
        for stat in config.stat_attributes['team_stats']:
            if stat in node_data:
                for i in range(3):  # For each period type
                    historical_stats[f'hist_{window_size}_{stat}'][i] += node_data[stat][i]

    return historical_stats


def get_historical_pgp_stats(data_graph, player_pgp, window_size, config):
    """
    Get historical statistics for a player game performance node.

    Args:
        data_graph: The NetworkX graph containing all game data
        player_pgp: The ID of the player's game performance node
        window_size: Size of the historical window to consider
        config: Configuration object with stat attributes

    Returns:
        Dictionary containing historical statistics
    """
    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_game_count': [0, 0, 0]
    }

    # Initialize stat arrays
    for stat in config.stat_attributes['player_stats']:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Get the game ID from the player_pgp ID
    current_game_id = player_pgp.split('_')[0]
    player_id = player_pgp.split('_')[1]

    # Find historical player game performances for this player
    historical_pgps = []
    for node in data_graph.nodes():
        if data_graph.nodes[node].get('type') == 'player_game_performance':
            if node.endswith(f'_{player_id}') and node != player_pgp:
                historical_pgps.append(node)

    # Sort by game date
    historical_pgps.sort(key=lambda x: data_graph.nodes[x].get('game_date', ''))

    # Get the most recent games up to the window size
    recent_pgps = historical_pgps[-window_size:] if len(historical_pgps) > window_size else historical_pgps

    # Aggregate statistics
    for pgp in recent_pgps:
        node_data = data_graph.nodes[pgp]

        # Add game counts for each period type
        if 'hist_game_count' in node_data:
            for i in range(3):  # For each period type (reg, ot, so)
                historical_stats[f'hist_{window_size}_game_count'][i] += node_data['hist_game_count'][i]

        # Add stats for each type
        for stat in config.stat_attributes['player_stats']:
            if stat in node_data:
                for i in range(3):  # For each period type
                    historical_stats[f'hist_{window_size}_{stat}'][i] += node_data[stat][i]

    return historical_stats


def get_historical_pgp_edge_stats(data_graph, player1_pgp, player2_pgp, window_size, config):
    """
    Get historical statistics for a player-to-player edge.

    Args:
        data_graph: The NetworkX graph containing all game data
        player1_pgp: The ID of the first player's game performance node
        player2_pgp: The ID of the second player's game performance node
        window_size: Size of the historical window to consider
        config: Configuration object with stat attributes

    Returns:
        Dictionary containing historical statistics
    """
    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_games_played': [0, 0, 0]
    }

    # Initialize stat arrays
    for stat in config.stat_attributes['player_pair_stats']:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Get the player IDs from the pgp IDs
    game1_id, player1_id = player1_pgp.split('_', 1)
    game2_id, player2_id = player2_pgp.split('_', 1)

    if game1_id != game2_id:
        return historical_stats  # Different games, no historical data to aggregate

    # Find historical edges between these players
    historical_edges = []
    for u, v, data in data_graph.edges(data=True):
        if data['type'] == 'player_game_performance_edge':
            u_game, u_player = u.split('_', 1)
            v_game, v_player = v.split('_', 1)

            if u_game == v_game and u_game != game1_id:  # Same game but not current game
                if (u_player == player1_id and v_player == player2_id) or \
                        (u_player == player2_id and v_player == player1_id):
                    historical_edges.append((u, v))

    # Sort by game date (assuming node attributes have game_date)
    historical_edges.sort(key=lambda x:
    data_graph.nodes[x[0]].get('game_date', '') if 'game_date' in data_graph.nodes[x[0]]
    else '')

    # Get the most recent edges up to the window size
    recent_edges = historical_edges[-window_size:] if len(historical_edges) > window_size else historical_edges

    # Aggregate statistics
    for u, v in recent_edges:
        edge_data = data_graph.get_edge_data(u, v)

        # Add game counts for each period type
        if 'hist_games_played' in edge_data:
            for i in range(3):  # For each period type (reg, ot, so)
                historical_stats[f'hist_{window_size}_games_played'][i] += edge_data['hist_games_played'][i]

        # Add stats for each type
        for stat in config.stat_attributes['player_pair_stats']:
            if stat in edge_data:
                for i in range(3):  # For each period type
                    historical_stats[f'hist_{window_size}_{stat}'][i] += edge_data[stat][i]

    return historical_stats


def get_subgraph_dict(data_graph, game_info, stat_window_sizes):
    """
    Extract minimal subgraph data needed for historical calculations of a game.

    Args:
        data_graph: The full NetworkX graph
        game_info: Dictionary containing game metadata
        config: Configuration object containing window sizes

    Returns:
        Dictionary containing subgraph data including nodes, edges, and team games
    """
    game_id = game_info['game_id']
    game_date = game_info['date']
    home_team = game_info['home_team']
    away_team = game_info['away_team']

    subgraph_nodes = {}
    subgraph_edges = {}

    # Add current game's nodes with their attributes
    subgraph_nodes[game_id] = dict(data_graph.nodes[game_id])
    subgraph_nodes[f"{game_id}_{home_team}"] = dict(data_graph.nodes[f"{game_id}_{home_team}"])
    subgraph_nodes[f"{game_id}_{away_team}"] = dict(data_graph.nodes[f"{game_id}_{away_team}"])

    # Add current game's PGP nodes first to ensure they're included
    for team_id in [home_team, away_team]:
        tgp_node = f"{game_id}_{team_id}"
        current_pgps = [
            node for node in data_graph.neighbors(tgp_node)
            if isinstance(node, str) and
               data_graph.nodes[node].get('type') == 'player_game_performance'
        ]

        # Add current PGP nodes and their connections
        for pgp in current_pgps:
            subgraph_nodes[pgp] = dict(data_graph.nodes[pgp])

            # Add player nodes
            player_nodes = [
                node for node in data_graph.neighbors(pgp)
                if data_graph.nodes[node].get('type') == 'player'
            ]
            for player in player_nodes:
                subgraph_nodes[player] = dict(data_graph.nodes[player])

            # Add edges between current PGPs
            for i, pgp1 in enumerate(current_pgps):
                for pgp2 in current_pgps[i + 1:]:
                    if data_graph.has_edge(pgp1, pgp2):
                        edge_key = tuple(sorted([pgp1, pgp2]))
                        subgraph_edges[edge_key] = dict(data_graph.get_edge_data(*edge_key))

    # Get historical games within max window for both teams
    max_window = max(stat_window_sizes)
    relevant_team_games = {team_id: [] for team_id in [home_team, away_team]}

    for team_id in [home_team, away_team]:
        team_games = data_graph.graph['team_games'].get(team_id, [])

        # Filter to games before current game
        historical_games = [
            g for g in team_games
            if g['game_id'] != game_id and
               (isinstance(g['date'], str) and
                datetime.strptime(g['date'], '%Y-%m-%d') < game_date or
                g['date'] < game_date)
        ]

        # Sort by date and take most recent max_window games
        historical_games.sort(key=lambda x: x['date'], reverse=True)
        relevant_games = historical_games[:max_window]
        relevant_team_games[team_id] = relevant_games

        # Add historical game nodes and their connections
        for game in relevant_games:
            hist_game_id = game['game_id']
            tgp_node = game['tgp_node']

            subgraph_nodes[hist_game_id] = dict(data_graph.nodes[hist_game_id])
            subgraph_nodes[tgp_node] = dict(data_graph.nodes[tgp_node])

            # Get historical PGP nodes
            pgp_nodes = [
                node for node in data_graph.neighbors(tgp_node)
                if isinstance(node, str) and
                   data_graph.nodes[node].get('type') == 'player_game_performance'
            ]

            # Add historical PGP nodes and their connections
            for pgp in pgp_nodes:
                subgraph_nodes[pgp] = dict(data_graph.nodes[pgp])

                # Add player nodes
                player_nodes = [
                    node for node in data_graph.neighbors(pgp)
                    if data_graph.nodes[node].get('type') == 'player'
                ]
                for player in player_nodes:
                    subgraph_nodes[player] = dict(data_graph.nodes[player])

                # Add edges between historical PGPs
                for i, pgp1 in enumerate(pgp_nodes):
                    for pgp2 in pgp_nodes[i + 1:]:
                        if data_graph.has_edge(pgp1, pgp2):
                            edge_key = tuple(sorted([pgp1, pgp2]))
                            if edge_key not in subgraph_edges:  # Avoid overwriting existing edges
                                subgraph_edges[edge_key] = dict(data_graph.get_edge_data(*edge_key))

    return {
        'nodes': subgraph_nodes,
        'edges': subgraph_edges,
        'team_games': relevant_team_games,
        'game_info': game_info
    }


def process_single_game(subgraph_data, stat_window_sizes, stat_attributes):
    """
    Process historical statistics for a single game using subgraph data.

    Args:
        subgraph_data: Dictionary containing subgraph information
        config: Configuration object containing window sizes and stat attributes

    Returns:
        Dictionary containing updates for nodes and edges
    """
    # Reconstruct subgraph from dictionary data
    subgraph = nx.Graph()

    # Add nodes with their attributes
    for node, attrs in subgraph_data['nodes'].items():
        subgraph.add_node(node, **attrs)

    # Add edges with their attributes
    for (node1, node2), attrs in subgraph_data['edges'].items():
        subgraph.add_edge(node1, node2, **attrs)

    # Add necessary graph attributes
    subgraph.graph['team_games'] = subgraph_data['team_games']

    game_info = subgraph_data['game_info']
    game_id = game_info['game_id']
    home_team = game_info['home_team']
    away_team = game_info['away_team']

    # Initialize updates dictionary
    updates = {
        'nodes': {},
        'edges': {}
    }

    # Process historical stats for both teams
    for team_id in [home_team, away_team]:
        team_tgp = f"{game_id}_{team_id}"
        current_pgps = [
            node for node in subgraph.nodes
            if subgraph.nodes[node].get('type') == 'player_game_performance'
        ]

        # Initialize updates for team TGP node
        updates['nodes'][team_tgp] = {}

        # Process stats for each window size
        for window in stat_window_sizes:
            # 1. Process team stats
            team_stats, team_game_count = get_historical_tgp_stats(
                subgraph, team_id, game_id, stat_attributes, window
            )

            if team_stats:
                updates['nodes'][team_tgp][f'hist_{window}_game_count'] = team_game_count
                for stat_name, values in team_stats.items():
                    hist_name = f'hist_{window}_{stat_name}'
                    updates['nodes'][team_tgp][hist_name] = values

            # 2. Process player stats
            player_stats, player_game_counts = get_historical_pgp_stats(
                subgraph, team_id, game_id, stat_attributes, window
            )

            if player_stats:
                for pgp in current_pgps:
                    # Extract player ID directly from PGP node (format: "game_id_player_id")
                    player_id = int(pgp.split('_')[1])

                    if player_id in player_stats:
                        if pgp not in updates['nodes']:
                            updates['nodes'][pgp] = {}

                        # Add game count
                        updates['nodes'][pgp][f'hist_{window}_game_count'] = player_game_counts[player_id]

                        # Add stats
                        for stat_name, values in player_stats[player_id].items():
                            hist_name = f'hist_{window}_{stat_name}'
                            updates['nodes'][pgp][hist_name] = values

            # 3. Process player-pair stats
            pair_stats, pair_game_counts = get_historical_pgp_edge_stats(
                subgraph, team_id, game_id, stat_attributes, window
            )

            if pair_stats:
                for i, pgp1 in enumerate(current_pgps):
                    for pgp2 in current_pgps[i + 1:]:
                        if subgraph.has_edge(pgp1, pgp2):
                            # Extract player IDs directly from PGP nodes
                            player1_id = int(pgp1.split('_')[1])
                            player2_id = int(pgp2.split('_')[1])

                            pair_key = tuple(sorted([player1_id, player2_id]))
                            if pair_key in pair_stats:
                                edge_key = tuple(sorted([pgp1, pgp2]))
                                if edge_key not in updates['edges']:
                                    updates['edges'][edge_key] = {}

                                # Add game count
                                updates['edges'][edge_key][f'hist_{window}_game_count'] = pair_game_counts[pair_key]

                                # Add stats
                                for stat_name, values in pair_stats[pair_key].items():
                                    hist_name = f'hist_{window}_{stat_name}'
                                    updates['edges'][edge_key][hist_name] = values

    return updates


def process_game_wrapper(args):
    """Wrapper function to unpack arguments for process_single_game."""
    subgraph_data, stat_window_sizes, stat_attributes = args
    return process_single_game(subgraph_data, stat_window_sizes, stat_attributes)


def calculate_historical_stats(config, data_graph):
    """
    Calculate historical statistics (averages only) for all nodes and edges in the graph.

    Args:
        config: Configuration object with stat window sizes and attributes
        data_graph: The NetworkX graph containing all game data

    Returns:
        The updated graph with historical statistics (averages only)
    """
    print("Calculating historical statistics...")

    # For each window size
    for window_size in config.stat_window_sizes:
        print(f"Processing window size {window_size}...")

        # Process team game performance nodes
        tgp_nodes = [n for n in data_graph.nodes() if data_graph.nodes[n].get('type') == 'team_game_performance']
        for tgp in tgp_nodes:
            # Get raw historical stats (we'll only keep game counts and calculate averages)
            raw_hist_stats = get_historical_tgp_stats(data_graph, tgp, window_size, config)

            # Store game counts
            data_graph.nodes[tgp][f'hist_{window_size}_game_count'] = raw_hist_stats[f'hist_{window_size}_game_count']

            # Calculate and store averages (replacing raw values)
            game_counts = raw_hist_stats[f'hist_{window_size}_game_count']
            for stat in config.stat_attributes['team_stats']:
                stat_key = f'hist_{window_size}_{stat}'
                if stat_key in raw_hist_stats:
                    raw_values = raw_hist_stats[stat_key]
                    avg_values = [0, 0, 0]

                    # Calculate average for each period type, avoiding division by zero
                    for i in range(3):
                        if game_counts[i] > 0:
                            avg_values[i] = raw_values[i] / game_counts[i]

                    # Store only the averages, no raw totals
                    data_graph.nodes[tgp][stat_key] = avg_values

        # Process player game performance nodes
        pgp_nodes = [n for n in data_graph.nodes() if data_graph.nodes[n].get('type') == 'player_game_performance']
        for pgp in pgp_nodes:
            # Get raw historical stats
            raw_hist_stats = get_historical_pgp_stats(data_graph, pgp, window_size, config)

            # Store game counts
            data_graph.nodes[pgp][f'hist_{window_size}_game_count'] = raw_hist_stats[f'hist_{window_size}_game_count']

            # Calculate and store averages (replacing raw values)
            game_counts = raw_hist_stats[f'hist_{window_size}_game_count']
            for stat in config.stat_attributes['player_stats']:
                stat_key = f'hist_{window_size}_{stat}'
                if stat_key in raw_hist_stats:
                    raw_values = raw_hist_stats[stat_key]
                    avg_values = [0, 0, 0]

                    # Calculate average for each period type, avoiding division by zero
                    for i in range(3):
                        if game_counts[i] > 0:
                            avg_values[i] = raw_values[i] / game_counts[i]

                    # Store only the averages, no raw totals
                    data_graph.nodes[pgp][stat_key] = avg_values

        # Process player-to-player edges
        pgp_edges = [(u, v) for u, v, d in data_graph.edges(data=True)
                     if d.get('type') == 'player_game_performance_edge']

        for u, v in pgp_edges:
            # Get raw historical stats
            raw_hist_stats = get_historical_pgp_edge_stats(data_graph, u, v, window_size, config)

            # Store games played
            data_graph.edges[u, v][f'hist_{window_size}_games_played'] = raw_hist_stats[
                f'hist_{window_size}_games_played']

            # Calculate and store averages (replacing raw values)
            game_counts = raw_hist_stats[f'hist_{window_size}_games_played']
            for stat in config.stat_attributes['player_pair_stats']:
                stat_key = f'hist_{window_size}_{stat}'
                if stat_key in raw_hist_stats:
                    raw_values = raw_hist_stats[stat_key]
                    avg_values = [0, 0, 0]

                    # Calculate average for each period type, avoiding division by zero
                    for i in range(3):
                        if game_counts[i] > 0:
                            avg_values[i] = raw_values[i] / game_counts[i]

                    # Store only the averages, no raw totals
                    data_graph.edges[u, v][stat_key] = avg_values

    print("Historical statistics calculation complete.")
    return data_graph
