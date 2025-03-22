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
    graph.add_node(team, type='team')


def add_player_node(graph, player, player_dict):
    player_dict_details = player_dict[player]
    player_dict_details['games'] = [0, 0, 0],
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

    graph.add_node(player, type='player', **player_dict_details)


def add_game(graph, game):
    """
    Create game nodes and edges, maintaining indices for quick lookups.
    """
    default_stats = {
        'games': [0, 0, 0],
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

    graph.add_node(away_tgp, type='team_game_performance', game_date=game_date ,home=0, **away_stats)
    graph.add_node(home_tgp, type='team_game_performance', game_date=game_date, home=1, **home_stats)
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
    """
    Add Player Game Performance (PGP) nodes to the graph for each player in the roster.
    Now includes player names in the PGP node attributes for easier debugging and visualization.

    Args:
        graph: NetworkX graph to update
        roster: List of player roster dictionaries

    Returns:
        Dictionary mapping team keys to lists of PGP nodes
    """
    default_stats = {
        'games': [0, 0, 0],
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
        game_date = player['game_date']
        player_id = player['player_id']
        player_team = player['player_team']
        player_position = player['player_position']
        player_pgp = str(game_id) + '_' + str(player_id)
        team_tgp = str(game_id) + '_' + player_team

        # Create copies of default stats for node and edge
        stat_node_copy = copy.deepcopy(default_stats)
        edge_node_copy = copy.deepcopy(default_stats)

        # Get player name information from player node if available
        player_name = ""
        player_first_name = ""
        player_last_name = ""

        if player_id in graph.nodes:
            player_node = graph.nodes[player_id]

            # Try to get first and last name (handling different possible field names)
            if 'first_name' in player_node and 'last_name' in player_node:
                player_first_name = player_node['first_name']
                player_last_name = player_node['last_name']
            elif 'firstName' in player_node and 'lastName' in player_node:
                player_first_name = player_node['firstName']
                player_last_name = player_node['lastName']
            elif 'first' in player_node and 'last' in player_node:
                player_first_name = player_node['first']
                player_last_name = player_node['last']

            # If we found a first and last name, combine them
            if player_first_name and player_last_name:
                player_name = f"{player_first_name} {player_last_name}"

            # If we couldn't find separate first/last names, look for a full name field
            if not player_name:
                if 'name' in player_node:
                    player_name = player_node['name']
                elif 'full_name' in player_node:
                    player_name = player_node['full_name']
                elif 'fullName' in player_node:
                    player_name = player_node['fullName']
                elif 'player_name' in player_node:
                    player_name = player_node['player_name']

            # If we still don't have a name, check any attribute with "name" in it
            if not player_name:
                for key, value in player_node.items():
                    if 'name' in key.lower() and isinstance(value, str) and value:
                        player_name = value
                        break

        # Add PGP node with player name information
        graph.add_node(player_pgp,
                       type='player_game_performance',
                       player_position=player_position,
                       player_team=player_team,
                       game_date=game_date,
                       player_name=player_name,
                       player_first_name=player_first_name,
                       player_last_name=player_last_name,
                       **stat_node_copy)

        # Add edges
        graph.add_edge(player_pgp, team_tgp, type='pgp_tgp_edge')
        graph.add_edge(player_pgp, player_id, type='pgp_player_edge',
                       player_team=player_team,
                       game_date=game_date, **edge_node_copy)

        # Build the team_game_map
        team_key = (game_id, player_team)
        if team_key not in team_game_map:
            team_game_map[team_key] = []
        team_game_map[team_key].append(player_pgp)

    # Add edges between players on the same team
    for team_key, player_pgps in team_game_map.items():
        for i in range(len(player_pgps)):
            for j in range(i + 1, len(player_pgps)):
                stat_copy = copy.deepcopy(default_stats)
                graph.add_edge(player_pgps[i], player_pgps[j],
                               type='pgp_pgp_edge',
                               game_date=game_date,
                               **stat_copy)

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
    if 'hist_games' not in data_graph.nodes[team_tgp]:
        data_graph.nodes[team_tgp]['hist_games'] = [0, 0, 0]

    # Only count this as a game/period played if the player had time on ice
    if player_dat.get('toi', [0, 0, 0])[period_code] > 0:
        data_graph.nodes[team_tgp]['games'][period_code] = 1

    if period_code == 2:
        data_graph.nodes[team_tgp]['games'][period_code] = 1

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

    # Only count this as a period played if the player had time on ice
    if player_dat.get('toi', [0, 0, 0])[period_code] > 0:
        data_graph.nodes[player_pgp]['games'][period_code] = 1

    if period_code == 2:
        data_graph.nodes[player_pgp]['games'][period_code] = 1

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
    # Only count this as a period played if the player had time on ice
    if player_dat.get('toi', [0, 0, 0])[period_code] > 0:
        data_graph.edges[player1_pgp, player2_pgp]['games'][period_code] = 1

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
    Only considers games that occurred BEFORE the current game.

    Args:
        data_graph: NetworkX graph containing all game data
        team_tgp: ID of the team's game performance node
        window_size: Number of previous games to consider
        config: Configuration object with stat attributes

    Returns:
        Dictionary containing historical statistics
    """
    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_games': [0, 0, 0]
    }

    # Initialize stat arrays for all team statistics
    for stat in config.stat_attributes['team_stats']:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Parse team_tgp to get game_id and team
    game_id, team = team_tgp.split('_')

    # Get current game's date
    current_game_date = data_graph.nodes[team_tgp]['game_date']
    if not current_game_date:
        print(f"Warning: No date found for game {game_id}")
        return historical_stats

    # Get all team's games from the graph's indexed structure
    team_games = data_graph.graph.get('team_games', {}).get(team, [])

    # Filter and sort historical games
    historical_games = []
    for game_info in team_games:
        if (game_info['game_id'] != game_id and
                game_info['date'] < current_game_date):
            historical_games.append(game_info)

    # Sort by date and get the most recent games up to window_size
    historical_games.sort(key=lambda x: x['date'])
    recent_games = historical_games[-window_size:] if len(historical_games) > window_size else historical_games

    # Process each historical game
    for game_info in recent_games:
        tgp_node = game_info['tgp_node']
        if tgp_node not in data_graph.nodes:
            continue

        node_data = data_graph.nodes[tgp_node]

        # Aggregate period-specific statistics
        for period_idx in range(3):  # 0=regulation, 1=overtime, 2=shootout
            # Check if the team participated in this period
            period_participation = False

            # Check various indicators of participation
            if ('toi' in node_data and node_data['toi'][period_idx] > 0 or
                    'goal' in node_data and node_data['goal'][period_idx] > 0 or
                    'games' in node_data and node_data['games'][period_idx] > 0):
                period_participation = True

            if period_participation:
                historical_stats[f'hist_{window_size}_games'][period_idx] += 1

                # Aggregate all team statistics for this period
                for stat in config.stat_attributes['team_stats']:
                    if stat in node_data:
                        stat_key = f'hist_{window_size}_{stat}'
                        historical_stats[stat_key][period_idx] += node_data[stat][period_idx]

    return historical_stats


def get_historical_pgp_stats(data_graph, player_pgp, window_size, config):
    """
    Get historical statistics for a player using the team games window
    and game-to-pgp mapping for quick lookups.
    """
    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_games': [0, 0, 0]
    }

    # Initialize stat arrays for all player statistics
    for stat in config.stat_attributes['player_stats']:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Parse player_pgp to get game_id and player_id
    game_id, player_id = player_pgp.split('_')
    player_id = int(player_id)

    # Get current player's team and game date
    current_player_team = data_graph.nodes[player_pgp].get('player_team')
    current_game_date = data_graph.nodes[player_pgp].get('game_date')

    if not current_player_team or not current_game_date:
        return historical_stats

    # Get team's historic games window using existing team_games index
    team_games = data_graph.graph.get('team_games', {}).get(current_player_team, [])

    # Filter for games before the current game
    historical_games = [
        game_info for game_info in team_games
        if (game_info['game_id'] != int(game_id) and
            game_info['date'] < current_game_date)
    ]

    # Sort by date and get window_size most recent games
    historical_games.sort(key=lambda x: x['date'])
    recent_games = historical_games[-window_size:] if len(historical_games) > window_size else historical_games

    # Ensure the game-to-pgp index exists
    if 'game_to_pgp' not in data_graph.graph:
        build_game_to_pgp_index(data_graph)

    # For each game in the window, check if the player participated
    for game_info in recent_games:
        game_id_int = game_info['game_id']

        # Get all PGP nodes for this game from our index
        pgp_entries = [
            entry for entry in data_graph.graph['game_to_pgp'].get(game_id_int, [])
            if entry['player_id'] == player_id and entry['player_team'] == current_player_team
        ]

        for entry in pgp_entries:
            pgp_node = entry['pgp_node']
            node_data = data_graph.nodes[pgp_node]

            # Aggregate period-specific statistics
            for period_idx in range(3):
                # Check if player participated in this period
                period_participation = False

                if ('toi' in node_data and node_data['toi'][period_idx] > 0 or
                        'games' in node_data and node_data['games'][period_idx] > 0):
                    period_participation = True

                if period_participation:
                    historical_stats[f'hist_{window_size}_games'][period_idx] += 1

                    # Aggregate all player statistics for this period
                    for stat in config.stat_attributes['player_stats']:
                        if stat in node_data:
                            stat_key = f'hist_{window_size}_{stat}'
                            historical_stats[stat_key][period_idx] += node_data[stat][period_idx]

    return historical_stats


def calculate_historical_pgp_pgp_edge_stats(data_graph, config):
    """
    Calculate historical statistics specifically for edges between PGP nodes
    (player_game_performance to player_game_performance edges) using a game-focused approach.

    Args:
        data_graph: NetworkX graph containing all game data
        config: Configuration object with stat attributes and window sizes

    Returns:
        Updated graph with historical edge statistics
    """
    print("Calculating historical PGP-PGP edge statistics...")

    # Get sorted list of all games by date
    game_nodes = [
        (game_id, data_graph.nodes[game_id]['game_date'])
        for game_id in data_graph.nodes()
        if data_graph.nodes[game_id].get('type') == 'game'
    ]

    # Sort games chronologically
    sorted_games = sorted(game_nodes, key=lambda x: x[1])

    # Create a dictionary to store historical player pair interactions
    # Key: (player1_id, player2_id, team)
    # Value: List of previous games with player pair statistics
    player_pair_history = {}

    # Track progress
    total_games = len(sorted_games)

    # Process each game chronologically
    for idx, (game_id, game_date) in enumerate(sorted_games):
        if idx % 100 == 0:
            print(f"Processing game {idx} of {total_games} ({idx / total_games * 100:.1f}%)")

        # Get all PGP-PGP edges for this game
        pgp_pgp_edges = get_pgp_pgp_edges_for_game(data_graph, game_id)

        # For each edge, calculate historical statistics
        for edge_info in pgp_pgp_edges:
            player1_id = edge_info['player1_id']
            player2_id = edge_info['player2_id']
            team = edge_info['team']
            pgp1_node = edge_info['pgp1_node']
            pgp2_node = edge_info['pgp2_node']

            # Create a unique key for this player pair
            # Use sorted player IDs to ensure consistent key regardless of order
            pair_key = (min(player1_id, player2_id), max(player1_id, player2_id), team)

            # Get historical statistics for each window size
            for window_size in config.stat_window_sizes:
                # Get previous interactions for this player pair
                previous_interactions = get_previous_interactions(
                    player_pair_history, pair_key, game_date, window_size
                )

                # Calculate historical statistics
                hist_stats = aggregate_historical_edge_stats(window_size,
                    previous_interactions, config.stat_attributes['player_pair_stats']
                )

                # Store the statistics on the edge
                for stat, values in hist_stats.items():
                    data_graph.edges[pgp1_node, pgp2_node][stat] = values

                # Calculate and store averages
                games_played = hist_stats[f'hist_{window_size}_games']
                for stat in config.stat_attributes['player_pair_stats']:
                    stat_key = f'hist_{window_size}_{stat}'
                    if stat_key in hist_stats:
                        raw_values = hist_stats[stat_key]
                        avg_values = [
                            raw_values[i] / games_played[i] if games_played[i] > 0 else 0
                            for i in range(3)
                        ]
                        data_graph.edges[pgp1_node, pgp2_node][f'{stat_key}_avg'] = avg_values

            # Add current interaction to the history for future games
            current_stats = extract_edge_stats(data_graph, pgp1_node, pgp2_node,
                                               config.stat_attributes['player_pair_stats'])

            if pair_key not in player_pair_history:
                player_pair_history[pair_key] = []

            player_pair_history[pair_key].append({
                'game_date': game_date,
                'stats': current_stats
            })

    print("Historical PGP-PGP edge statistics calculation complete.")
    return data_graph


def get_pgp_pgp_edges_for_game(data_graph, game_id):
    """
    Get all PGP-to-PGP edges for a specific game.
    These are edges between two player_game_performance nodes.

    Args:
        data_graph: NetworkX graph containing all game data
        game_id: ID of the game

    Returns:
        List of dictionaries containing edge information
    """
    edges = []

    # First, get all PGP nodes for this game
    pgp_nodes = []
    for node, data in data_graph.nodes(data=True):
        if (isinstance(node, str) and
                str(game_id) in node and
                data.get('type') == 'player_game_performance'):
            pgp_nodes.append((node, data))

    # Now find edges between PGP nodes
    for i, (pgp1, data1) in enumerate(pgp_nodes):
        player1_id = int(pgp1.split('_')[1])
        team1 = data1.get('player_team')

        for j in range(i + 1, len(pgp_nodes)):
            pgp2, data2 = pgp_nodes[j]
            player2_id = int(pgp2.split('_')[1])
            team2 = data2.get('player_team')

            # Only consider edges between players on the same team
            if team1 == team2 and data_graph.has_edge(pgp1, pgp2):
                edge_data = data_graph.get_edge_data(pgp1, pgp2)

                # Ensure it's a PGP-PGP edge (not a different type of edge)
                if edge_data.get('type', '') == 'pgp_pgp_edge':
                    edges.append({
                        'pgp1_node': pgp1,
                        'pgp2_node': pgp2,
                        'player1_id': player1_id,
                        'player2_id': player2_id,
                        'team': team1,
                        'game_date': data1.get('game_date')
                    })

    return edges


def get_previous_interactions(player_pair_history, pair_key, current_date, window_size):
    """
    Get previous interactions for a player pair within the window size.

    Args:
        player_pair_history: Dictionary of historical player pair interactions
        pair_key: Tuple (player1_id, player2_id, team)
        current_date: Date of the current game
        window_size: Number of previous games to consider

    Returns:
        List of previous interactions
    """
    if pair_key not in player_pair_history:
        return []

    # Get all previous interactions
    previous = [
        interaction for interaction in player_pair_history[pair_key]
        if interaction['game_date'] < current_date
    ]

    # Sort by date (most recent first)
    previous.sort(key=lambda x: x['game_date'], reverse=True)

    # Return the most recent interactions up to window_size
    return previous[:window_size]


def aggregate_historical_edge_stats(window_size, previous_interactions, stat_attributes):
    """
    Aggregate historical statistics for a player pair.

    Args:
        previous_interactions: List of previous interactions
        stat_attributes: List of statistics to aggregate

    Returns:
        Dictionary containing aggregated historical statistics
    """

    # Initialize historical stats
    historical_stats = {
        f'hist_{window_size}_games': [0, 0, 0]
    }

    # Initialize stat arrays for all player pair statistics
    for stat in stat_attributes:
        historical_stats[f'hist_{window_size}_{stat}'] = [0, 0, 0]

    # Aggregate statistics from previous interactions
    for interaction in previous_interactions:
        stats = interaction['stats']

        for period_idx in range(3):
            # Check if there was participation in this period
            if stats['games'][period_idx] > 0:
                historical_stats[f'hist_{window_size}_games'][period_idx] += 1

                # Aggregate all statistics for this period
                for stat in stat_attributes:
                    if stat in stats:
                        historical_stats[f'hist_{window_size}_{stat}'][period_idx] += stats[stat][period_idx]

    return historical_stats


def extract_edge_stats(data_graph, pgp1_node, pgp2_node, stat_attributes):
    """
    Extract statistics from a PGP-to-PGP edge.

    Args:
        data_graph: NetworkX graph containing all game data
        pgp1_node: ID of the first player's game performance node
        pgp2_node: ID of the second player's game performance node
        stat_attributes: List of statistics to extract

    Returns:
        Dictionary containing edge statistics
    """
    edge_data = data_graph.get_edge_data(pgp1_node, pgp2_node)

    # Extract relevant statistics
    stats = {
        'games': edge_data.get('games', [0, 0, 0])
    }

    for stat in stat_attributes:
        if stat in edge_data:
            stats[stat] = edge_data[stat]
        else:
            stats[stat] = [0, 0, 0]

    return stats


def calculate_historical_stats(config, data_graph):
    """
    Calculate historical statistics for all nodes in the graph.

    Args:
        config: Configuration object with stat attributes and window sizes
        data_graph: NetworkX graph containing all game data

    Returns:
        Updated graph with historical statistics
    """
    print("Calculating historical statistics...")

    # Ensure indices are built for efficient lookups
    if 'game_to_pgp' not in data_graph.graph:
        print("Building game to PGP index...")
        build_game_to_pgp_index(data_graph)

    if 'game_to_pgp_edges' not in data_graph.graph:
        print("Building game to PGP edges index...")
        build_game_to_pgp_edges_index(data_graph)

    # Process each window size
    for window_size in config.stat_window_sizes:
        print(f"Processing window size {window_size}...")

        # Process team game performance nodes
        tgp_nodes = [n for n, d in data_graph.nodes(data=True)
                     if d.get('type') == 'team_game_performance']

        for m, tgp in enumerate(tgp_nodes):
            if m % 100 == 0:
                print(
                    f'Window size: {window_size} Processing TGP node {m} of {len(tgp_nodes)} ({(m / len(tgp_nodes)) * 100:.1f}%)')

            # Calculate historical stats
            hist_stats = get_historical_tgp_stats(data_graph, tgp, window_size, config)

            # Store raw historical stats
            for stat, values in hist_stats.items():
                data_graph.nodes[tgp][stat] = values

            # Calculate and store averages
            games_played = hist_stats[f'hist_{window_size}_games']
            for stat in config.stat_attributes['team_stats']:
                stat_key = f'hist_{window_size}_{stat}'
                if stat_key in hist_stats:
                    raw_values = hist_stats[stat_key]
                    avg_values = [
                        raw_values[i] / games_played[i] if games_played[i] > 0 else 0
                        for i in range(3)
                    ]
                    data_graph.nodes[tgp][f'{stat_key}_avg'] = avg_values

        # Process player game performance nodes
        pgp_nodes = [n for n, d in data_graph.nodes(data=True)
                     if d.get('type') == 'player_game_performance']

        for n, pgp in enumerate(pgp_nodes):
            if n % 1000 == 0:
                print(
                    f'Window size: {window_size} Processing PGP node {n} of {len(pgp_nodes)} ({(n / len(pgp_nodes)) * 100:.1f}%)')

            # Calculate historical stats
            hist_stats = get_historical_pgp_stats(data_graph, pgp, window_size, config)

            # Store raw historical stats
            for stat, values in hist_stats.items():
                data_graph.nodes[pgp][stat] = values

            # Calculate and store averages
            games_played = hist_stats[f'hist_{window_size}_games']
            for stat in config.stat_attributes['player_stats']:
                stat_key = f'hist_{window_size}_{stat}'
                if stat_key in hist_stats:
                    raw_values = hist_stats[stat_key]
                    avg_values = [
                        raw_values[i] / games_played[i] if games_played[i] > 0 else 0
                        for i in range(3)
                    ]
                    data_graph.nodes[pgp][f'{stat_key}_avg'] = avg_values

    calculate_historical_pgp_pgp_edge_stats(data_graph, config)
    print("Historical statistics calculation complete.")
    return data_graph


def build_game_to_pgp_index(graph):
    """
    Build an index mapping games to their player game performance nodes.
    This allows quick retrieval of all PGP nodes for a given game.
    """
    if 'game_to_pgp' not in graph.graph:
        graph.graph['game_to_pgp'] = defaultdict(list)

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') == 'player_game_performance':
            parts = node_id.split('_')
            if len(parts) == 2:
                game_id, player_id = parts
                if game_id.isdigit():
                    graph.graph['game_to_pgp'][int(game_id)].append({
                        'pgp_node': node_id,
                        'player_id': int(player_id),
                        'player_team': node_data.get('player_team')
                    })

    return graph


def build_game_to_pgp_edges_index(graph):
    """
    Build an index mapping games to their PGP-PGP edges.
    This allows quick retrieval of all player pair interactions for a given game.
    """
    if 'game_to_pgp_edges' not in graph.graph:
        graph.graph['game_to_pgp_edges'] = defaultdict(list)

    for node1, node2, edge_data in graph.edges(data=True):
        # Check if it's a PGP-PGP edge
        if (isinstance(node1, str) and isinstance(node2, str) and
                edge_data.get('type') == 'pgp_pgp_edge'):

            parts1 = node1.split('_')
            parts2 = node2.split('_')

            if len(parts1) == 2 and len(parts2) == 2 and parts1[0] == parts2[0]:
                game_id = parts1[0]
                if game_id.isdigit():
                    graph.graph['game_to_pgp_edges'][int(game_id)].append({
                        'pgp1_node': node1,
                        'pgp2_node': node2,
                        'player1_id': int(parts1[1]),
                        'player2_id': int(parts2[1]),
                        'team': graph.nodes[node1].get('player_team')
                    })

    return graph
