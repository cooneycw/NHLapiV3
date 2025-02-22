from collections import defaultdict
from datetime import datetime, timedelta
import copy
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
    player_dict_details['shot_on_goal'] = [0, 0, 0]
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
    game_id = game['id']
    game_date = game['game_date']
    graph.add_node(game_id, type = 'game', game_date=game_date, home_team=game['homeTeam'], away_team=game['awayTeam'])
    graph.add_edge(game['awayTeam'], game_id, home = 0)
    graph.add_edge(game['homeTeam'], game_id, home = 1)
    away_tgp = str(game_id) + '_' + game['awayTeam']
    home_tgp = str(game_id) + '_' +game['homeTeam']

    away_stats = copy.deepcopy(default_stats)
    home_stats = copy.deepcopy(default_stats)

    graph.add_node(away_tgp, type = 'team_game_performance', home=0, **away_stats)
    graph.add_node(home_tgp, type = 'team_game_performance', home=1, **home_stats)
    graph.add_edge(away_tgp, game_id)
    graph.add_edge(home_tgp, game_id)


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


def update_tgp_stats(graph, team_tgp, period_code, stat_dict):
    """Update stats for a Team Game Performance node."""
    tgp_node = graph.nodes[team_tgp]
    # Add logging before update
    # if stat_dict['goal'][period_code] == 1:
    #     print(f"\nUpdating node {team_tgp}")
    #     print(f"Current goals: {tgp_node['goal']}")
    #     print(f"Adding goals: {stat_dict['goal']}")
    #     print(f"Period: {period_code}")

    tgp_node['faceoff_taken'][period_code] += stat_dict['faceoff_taken'][period_code]
    tgp_node['faceoff_won'][period_code] += stat_dict['faceoff_won'][period_code]
    tgp_node['shot_attempt'][period_code] += stat_dict['shot_attempt'][period_code]
    tgp_node['shot_missed'][period_code] += stat_dict['shot_missed'][period_code]
    tgp_node['shot_missed_shootout'][period_code] += stat_dict['shot_missed_shootout'][period_code]
    tgp_node['shot_on_goal'][period_code] += stat_dict['shot_on_goal'][period_code]
    tgp_node['shot_blocked'][period_code] += stat_dict['shot_blocked'][period_code]
    tgp_node['shot_saved'][period_code] += stat_dict['shot_saved'][period_code]
    tgp_node['goal'][period_code] += stat_dict['goal'][period_code]
    tgp_node['goal_against'][period_code] += stat_dict['goal_against'][period_code]
    tgp_node['giveaways'][period_code] += stat_dict['giveaways'][period_code]
    tgp_node['takeaways'][period_code] += stat_dict['takeaways'][period_code]
    tgp_node['hit_another_player'][period_code] += stat_dict['hit_another_player'][period_code]
    tgp_node['hit_by_player'][period_code] += stat_dict['hit_by_player'][period_code]
    tgp_node['penalties'][period_code] += stat_dict['penalties'][period_code]
    tgp_node['penalties_served'][period_code] += stat_dict['penalties_served'][period_code]
    tgp_node['penalties_drawn'][period_code] += stat_dict['penalties_drawn'][period_code]
    tgp_node['penalty_shot'][period_code] += stat_dict['penalty_shot'][period_code]
    tgp_node['penalty_shot_goal'][period_code] += stat_dict['penalty_shot_goal'][period_code]
    tgp_node['penalty_shot_saved'][period_code] += stat_dict['penalty_shot_saved'][period_code]
    tgp_node['penalties_duration'][period_code] += stat_dict['penalties_duration'][period_code]

    # if stat_dict['goal'][period_code] == 1:
    #     print(f"Updated goals: {tgp_node['goal']}")


def update_pgp_stats(graph, player_pgp, period_code, stat_dict):
    """Update stats for a Player Game Performance node."""
    pgp_node = graph.nodes[player_pgp]

    # if stat_dict['goal'][period_code] == 1:
    #     print(f"\nUpdating node {player_pgp}")
    #     print(f"Current goals: {pgp_node['goal']}")
    #     print(f"Adding goals: {stat_dict['goal']}")
    #     print(f"Period: {period_code}")

    # print(f' updating {player_pgp} {pgp_node["faceoff_taken"]}')
    pgp_node['toi'][period_code] += stat_dict['toi'][period_code]
    pgp_node['faceoff_taken'][period_code] += stat_dict['faceoff_taken'][period_code]
    pgp_node['faceoff_won'][period_code] += stat_dict['faceoff_won'][period_code]
    pgp_node['shot_attempt'][period_code] += stat_dict['shot_attempt'][period_code]
    pgp_node['shot_missed'][period_code] += stat_dict['shot_missed'][period_code]
    pgp_node['shot_missed_shootout'][period_code] += stat_dict['shot_missed_shootout'][period_code]
    pgp_node['shot_on_goal'][period_code] += stat_dict['shot_on_goal'][period_code]
    pgp_node['shot_blocked'][period_code] += stat_dict['shot_blocked'][period_code]
    pgp_node['shot_saved'][period_code] += stat_dict['shot_saved'][period_code]
    pgp_node['goal'][period_code] += stat_dict['goal'][period_code]
    pgp_node['assist'][period_code] += stat_dict['assist'][period_code]
    pgp_node['point'][period_code] += (stat_dict['goal'][period_code] + stat_dict['assist'][period_code])
    pgp_node['goal_against'][period_code] += stat_dict['goal_against'][period_code]
    pgp_node['giveaways'][period_code] += stat_dict['giveaways'][period_code]
    pgp_node['takeaways'][period_code] += stat_dict['takeaways'][period_code]
    pgp_node['hit_another_player'][period_code] += stat_dict['hit_another_player'][period_code]
    pgp_node['hit_by_player'][period_code] += stat_dict['hit_by_player'][period_code]
    pgp_node['penalties'][period_code] += stat_dict['penalties'][period_code]
    pgp_node['penalties_served'][period_code] += stat_dict['penalties_served'][period_code]
    pgp_node['penalties_drawn'][period_code] += stat_dict['penalties_drawn'][period_code]
    pgp_node['penalty_shot'][period_code] += stat_dict['penalty_shot'][period_code]
    pgp_node['penalty_shot_goal'][period_code] += stat_dict['penalty_shot_goal'][period_code]
    pgp_node['penalty_shot_saved'][period_code] += stat_dict['penalty_shot_saved'][period_code]
    pgp_node['penalties_duration'][period_code] += stat_dict['penalties_duration'][period_code]

    # if stat_dict['goal'][period_code] == 1:
    #     print(f"Updated goals: {pgp_node['goal']}")


def update_pgp_edge_stats(graph, player_pgp, other_pgp, period_id, stat_dict):
    # pgp_edge_stats = graph[player_pgp][other_pgp]
    pgp_edge_stats = graph.get_edge_data(player_pgp, other_pgp, default={})
    # print(f'{player_pgp}:{other_pgp}:{pgp_edge_stats}')
    if pgp_edge_stats == {}:
        cwc = 0
    # else:
    #     print(f'amending: {player_pgp} -> {other_pgp}')
    pgp_edge_stats['toi'][period_id] += stat_dict['toi'][period_id]
    pgp_edge_stats['faceoff_taken'][period_id] += stat_dict['faceoff_taken'][period_id]
    pgp_edge_stats['faceoff_won'][period_id] += stat_dict['faceoff_won'][period_id]
    pgp_edge_stats['shot_attempt'][period_id] += stat_dict['shot_attempt'][period_id]
    pgp_edge_stats['shot_on_goal'][period_id] += stat_dict['shot_on_goal'][period_id]
    pgp_edge_stats['shot_blocked'][period_id] += stat_dict['shot_blocked'][period_id]
    pgp_edge_stats['shot_saved'][period_id] += stat_dict['shot_saved'][period_id]
    pgp_edge_stats['goal'][period_id] += stat_dict['goal'][period_id]
    pgp_edge_stats['goal_against'][period_id] += stat_dict['goal_against'][period_id]
    pgp_edge_stats['hit_another_player'][period_id] += stat_dict['hit_another_player'][period_id]
    pgp_edge_stats['hit_by_player'][period_id] += stat_dict['hit_by_player'][period_id]
    pgp_edge_stats['penalties_duration'][period_id] += stat_dict['penalties_duration'][period_id]


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


def get_historical_tgp_stats(graph, team_id, current_game_id, stat_attributes, n_games):
    """
    Calculate historical team game performance stats.

    Args:
        graph: NetworkX graph containing all game data
        team_id: ID of the team to analyze
        current_game_id: Current game ID to exclude
        stat_attributes: Dictionary of attributes to track
        n_games: Number of previous games to include in the window

    Returns:
        tuple: (stats dictionary, actual game count) where stats contains the averaged stats
        and actual_games is the number of games actually found in the window
    """
    current_game_date = graph.nodes[current_game_id]['game_date']
    if isinstance(current_game_date, str):
        current_game_date = datetime.strptime(current_game_date, '%Y-%m-%d')

    # Find all previous TGP nodes for this team
    team_games = []
    for node in graph.nodes():
        if isinstance(node, str) and node.endswith(f"_{team_id}"):
            if graph.nodes[node].get('type') == 'team_game_performance':
                game_id_from_node = int(node.split('_')[0])
                if game_id_from_node != current_game_id:
                    game_date = graph.nodes[game_id_from_node]['game_date']
                    if isinstance(game_date, str):
                        game_date = datetime.strptime(game_date, '%Y-%m-%d')
                    if game_date < current_game_date:
                        team_games.append((node, game_date))

    team_games.sort(key=lambda x: x[1], reverse=True)
    recent_games = team_games[:n_games]

    if not recent_games:
        return None, 0

    # Initialize stats
    team_stats = {attr: [0, 0, 0] for attr in stat_attributes['team_stats']}

    # Aggregate stats
    for tgp_node, _ in recent_games:
        node_data = graph.nodes[tgp_node]
        for stat_name in stat_attributes['team_stats']:
            if stat_name in node_data:
                team_stats[stat_name] = [
                    team_stats[stat_name][i] + node_data[stat_name][i]
                    for i in range(3)
                ]

    # Calculate averages
    actual_games = len(recent_games)  # Track actual number of games found
    team_stats = {k: [v[i] / actual_games for i in range(3)] for k, v in team_stats.items()}

    return team_stats, actual_games


def get_historical_pgp_stats(graph, team_id, current_game_id, stat_attributes, n_games):
    """
    Calculate historical player game performance stats with game counts.

    Args:
        graph: NetworkX graph containing all game data
        team_id: ID of the team to analyze
        current_game_id: Current game ID to exclude
        stat_attributes: Dictionary of attributes to track
        n_games: Number of previous games to include

    Returns:
        tuple: (stats dictionary, games_played dictionary) where stats contains the averaged stats
        and games_played contains the count of games used for each player's calculation
    """
    current_game_date = graph.nodes[current_game_id]['game_date']
    if isinstance(current_game_date, str):
        current_game_date = datetime.strptime(current_game_date, '%Y-%m-%d')

    # Find relevant games
    team_games = []
    for node in graph.nodes():
        if isinstance(node, str) and node.endswith(f"_{team_id}"):
            if graph.nodes[node].get('type') == 'team_game_performance':
                game_id_from_node = int(node.split('_')[0])
                if game_id_from_node != current_game_id:
                    game_date = graph.nodes[game_id_from_node]['game_date']
                    if isinstance(game_date, str):
                        game_date = datetime.strptime(game_date, '%Y-%m-%d')
                    if game_date < current_game_date:
                        team_games.append((node, game_date))

    team_games.sort(key=lambda x: x[1], reverse=True)
    recent_games = team_games[:n_games]

    if not recent_games:
        return None, None

    player_stats = defaultdict(lambda: {attr: [0, 0, 0] for attr in stat_attributes['player_stats']})
    player_games = defaultdict(int)
    player_date_ranges = defaultdict(lambda: {'first_game': None, 'last_game': None})

    # Aggregate player stats from recent games
    for tgp_node, game_date in recent_games:
        game_id_from_node = tgp_node.split('_')[0]
        player_pgps = [
            node for node in graph.nodes()
            if isinstance(node, str)
               and node.startswith(f"{game_id_from_node}_")
               and graph.nodes[node].get('type') == 'player_game_performance'
        ]

        for pgp in player_pgps:
            node_data = graph.nodes[pgp]
            player_id = None
            for neighbor in graph.neighbors(pgp):
                if graph.nodes[neighbor].get('type') == 'player':
                    player_id = neighbor
                    break

            if player_id:
                player_games[player_id] += 1

                # Update date range for player
                if player_date_ranges[player_id]['first_game'] is None or game_date < player_date_ranges[player_id][
                    'first_game']:
                    player_date_ranges[player_id]['first_game'] = game_date
                if player_date_ranges[player_id]['last_game'] is None or game_date > player_date_ranges[player_id][
                    'last_game']:
                    player_date_ranges[player_id]['last_game'] = game_date

                for stat_name in stat_attributes['player_stats']:
                    if stat_name in node_data:
                        player_stats[player_id][stat_name] = [
                            player_stats[player_id][stat_name][i] + node_data[stat_name][i]
                            for i in range(3)
                        ]

    # Calculate averages and include metadata
    stats = {}
    for player, stats_dict in player_stats.items():
        player_avg_stats = {
            k: [v[i] / player_games[player] for i in range(3)]
            for k, v in stats_dict.items()
        }
        # Add metadata
        player_avg_stats['games_played'] = player_games[player]
        player_avg_stats['date_range'] = {
            'first_game': player_date_ranges[player]['first_game'],
            'last_game': player_date_ranges[player]['last_game']
        }
        stats[player] = player_avg_stats

    return stats


def get_historical_pgp_edge_stats(graph, team_id, current_game_id, stat_attributes, n_games):
    """
    Calculate historical player-pair stats with game counts and date ranges.

    Args:
        graph: NetworkX graph containing all game data
        team_id: ID of the team to analyze
        current_game_id: Current game ID to exclude
        stat_attributes: Dictionary of attributes to track
        n_games: Number of previous games to include

    Returns:
        dict: Dictionary containing pair stats, including games played and date ranges
    """
    current_game_date = graph.nodes[current_game_id]['game_date']
    if isinstance(current_game_date, str):
        current_game_date = datetime.strptime(current_game_date, '%Y-%m-%d')

    # Find relevant games
    team_games = []
    for node in graph.nodes():
        if isinstance(node, str) and node.endswith(f"_{team_id}"):
            if graph.nodes[node].get('type') == 'team_game_performance':
                game_id_from_node = int(node.split('_')[0])
                if game_id_from_node != current_game_id:
                    game_date = graph.nodes[game_id_from_node]['game_date']
                    if isinstance(game_date, str):
                        game_date = datetime.strptime(game_date, '%Y-%m-%d')
                    if game_date < current_game_date:
                        team_games.append((node, game_date))

    team_games.sort(key=lambda x: x[1], reverse=True)
    recent_games = team_games[:n_games]

    if not recent_games:
        return None

    pair_stats = defaultdict(lambda: {attr: [0, 0, 0] for attr in stat_attributes['player_pair_stats']})
    pair_games = defaultdict(int)
    pair_date_ranges = defaultdict(lambda: {'first_game': None, 'last_game': None})

    # Aggregate stats from recent games
    for tgp_node, game_date in recent_games:
        game_id_from_node = tgp_node.split('_')[0]
        player_pgps = [
            node for node in graph.nodes()
            if isinstance(node, str)
               and node.startswith(f"{game_id_from_node}_")
               and graph.nodes[node].get('type') == 'player_game_performance'
        ]

        for pgp1 in player_pgps:
            for pgp2 in player_pgps:
                if pgp1 < pgp2 and graph.has_edge(pgp1, pgp2):
                    player1_id = None
                    player2_id = None

                    for neighbor in graph.neighbors(pgp1):
                        if graph.nodes[neighbor].get('type') == 'player':
                            player1_id = neighbor
                            break
                    for neighbor in graph.neighbors(pgp2):
                        if graph.nodes[neighbor].get('type') == 'player':
                            player2_id = neighbor
                            break

                    if player1_id and player2_id:
                        pair_key = tuple(sorted([player1_id, player2_id]))
                        pair_games[pair_key] += 1

                        # Update date range for pair
                        if pair_date_ranges[pair_key]['first_game'] is None or game_date < pair_date_ranges[pair_key][
                            'first_game']:
                            pair_date_ranges[pair_key]['first_game'] = game_date
                        if pair_date_ranges[pair_key]['last_game'] is None or game_date > pair_date_ranges[pair_key][
                            'last_game']:
                            pair_date_ranges[pair_key]['last_game'] = game_date

                        edge_data = graph[pgp1][pgp2]
                        for stat_name in stat_attributes['player_pair_stats']:
                            if stat_name in edge_data:
                                pair_stats[pair_key][stat_name] = [
                                    pair_stats[pair_key][stat_name][i] + edge_data[stat_name][i]
                                    for i in range(3)
                                ]

    # Calculate averages and include metadata
    stats = {}
    for pair, stats_dict in pair_stats.items():
        pair_avg_stats = {
            k: [v[i] / pair_games[pair] for i in range(3)]
            for k, v in stats_dict.items()
        }
        # Add metadata
        pair_avg_stats['games_played'] = pair_games[pair]
        pair_avg_stats['date_range'] = {
            'first_game': pair_date_ranges[pair]['first_game'],
            'last_game': pair_date_ranges[pair]['last_game']
        }
        stats[pair] = pair_avg_stats

    return stats


def calculate_historical_stats(config, data_graph):
    """
    Calculate all historical statistics after the graph is fully built.
    This should be called after all games are processed.
    """
    # Get all game nodes sorted by date
    game_nodes = []
    for node in data_graph.nodes():
        if data_graph.nodes[node].get('type') == 'game':
            game_date = data_graph.nodes[node]['game_date']
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d')
            game_nodes.append((node, game_date))

    game_nodes.sort(key=lambda x: x[1])  # Sort chronologically

    # Process each game
    for idx, (game_id, _) in enumerate(game_nodes, 1):
        if idx % 20 == 0:
            print(f"Processing game {idx} of {len(game_nodes)} ({(idx / len(game_nodes)) * 100:.1f}%)")
        game_node = data_graph.nodes[game_id]
        home_team = game_node['home_team']
        away_team = game_node['away_team']

        # Calculate historical stats for both teams' TGP nodes
        for team_id in [home_team, away_team]:
            team_tgp = f"{game_id}_{team_id}"
            for window in config.stat_window_sizes:
                hist_stats, actual_games = get_historical_tgp_stats(
                    data_graph, team_id, game_id, config.stat_attributes, window
                )
                if hist_stats:
                    # Store the actual game count for this window
                    data_graph.nodes[team_tgp][f'hist_{window}_game_count'] = actual_games

                    # Store the historical stats
                    for stat_name, values in hist_stats.items():
                        hist_name = f'hist_{window}_{stat_name}'
                        data_graph.nodes[team_tgp][hist_name] = values

        # Get all PGP nodes for this game
        pgp_nodes = [
            node for node in data_graph.nodes()
            if isinstance(node, str)
               and node.startswith(f"{game_id}_")
               and data_graph.nodes[node].get('type') == 'player_game_performance'
        ]

        # Calculate historical stats for PGP nodes
        for pgp in pgp_nodes:
            player_id = None
            player_team = None
            for neighbor in data_graph.neighbors(pgp):
                if data_graph.nodes[neighbor].get('type') == 'player':
                    player_id = neighbor
                elif data_graph.nodes[neighbor].get('type') == 'team_game_performance':
                    player_team = neighbor.split('_')[1]

            if player_id and player_team:
                for window in config.stat_window_sizes:
                    hist_stats = get_historical_pgp_stats(data_graph, player_team, game_id, config.stat_attributes,
                                                          window)
                    if hist_stats and player_id in hist_stats:
                        for stat_name, values in hist_stats[player_id].items():
                            if stat_name not in ('games_played', 'date_range'):
                                hist_name = f'hist_{window}_{stat_name}'
                                data_graph.nodes[pgp][hist_name] = values
                        # Add metadata
                        data_graph.nodes[pgp][f'hist_{window}_games_played'] = hist_stats[player_id]['games_played']
                        data_graph.nodes[pgp][f'hist_{window}_date_range'] = hist_stats[player_id]['date_range']

        # Calculate historical stats for edges between PGP nodes
        for i in range(len(pgp_nodes)):
            for j in range(i + 1, len(pgp_nodes)):
                if data_graph.has_edge(pgp_nodes[i], pgp_nodes[j]):
                    player1_id = None
                    player2_id = None
                    team_id = None

                    # Get player IDs and team
                    for neighbor in data_graph.neighbors(pgp_nodes[i]):
                        if data_graph.nodes[neighbor].get('type') == 'player':
                            player1_id = neighbor
                        elif data_graph.nodes[neighbor].get('type') == 'team_game_performance':
                            team_id = neighbor.split('_')[1]

                    for neighbor in data_graph.neighbors(pgp_nodes[j]):
                        if data_graph.nodes[neighbor].get('type') == 'player':
                            player2_id = neighbor

                    if player1_id and player2_id and team_id:
                        for window in config.stat_window_sizes:
                            hist_stats = get_historical_pgp_edge_stats(
                                data_graph, team_id, game_id, config.stat_attributes, window
                            )
                            if hist_stats:
                                pair_key = tuple(sorted([player1_id, player2_id]))
                                if pair_key in hist_stats:
                                    edge_data = data_graph.get_edge_data(pgp_nodes[i], pgp_nodes[j])
                                    for stat_name, values in hist_stats[pair_key].items():
                                        if stat_name not in ('games_played', 'date_range'):
                                            hist_name = f'hist_{window}_{stat_name}'
                                            edge_data[hist_name] = values
                                    # Add metadata
                                    edge_data[f'hist_{window}_games_played'] = hist_stats[pair_key]['games_played']
                                    edge_data[f'hist_{window}_date_range'] = hist_stats[pair_key]['date_range']

