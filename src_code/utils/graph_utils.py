from datetime import datetime
from typing import List, Dict, Any
import copy
import networkx as nx
import numpy as np


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
    player_dict_details['faceofff_wons'] = [0, 0, 0]
    player_dict_details['shot_on_goal'] = [0, 0, 0]
    player_dict_details['shot_saved'] = [0, 0, 0]
    player_dict_details['goal'] = [0, 0, 0]
    player_dict_details['hit_another_player'] = [0, 0, 0]
    player_dict_details['hit_by_player'] = [0, 0, 0]
    player_dict_details['penalties_duration'] = [0, 0, 0]

    graph.add_node(player, type = 'player', **player_dict_details)


def add_game(graph, game):
    default_stats = {
        'win': [0, 0, 0],
        'loss': [0, 0, 0],
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


def add_player_game_performance(graph, roster):
    default_stats = {
        'toi': [0, 0, 0],
        'faceoff_taken': [0, 0, 0],
        'faceoff_won': [0, 0, 0],
        'shot_on_goal': [0, 0, 0],
        'shot_saved': [0, 0, 0],
        'goal': [0, 0, 0],
        'assist': [0, 0, 0],
        'point': [0, 0, 0],
        'hit_another_player': [0, 0, 0],
        'hit_by_player': [0, 0, 0],
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
    tgp_node['shot_on_goal'][period_code] += stat_dict['shot_on_goal'][period_code]
    tgp_node['shot_saved'][period_code] += stat_dict['shot_saved'][period_code]
    tgp_node['goal'][period_code] += stat_dict['goal'][period_code]
    tgp_node['hit_another_player'][period_code] += stat_dict['hit_another_player'][period_code]
    tgp_node['hit_by_player'][period_code] += stat_dict['hit_by_player'][period_code]
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
    pgp_node['shot_on_goal'][period_code] += stat_dict['shot_on_goal'][period_code]
    pgp_node['shot_saved'][period_code] += stat_dict['shot_saved'][period_code]
    pgp_node['goal'][period_code] += stat_dict['goal'][period_code]
    pgp_node['assist'][period_code] += stat_dict['assist'][period_code]
    pgp_node['point'][period_code] += (stat_dict['goal'][period_code] + stat_dict['assist'][period_code])
    pgp_node['hit_another_player'][period_code] += stat_dict['hit_another_player'][period_code]
    pgp_node['hit_by_player'][period_code] += stat_dict['hit_by_player'][period_code]
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
    pgp_edge_stats['shot_on_goal'][period_id] += stat_dict['shot_on_goal'][period_id]
    pgp_edge_stats['shot_saved'][period_id] += stat_dict['shot_saved'][period_id]
    pgp_edge_stats['goal'][period_id] += stat_dict['goal'][period_id]
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


def get_player_game_nodes(data_graph, player_id: int, game_id: str) -> List[Dict]:
    """
    Get previous game nodes for a player from the graph, up to current game.

    Args:
        data_graph: NetworkX graph containing all game data
        player_id: Player's ID
        game_id: Current game ID to establish cutoff

    Returns:
        List of game nodes, each containing game data
    """
    # Find all PGP (Player Game Performance) nodes for this player
    pgp_nodes = []

    # Convert game_id to string for consistent comparison
    game_id_str = str(game_id)

    for node in data_graph.nodes():
        # Check if this is a PGP node for our player
        if isinstance(node, str) and '_' in node:
            node_parts = node.split('_')
            if len(node_parts) == 2:
                node_game_id, node_player_id = node_parts

                # Check if this is for our player and from a previous game
                if (str(node_player_id) == str(player_id) and
                        node_game_id < game_id_str and
                        data_graph.nodes[node].get('type') == 'player_game_performance'):

                    # Get game date from the game node
                    game_node = node_game_id  # The game ID itself is the node
                    if game_node in data_graph.nodes:
                        game_data = data_graph.nodes[game_node]
                        game_date = game_data.get('game_date')

                        node_data = data_graph.nodes[node]
                        pgp_nodes.append({
                            'node_id': node,
                            'game_id': node_game_id,
                            'game_date': game_date,
                            'toi': node_data.get('toi', [0, 0, 0]),
                            'goals': node_data.get('goal', [0, 0, 0])
                        })

    # Sort by game date if available, otherwise by game_id
    pgp_nodes.sort(key=lambda x: (x.get('game_date', ''), x['game_id']))
    return pgp_nodes[-5:]  # Return last 5 games


def calculate_temporal_features(data_graph, player_id: int, current_game_id: str,
                                current_game_date: str) -> Dict[str, Any]:
    """
    Calculate temporal features for a player leading up to current game.
    """
    # Get last 5 games
    last_5_games = get_player_game_nodes(data_graph, player_id, current_game_id)

    if not last_5_games:
        # Return default values if no previous games
        return {
            'days_since_last_game': None,
            'avg_days_between_games': None,
            'toi_fatigue': 0.0,
            'games_fatigue': 0.0,
            'recent_form': 0.0,
            'last_5_goals': [0, 0, 0],  # [reg, ot, shootout]
            'last_5_toi': [0, 0, 0],  # [reg, ot, shootout]
        }

    # Calculate days since last game
    last_game_date = last_5_games[-1].get('game_date')
    days_since_last = (calculate_days_between_games(last_game_date, current_game_date)
                       if last_game_date else None)

    # Calculate average days between games
    between_game_days = []
    for i in range(len(last_5_games) - 1):
        game1_date = last_5_games[i].get('game_date')
        game2_date = last_5_games[i + 1].get('game_date')
        if game1_date and game2_date:
            days = calculate_days_between_games(game1_date, game2_date)
            between_game_days.append(days)

    avg_days_between = np.mean(between_game_days) if between_game_days else None

    # Calculate weighted features
    weights = np.exp([-i / 2.5 for i in range(len(last_5_games))])
    weights = weights / np.sum(weights)

    # Sum all periods for each game
    toi_values = [sum(game.get('toi', [0, 0, 0])) for game in last_5_games]
    goal_values = [sum(game.get('goals', [0, 0, 0])) for game in last_5_games]

    toi_fatigue = np.sum(weights * np.array(toi_values))
    recent_form = np.sum(weights * np.array(goal_values))

    # Games fatigue calculation
    days_to_games = [calculate_days_between_games(game.get('game_date', current_game_date),
                                                  current_game_date)
                     for game in last_5_games]
    games_fatigue = sum(1 / max(1, days) for days in days_to_games)

    # Aggregate last 5 games statistics
    last_5_goals = [0, 0, 0]
    last_5_toi = [0, 0, 0]

    for game in last_5_games:
        goals = game.get('goals', [0, 0, 0])
        toi = game.get('toi', [0, 0, 0])
        for i in range(3):
            last_5_goals[i] += goals[i]
            last_5_toi[i] += toi[i]

    return {
        'days_since_last_game': days_since_last,
        'avg_days_between_games': avg_days_between,
        'toi_fatigue': toi_fatigue,
        'games_fatigue': games_fatigue,
        'recent_form': recent_form,
        'last_5_goals': last_5_goals,
        'last_5_toi': last_5_toi,
    }


def calculate_days_between_games(game1_date: str, game2_date: str) -> float:
    """Calculate days between two games."""
    date1 = datetime.strptime(game1_date, '%Y-%m-%d')
    date2 = datetime.strptime(game2_date, '%Y-%m-%d')
    return abs((date2 - date1).days)


def update_player_temporal_features(data_graph, player_id: int,
                                    current_game_id: str, current_game_date: str) -> None:
    """
    Update player's node in graph with temporal features.

    Args:
        data_graph: NetworkX graph containing all game data
        player_id: Player's ID
        current_game_id: Current game's ID
        current_game_date: Current game's date (YYYY-MM-DD)
    """
    # Calculate features
    temporal_features = calculate_temporal_features(
        data_graph,
        player_id,
        current_game_id,
        current_game_date
    )

    # Update node in graph
    node_id = f"{current_game_id}_{player_id}"
    if node_id in data_graph:
        # Add temporal features to existing node data
        node_data = data_graph.nodes[node_id]
        node_data.update({
            'temporal_features': temporal_features
        })
        data_graph.nodes[node_id].update(node_data)


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

    if period_index == 2:
        cwc = 0

    # Update win/loss arrays - only one element will be 1
    if home_goals > away_goals:
        home_node['win'] = [0] * 3
        home_node['loss'] = [0] * 3
        away_node['win'] = [0] * 3
        away_node['loss'] = [0] * 3
        home_node['win'][period_index] = 1
        away_node['loss'][period_index] = 1
    else:
        home_node['win'] = [0] * 3
        home_node['loss'] = [0] * 3
        away_node['win'] = [0] * 3
        away_node['loss'] = [0] * 3
        away_node['win'][period_index] = 1
        home_node['loss'][period_index] = 1

