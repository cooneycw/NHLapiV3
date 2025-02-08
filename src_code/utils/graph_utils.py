import copy
import matplotlib.pyplot as plt
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

    graph.add_node(away_tgp, type = 'team_game_performance', **away_stats)
    graph.add_node(home_tgp, type = 'team_game_performance', **home_stats)
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
    tgp_node['faceoff_taken'][period_code] += stat_dict['faceoff_taken'][period_code]
    tgp_node['faceoff_won'][period_code] += stat_dict['faceoff_won'][period_code]
    tgp_node['shot_on_goal'][period_code] += stat_dict['shot_on_goal'][period_code]
    tgp_node['shot_saved'][period_code] += stat_dict['shot_saved'][period_code]
    tgp_node['goal'][period_code] += stat_dict['goal'][period_code]
    tgp_node['hit_another_player'][period_code] += stat_dict['hit_another_player'][period_code]
    tgp_node['hit_by_player'][period_code] += stat_dict['hit_by_player'][period_code]
    tgp_node['penalties_duration'][period_code] += stat_dict['penalties_duration'][period_code]


def update_pgp_stats(graph, player_pgp, period_code, stat_dict):
    """Update stats for a Player Game Performance node."""
    pgp_node = graph.nodes[player_pgp]
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


def show_single_game_trimmed(graph, game_id):
    """
    Show only the subgraph for one game_id, containing:
      - The game node itself
      - TGP and PGP nodes for that game
      - Any neighbors of those TGP/PGP nodes (teams/players)
    """
    # 1) Identify TGP nodes for this game
    tgp_nodes_for_game = [
        n for n, d in graph.nodes(data=True)
        if d.get("type") == "team_game_performance"
           and n.startswith(f"{game_id}_")
    ]
    # 2) Identify PGP nodes for this game
    pgp_nodes_for_game = [
        n for n, d in graph.nodes(data=True)
        if d.get("type") == "player_game_performance"
           and n.startswith(f"{game_id}_")
    ]

    # 3) Keep the game node itself
    keep_nodes = set([game_id])

    # 4) Add TGP and PGP nodes for that game
    keep_nodes.update(tgp_nodes_for_game)
    keep_nodes.update(pgp_nodes_for_game)

    # 5) Also include neighbors of those TGP/PGP nodes
    #    (which should be the actual team or player nodes)
    for node in (tgp_nodes_for_game + pgp_nodes_for_game):
        keep_nodes.update(graph.neighbors(node))

    # 6) Build the subgraph
    single_game_subgraph = graph.subgraph(keep_nodes).copy()

    # ----- Drawing -----
    # You can reuse the color scheme from your existing code
    # or do something simpler:
    pos = nx.spring_layout(single_game_subgraph, seed=42)

    plt.figure(figsize=(8, 8))
    plt.title(f"Game {game_id} (trimmed to TGP/PGP)")

    # Draw edges
    nx.draw_networkx_edges(single_game_subgraph, pos, edge_color='gray')

    # Separate node sets in this subgraph
    node_types = nx.get_node_attributes(single_game_subgraph, 'type')
    team_nodes = [n for n, t in node_types.items() if t == 'team']
    player_nodes = [n for n, t in node_types.items() if t == 'player']
    game_nodes = [n for n, t in node_types.items() if t == 'game']
    tgp_nodes = [n for n, t in node_types.items() if t == 'team_game_performance']
    pgp_nodes = [n for n, t in node_types.items() if t == 'player_game_performance']

    # Draw each set with a distinct color/size
    nx.draw_networkx_nodes(
        single_game_subgraph, pos,
        nodelist=team_nodes,
        node_color='red', node_size=1200,
        label='Teams'
    )
    nx.draw_networkx_nodes(
        single_game_subgraph, pos,
        nodelist=player_nodes,
        node_color='skyblue', node_size=300,
        edgecolors='black', label='Players'
    )
    nx.draw_networkx_nodes(
        single_game_subgraph, pos,
        nodelist=game_nodes,
        node_color='gold', node_size=800,
        edgecolors='black', label='Game'
    )
    nx.draw_networkx_nodes(
        single_game_subgraph, pos,
        nodelist=tgp_nodes,
        node_color='green', node_size=500,
        edgecolors='black', label='TGP'
    )
    nx.draw_networkx_nodes(
        single_game_subgraph, pos,
        nodelist=pgp_nodes,
        node_color='pink', node_size=300,
        edgecolors='black', label='PGP'
    )

    # Optional: Add labels (could be refined, but here's the idea)
    nx.draw_networkx_labels(single_game_subgraph, pos, font_size=8)

    plt.axis("off")
    plt.show()