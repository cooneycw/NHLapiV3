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
    graph.add_node(player, type = 'player', **player_dict[player])


def add_game(graph, game):
    game_id = game['id']
    game_date = game['game_date']
    graph.add_node(game_id, type = 'game', game_date=game_date)
    graph.add_edge(game['awayTeam'], game_id, home = 0)
    graph.add_edge(game['homeTeam'], game_id, home=1)
    away_tgp = str(game_id) + '_' + game['awayTeam']
    home_tgp = str(game_id) + '_' +game['homeTeam']
    graph.add_node(away_tgp, type = 'team_game_performance')
    graph.add_node(home_tgp, type = 'team_game_performance')
    graph.add_edge(away_tgp, game['awayTeam'])
    graph.add_edge(home_tgp, game['homeTeam'])
    graph.add_edge(away_tgp, game_id)
    graph.add_edge(home_tgp, game_id)


def add_player_game_performance(graph, roster):
    for player in roster:
        game_id = player['game_id']
        player_id = player['player_id']
        player_team = player['player_team']
        player_pgp = str(game_id) + '_' + str(player_id)
        team_tgp = str(game_id) + '_' + player_team
        graph.add_node(player_pgp, type = 'player_game_performance')
        graph.add_edge(player_pgp, player_id)
        graph.add_edge(player_pgp, team_tgp)


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