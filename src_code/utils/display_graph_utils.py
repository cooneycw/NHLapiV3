import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


def visualize_game_graph(data_graph, game_id, window_size=10, output_path=None, edge_sample_rate=0.05):
    """
    Create a detailed visualization of a specific game's subgraph showing TGP and PGP nodes,
    with historical aggregated data and sampled player interactions.
    Now uses player names directly from PGP nodes.

    Args:
        data_graph: NetworkX graph containing all game data
        game_id: ID of the game to visualize
        output_path: Path to save the JPG file
        edge_sample_rate: Fraction of PGP-to-PGP edges to display (default: 0.05 or 5%)
        window_size: Size of the window for historical statistics (default: 10 games)
    """
    # Get game data to identify teams
    game_data = data_graph.nodes[game_id]
    home_team = game_data.get('home_team')
    away_team = game_data.get('away_team')
    selected_team = random.choice([home_team, away_team])

    # Create a subgraph containing only game, team, TGP, and PGP nodes
    game_nodes = {game_id}

    # Add team nodes
    team_nodes = [n for n in nx.neighbors(data_graph, game_id)
                  if data_graph.nodes[n].get('type') == 'team']
    game_nodes.update(team_nodes)

    # Add TGP nodes
    tgp_nodes = [n for n in nx.neighbors(data_graph, game_id)
                 if data_graph.nodes[n].get('type') == 'team_game_performance']
    game_nodes.update(tgp_nodes)

    # Add PGP nodes only for selected team
    for tgp in tgp_nodes:
        if selected_team in tgp:
            pgp_nodes = [n for n in nx.neighbors(data_graph, tgp)
                         if data_graph.nodes[n].get('type') == 'player_game_performance']
            game_nodes.update(pgp_nodes)

    game_subgraph = data_graph.subgraph(game_nodes)

    # Create figure
    plt.figure(figsize=(30, 45), dpi=300)

    # Create simplified layout without player nodes
    pos = create_hierarchical_layout(game_subgraph, game_id)

    # Draw different types of nodes
    node_colors = {
        'game': '#FF9999',
        'team': '#99FF99',
        'team_game_performance': '#FFCC99',
        'player_game_performance': '#99FFCC'
    }

    # Draw nodes
    for node_type, color in node_colors.items():
        nodes = [n for n in game_subgraph.nodes() if game_subgraph.nodes[n].get('type') == node_type]
        nx.draw_networkx_nodes(game_subgraph, pos,
                               nodelist=nodes,
                               node_color=color,
                               node_size=6000,
                               alpha=0.7)

    # Draw edges
    edge_styles = {
        'player_game_performance_team_game_performance_edge': {'color': 'blue', 'style': 'solid', 'width': 2}
    }

    for edge_type, style in edge_styles.items():
        edges = [(u, v) for (u, v, d) in game_subgraph.edges(data=True)
                 if d.get('type') == edge_type]
        nx.draw_networkx_edges(game_subgraph, pos,
                               edgelist=edges,
                               edge_color=style['color'],
                               style=style['style'],
                               width=style['width'],
                               alpha=0.5)

    # Get and sample PGP-to-PGP edges
    pgp_edges = [(u, v) for (u, v, d) in game_subgraph.edges(data=True)
                 if (game_subgraph.nodes[u].get('type') == 'player_game_performance' and
                     game_subgraph.nodes[v].get('type') == 'player_game_performance' and
                     'goal' in d)]

    if pgp_edges:
        num_edges_to_show = max(1, int(len(pgp_edges) * edge_sample_rate))
        sampled_edges = random.sample(pgp_edges, num_edges_to_show)

        nx.draw_networkx_edges(game_subgraph, pos,
                               edgelist=sampled_edges,
                               edge_color='purple',
                               style='solid',
                               width=2,
                               alpha=0.3)

        # Add edge labels for sampled PGP-to-PGP edges
        edge_labels = {}
        for u, v in sampled_edges:
            edge_data = game_subgraph.get_edge_data(u, v)
            if edge_data:
                goals = edge_data.get('goal', [0, 0, 0])
                toi = edge_data.get('toi', [0, 0, 0])

                # Get games played count for context
                games = edge_data.get(f'hist_{window_size}_games', [0, 0, 0])

                # Get historical stats (assuming these are now stored as averages)
                hist_goal = edge_data.get(f'hist_{window_size}_goal_avg', [0, 0, 0])
                hist_asst = edge_data.get(f'hist_{window_size}_assist_avg', [0, 0, 0])
                hist_shot = edge_data.get(f'hist_{window_size}_shot_on_goal_avg', [0, 0, 0])
                hist_toi = edge_data.get(f'hist_{window_size}_toi_avg', [0, 0, 0])

                # Current game info
                label = f'Current - Goals:{goals}\nTOI:{toi}\n'

                # Historical stats (implicitly averages)
                if sum(games) > 0:
                    label += f'\nHist(Reg:{games[0]}/OT:{games[1]}/SO:{games[2]}\n'
                    label += f'Goals:{[round(x, 2) for x in hist_goal]}\n'
                    label += f'Assists:{[round(x, 2) for x in hist_asst]}\n'
                    label += f'TOI:{[round(x, 2) for x in hist_toi]}\n'
                    label += f'Shots on Goal:{[round(x, 2) for x in hist_shot]}\n'

                edge_labels[(u, v)] = label

        nx.draw_networkx_edge_labels(game_subgraph, pos,
                                     edge_labels=edge_labels,
                                     font_size=8)

    # Add node labels with enhanced statistics and player names
    labels = {}
    for node in game_subgraph.nodes():
        node_data = game_subgraph.nodes[node]

        # Start with node name
        label = f"{node}\n"

        if node_data.get('type') == 'player_game_performance':
            # Use the player name directly from the PGP node
            player_name = node_data.get('player_name', '')

            # If no combined name, try to construct from first/last name
            if not player_name:
                first_name = node_data.get('player_first_name', '')
                last_name = node_data.get('player_last_name', '')
                if first_name or last_name:
                    player_name = f"{first_name} {last_name}".strip()

            # Add player name at the top of the label if available
            if player_name:
                label = f"{player_name}\n{node}\n"

            # Current game stats
            current_stats = ['player_position',
                             'toi', 'goal', 'assist', 'faceoff_taken', 'faceoff_won',
                             'shot_on_goal', 'shot_blocked', 'shot_saved',
                             'goal_saved', 'goal_against']

            label += "\nCurrent Game:"
            for stat in current_stats:
                if stat in node_data:
                    label += f"\n{stat}: {node_data[stat]}"

            # Historical stats
            hist_games = node_data.get(f'hist_{window_size}_games')
            if hist_games:
                label += f"\n\nHistorical (Reg:{hist_games[0]}/OT:{hist_games[1]}/SO:{hist_games[2]}):"

                # Include key stats (now just averages without the _avg suffix)
                for stat in ['toi', 'goal', 'goal_against', 'assist', 'faceoff_won', 'shot_on_goal']:
                    hist_stat = node_data.get(f'hist_{window_size}_{stat}_avg')
                    if hist_stat:
                        label += f"\n{stat}_avg: {[round(x, 2) for x in hist_stat]}"

        elif node_data.get('type') == 'team_game_performance':
            current_stats = [
                'home', 'win', 'loss', 'days_since_last_game', 'valid',
                'goal', 'goal_against', 'shot_attempt', 'shot_on_goal',
                'shot_blocked', 'shot_saved', 'faceoff_taken',
                'faceoff_won', 'hit_another_player', 'hit_by_player',
                'penalties_duration'
            ]

            label += "\nCurrent Game:"
            for stat in current_stats:
                if stat in node_data:
                    label += f"\n{stat}: {node_data[stat]}"

            # Historical team stats
            hist_games = node_data.get(f'hist_{window_size}_games')
            if hist_games:
                label += f"\n\nHistorical (Reg:{hist_games[0]}/OT:{hist_games[1]}/SO:{hist_games[2]}):"

                # Include key stats (now just averages without the _avg suffix)
                for stat in ['win', 'loss', 'goal', 'goal_against', 'shot_attempt', 'shot_on_goal', 'faceoff_won']:
                    hist_stat = node_data.get(f'hist_{window_size}_{stat}_avg')
                    if hist_stat:
                        label += f"\n{stat}: {[round(x, 2) for x in hist_stat]}"

        labels[node] = label

    # Add labels
    nx.draw_networkx_labels(game_subgraph, pos, labels, font_size=10)

    # Add title
    game_date = data_graph.nodes[game_id].get('game_date', '')
    plt.title(f"Game {game_id} Network Graph - {game_date}\n"
              f"Showing {selected_team} players only\n"
              f"({edge_sample_rate * 100}% of player interactions and {window_size}-game historical averages)",
              pad=20, size=24)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, marker='o',
                   label=node_type.replace('_', ' ').title(), markersize=15, linestyle='None')
        for node_type, color in node_colors.items()
    ]
    legend_elements.extend([
        plt.Line2D([0], [0], color=style['color'], linestyle=style['style'],
                   label='PGP-TGP Connection', linewidth=2)
        for edge_type, style in edge_styles.items()
    ])
    legend_elements.append(
        plt.Line2D([0], [0], color='purple', linestyle='solid',
                   label='Player Interaction', linewidth=2, alpha=0.3)
    )

    # Add a legend section for period types
    legend_elements.append(
        plt.Line2D([0], [0], color='white', marker='', linestyle='',
                   label='\nPeriod Types:', markersize=0)
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='white', marker='', linestyle='',
                   label='Reg: Regulation', markersize=0)
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='white', marker='', linestyle='',
                   label='OT: Overtime', markersize=0)
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='white', marker='', linestyle='',
                   label='SO: Shootout', markersize=0)
    )

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)
    plt.close()


def create_hierarchical_layout(graph, game_id):
    """
    Create a simplified hierarchical layout without player nodes.
    """
    pos = {}

    # Place game node at center
    pos[game_id] = np.array([0, 0])

    # Get different types of nodes
    team_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'team']
    tgp_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'team_game_performance']
    pgp_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'player_game_performance']

    # Place team nodes in a circle around the game node
    n_teams = len(team_nodes)
    for i, node in enumerate(team_nodes):
        angle = 2 * np.pi * i / n_teams
        pos[node] = np.array([4 * np.cos(angle), 4 * np.sin(angle)])

    # Place TGP nodes between game and their respective teams
    for node in tgp_nodes:
        team_pos = None
        for team in team_nodes:
            if team in node:
                team_pos = pos[team]
                break
        if team_pos is not None:
            pos[node] = (pos[game_id] + team_pos) / 2
        else:
            angle = 2 * np.pi * tgp_nodes.index(node) / len(tgp_nodes)
            pos[node] = np.array([2 * np.cos(angle), 2 * np.sin(angle)])

    # Place PGP nodes in an outer circle with more spacing
    n_pgp = len(pgp_nodes)
    if n_pgp > 0:
        radius = 8
        for i, node in enumerate(pgp_nodes):
            angle = 2 * np.pi * i / n_pgp
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    return pos