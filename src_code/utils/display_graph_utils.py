import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from matplotlib.patches import Rectangle, Circle


def visualize_game_graph(data_graph, game_id, output_path=None, edge_sample_rate=0.05):
    """
    Create a detailed visualization of a specific game's subgraph showing all related nodes and statistics.

    Args:
        data_graph: NetworkX graph containing all game data
        game_id: ID of the game to visualize
        output_path: Path to save the JPG file
        edge_sample_rate: Fraction of PGP-to-PGP edges to display (default: 0.05 or 5%)
    """
    # Create a subgraph containing the game and all related nodes
    game_nodes = {game_id}
    game_nodes.update(nx.neighbors(data_graph, game_id))

    # Add player nodes and team nodes
    for node in list(game_nodes):
        if data_graph.nodes[node].get('type') in ['player_game_performance', 'team_game_performance']:
            game_nodes.update(nx.neighbors(data_graph, node))

    game_subgraph = data_graph.subgraph(game_nodes)

    # Create figure
    plt.figure(figsize=(30, 45), dpi=300)

    # Create layout with nodes properly spaced
    pos = create_hierarchical_layout(game_subgraph, game_id)

    # Draw different types of nodes with different colors and sizes
    node_colors = {
        'game': '#FF9999',
        'team': '#99FF99',
        'player': '#9999FF',
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

    # Draw edges with different styles based on type
    edge_styles = {
        'player_game_performance_team_game_performance_edge': {'color': 'blue', 'style': 'solid', 'width': 2},
        'player_game_performance_player_edge': {'color': 'green', 'style': 'dashed', 'width': 2}
    }

    # Draw regular edges
    for edge_type, style in edge_styles.items():
        edges = [(u, v) for (u, v, d) in game_subgraph.edges(data=True)
                 if d.get('type') == edge_type]
        nx.draw_networkx_edges(game_subgraph, pos,
                               edgelist=edges,
                               edge_color=style['color'],
                               style=style['style'],
                               width=style['width'],
                               alpha=0.5)

    # Get PGP-to-PGP edges and sample them
    pgp_edges = [(u, v) for (u, v, d) in game_subgraph.edges(data=True)
                 if (game_subgraph.nodes[u].get('type') == 'player_game_performance' and
                     game_subgraph.nodes[v].get('type') == 'player_game_performance' and
                     'goal' in d)]

    # Randomly sample edges
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
            total_goals = sum(edge_data.get('goal', [0, 0, 0]))
            total_toi = sum(edge_data.get('toi', [0, 0, 0]))
            if total_goals > 0 or total_toi > 0:
                edge_labels[(u, v)] = f'G:{total_goals}\nTOI:{total_toi:.1f}'

    nx.draw_networkx_edge_labels(game_subgraph, pos,
                                 edge_labels=edge_labels,
                                 font_size=8)

    # Add node labels with statistics
    labels = {}
    for node in game_subgraph.nodes():
        node_data = game_subgraph.nodes[node]
        label = f"{node}\n"

        if node_data.get('type') == 'player_game_performance':
            stats = ['player_position',
                     'toi', 'goal', 'assist', 'point', 'faceoff', 'faceoff_won',
                     'shot_on_goal', 'shot_blocked', 'shot_saved',
                     'goal_saved', 'goal_against']
            for stat in stats:
                if stat in node_data:
                    if isinstance(node_data[stat], list):
                        total = sum(node_data[stat])
                        label += f"\n{stat}: {total}"
                    else:
                        label += f"\n{stat}: {node_data[stat]}"
        elif node_data.get('type') == 'team_game_performance':
            # Enhanced team stats
            stats = [
                'home', 'days_since_last_game', 'valid',
                'goal', 'goal_against', 'shot_attempt', 'shot_on_goal', 'shot_blocked', 'shot_saved', 'faceoff_taken',
                'faceoff_won', 'hit_another_player', 'hit_by_player',
                'penalties_duration', 'win', 'loss'
            ]
            for stat in stats:
                if stat in node_data:
                    if isinstance(node_data[stat], list):
                        total = sum(node_data[stat])
                        label += f"\n{stat}: {total}"
                    else:
                        label += f"\n{stat}: {node_data[stat]}"

        labels[node] = label

    # Add labels with increased font size
    nx.draw_networkx_labels(game_subgraph, pos, labels, font_size=10)

    # Add title
    game_date = data_graph.nodes[game_id].get('game_date', '')
    plt.title(f"Game {game_id} Network Graph - {game_date}\n(Showing {edge_sample_rate * 100}% of player interactions)",
              pad=20, size=24)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, marker='o',
                   label=node_type, markersize=15, linestyle='None')
        for node_type, color in node_colors.items()
    ]
    legend_elements.extend([
        plt.Line2D([0], [0], color=style['color'], linestyle=style['style'],
                   label=edge_type.replace('_', ' ').title(), linewidth=2)
        for edge_type, style in edge_styles.items()
    ])
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, format='jpg', bbox_inches='tight', dpi=300)
    plt.close()


def create_hierarchical_layout(graph, game_id):
    """
    Create a hierarchical layout with the game node at the center.
    """
    pos = {}

    # Place game node at center
    pos[game_id] = np.array([0, 0])

    # Get different types of nodes
    team_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'team']
    tgp_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'team_game_performance']
    pgp_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'player_game_performance']
    player_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'player']

    # Place team nodes in a circle around the game node
    n_teams = len(team_nodes)
    for i, node in enumerate(team_nodes):
        angle = 2 * np.pi * i / n_teams
        pos[node] = np.array([4 * np.cos(angle), 4 * np.sin(angle)])

    # Place TGP nodes between game and their respective teams
    game_data = graph.nodes[game_id]
    home_team = game_data.get('home_team')
    away_team = game_data.get('away_team')

    for node in tgp_nodes:
        node_team = node.split('_')[1]
        team_pos = None
        if node_team == home_team:
            team_pos = pos.get(home_team)
        elif node_team == away_team:
            team_pos = pos.get(away_team)

        if team_pos is not None:
            pos[node] = (pos[game_id] + team_pos) / 2
        else:
            angle = 2 * np.pi * tgp_nodes.index(node) / len(tgp_nodes)
            pos[node] = np.array([2 * np.cos(angle), 2 * np.sin(angle)])

    # Place PGP nodes in an outer circle with more spacing
    n_pgp = len(pgp_nodes)
    radius = 8  # Increased radius for better spacing
    for i, node in enumerate(pgp_nodes):
        angle = 2 * np.pi * i / n_pgp
        pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Place player nodes in the outermost circle with even more spacing
    n_players = len(player_nodes)
    outer_radius = 12  # Increased radius for better spacing
    for i, node in enumerate(player_nodes):
        angle = 2 * np.pi * i / n_players
        pos[node] = np.array([outer_radius * np.cos(angle), outer_radius * np.sin(angle)])

    return pos