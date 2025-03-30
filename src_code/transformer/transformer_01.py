import os
import pickle
import networkx as nx
import pandas as pd
import json
from collections import defaultdict
from datetime import datetime
from src_code.utils.save_graph_utils import load_filtered_graph


def prepare_hockey_embeddings(config, output_dir=None, format='json'):
    """
    Process hockey graph data into hierarchical embeddings for transformer models.

    Args:
        config: Config object containing settings and graph data
        output_dir: Directory to save the output files (default: uses config path)
        format: Output format - 'json' or 'csv' or 'both'

    Returns:
        Dictionary with paths to the created embedding files
    """
    print("Starting hierarchical embedding extraction...")

    # Set output directory (use config path if not specified)
    if output_dir is None:
        output_dir = config.file_paths['embedding_paths']
    os.makedirs(output_dir, exist_ok=True)

    # Load graph with same filtering logic used in GNN code
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    print(f"Loading graph from {config.file_paths['graph']}")
    if training_cutoff_date:
        print(f"Applying date filter to include only games on or after {training_cutoff_date}")

    data_graph = load_filtered_graph(config.file_paths["graph"], cutoff_date=training_cutoff_date)
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Use window sizes from config
    window_sizes = config.stat_window_sizes
    print(f"Using statistical window sizes from config: {window_sizes}")

    # Create dictionary to store all embedding data
    embeddings = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'player_count': sum(1 for _, data in data_graph.nodes(data=True) if data.get('type') == 'player'),
            'team_count': sum(1 for _, data in data_graph.nodes(data=True) if data.get('type') == 'team'),
            'game_count': sum(1 for _, data in data_graph.nodes(data=True) if data.get('type') == 'game'),
            'window_sizes': window_sizes,
            'cutoff_date': training_cutoff_date.isoformat() if training_cutoff_date else None
        },
        'players': {},  # Player-level embeddings
        'player_pairs': {},  # Player-pair interaction embeddings
        'teams': {},  # Team-level embeddings
        'games': []  # Game sequence embeddings
    }

    # Step 1: Get all games, sorted chronologically
    game_nodes = [(node_id, data['game_date'])
                  for node_id, data in data_graph.nodes(data=True)
                  if data.get('type') == 'game']

    if not game_nodes:
        print("Warning: No game nodes found in the graph. Check that graph has correct node types.")
        print("Available node types:", set(data.get('type', 'unknown') for _, data in data_graph.nodes(data=True)))
        raise ValueError("No game nodes found in the graph")

    game_nodes.sort(key=lambda x: x[1])  # Sort by date
    game_ids = [g[0] for g in game_nodes]

    print(f"Processing {len(game_ids)} games chronologically...")

    # Step 2: Process each game to extract player, pair, and team embeddings
    for game_idx, game_id in enumerate(game_ids):
        if game_idx % 100 == 0:
            print(f"Processing game {game_idx}/{len(game_ids)}")

        game_data = data_graph.nodes[game_id]
        game_date = game_data.get('game_date')
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        # Get Team Game Performance nodes
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        # Process each team
        for team_id, tgp_id in [(home_team, home_tgp), (away_team, away_tgp)]:
            # 2.1: Process player embeddings for this team in this game
            player_embeddings = extract_player_embeddings(data_graph, game_id, team_id, window_sizes, config)

            # 2.2: Process player-pair embeddings for this team in this game
            pair_embeddings = extract_pair_embeddings(data_graph, game_id, team_id, window_sizes, config)

            # 2.3: Process team-level embeddings for this team in this game
            team_embedding = extract_team_embedding(data_graph, tgp_id, window_sizes, config)

            # Store embeddings
            game_key = f"{game_id}_{team_id}"
            embeddings['players'][game_key] = player_embeddings
            embeddings['player_pairs'][game_key] = pair_embeddings
            embeddings['teams'][game_key] = team_embedding

        # 2.4: Create game-level embedding that includes both teams
        game_embedding = {
            'game_id': game_id,
            'date': game_date.isoformat() if isinstance(game_date, datetime) else game_date,
            'home_team': home_team,
            'away_team': away_team,
            # Reference the team embeddings
            'home_team_key': f"{game_id}_{home_team}",
            'away_team_key': f"{game_id}_{away_team}",
            # Game outcome (if available)
            'outcome': extract_game_outcome(data_graph, game_id, home_team, away_team)
        }
        embeddings['games'].append(game_embedding)

    # Step 3: Build sequential game data for each team
    team_sequences = build_team_sequences(embeddings, window_sizes)
    embeddings['team_sequences'] = team_sequences

    # Step 4: Save the embeddings in the requested format
    output_paths = {}

    if format in ['json', 'both']:
        json_path = output_dir + '.json'
        with open(json_path, 'w') as f:
            json.dump(embeddings, f)
        output_paths['json'] = json_path
        print(f"Saved JSON embeddings to {json_path}")

    if format in ['csv', 'both']:
        # For CSV, we need to flatten the hierarchical structure
        csv_paths = export_to_csv(embeddings, output_dir)
        output_paths['csv'] = csv_paths
        print(f"Saved CSV embeddings to {output_dir}")

    print("Hierarchical embedding extraction complete!")
    return


def extract_player_embeddings(graph, game_id, team_id, window_sizes, config):
    """
    Extract player-level embeddings for all players on a team in a specific game.
    Uses config's stat_attributes for player stats.
    """
    player_embeddings = []

    # Find all PGP nodes for this game and team
    pgp_nodes = [
        node for node in graph.nodes()
        if (isinstance(node, str) and
            node.startswith(f"{game_id}_") and
            graph.nodes[node].get('type') == 'player_game_performance' and
            graph.nodes[node].get('player_team') == team_id)
    ]

    # Collect player stat attributes from config
    player_stats = config.stat_attributes['player_stats']

    for pgp_node in pgp_nodes:
        pgp_data = graph.nodes[pgp_node]
        player_id = int(pgp_node.split('_')[1])

        # Basic player information
        player_embedding = {
            'player_id': player_id,
            'player_name': pgp_data.get('player_name', ''),
            'position': pgp_data.get('player_position', ''),
            'team_id': team_id,
            'game_id': game_id,
        }

        # Current game statistics - use stats from config
        for stat in player_stats:
            if stat in pgp_data:
                # Sum across periods (regulation, OT, shootout)
                player_embedding[f'current_{stat}'] = sum(pgp_data.get(stat, [0, 0, 0]))

                # Also keep period-specific data
                for period_idx, period_name in enumerate(['regulation', 'overtime', 'shootout']):
                    try:
                        player_embedding[f'current_{stat}_{period_name}'] = pgp_data.get(stat, [0, 0, 0])[period_idx]
                    except IndexError:
                        player_embedding[f'current_{stat}_{period_name}'] = 0

        # Historical window statistics - use all window sizes from config
        for window in window_sizes:
            for stat in player_stats:
                # Average stats across historical games
                hist_key = f'hist_{window}_{stat}_avg'
                if hist_key in pgp_data:
                    # Average across periods
                    player_embedding[hist_key] = sum(pgp_data.get(hist_key, [0, 0, 0])) / 3

                # Games played in window
                games_key = f'hist_{window}_games'
                if games_key in pgp_data:
                    player_embedding[games_key] = sum(pgp_data.get(games_key, [0, 0, 0]))

        player_embeddings.append(player_embedding)

    return player_embeddings


def extract_pair_embeddings(graph, game_id, team_id, window_sizes, config):
    """
    Extract player-pair interaction embeddings for all player pairs on a team in a specific game.
    Uses config's stat_attributes for player_pair_stats.
    """
    pair_embeddings = []

    # Find all PGP nodes for this game and team
    pgp_nodes = [
        node for node in graph.nodes()
        if (isinstance(node, str) and
            node.startswith(f"{game_id}_") and
            graph.nodes[node].get('type') == 'player_game_performance' and
            graph.nodes[node].get('player_team') == team_id)
    ]

    # Get player pair stats from config
    pair_stats = config.stat_attributes['player_pair_stats']

    # Get all player-pair edges
    for i in range(len(pgp_nodes)):
        pgp1 = pgp_nodes[i]
        player1_id = int(pgp1.split('_')[1])

        for j in range(i + 1, len(pgp_nodes)):
            pgp2 = pgp_nodes[j]
            player2_id = int(pgp2.split('_')[1])

            # Check if there's an edge between these players
            if graph.has_edge(pgp1, pgp2):
                edge_data = graph.get_edge_data(pgp1, pgp2)

                # Only process PGP-PGP edges
                if edge_data.get('type') == 'pgp_pgp_edge':
                    # Basic pair information
                    pair_embedding = {
                        'game_id': game_id,
                        'team_id': team_id,
                        'player1_id': player1_id,
                        'player2_id': player2_id,
                        'player1_name': graph.nodes[pgp1].get('player_name', ''),
                        'player2_name': graph.nodes[pgp2].get('player_name', ''),
                        'player1_position': graph.nodes[pgp1].get('player_position', ''),
                        'player2_position': graph.nodes[pgp2].get('player_position', '')
                    }

                    # Current game interaction statistics - use stats from config
                    for stat in pair_stats:
                        if stat in edge_data:
                            # Sum across periods
                            pair_embedding[f'current_{stat}'] = sum(edge_data.get(stat, [0, 0, 0]))

                            # Period-specific data
                            for period_idx, period_name in enumerate(['regulation', 'overtime', 'shootout']):
                                try:
                                    pair_embedding[f'current_{stat}_{period_name}'] = edge_data.get(stat, [0, 0, 0])[
                                        period_idx]
                                except IndexError:
                                    pair_embedding[f'current_{stat}_{period_name}'] = 0

                    # Historical window statistics - use window sizes from config
                    for window in window_sizes:
                        for stat in pair_stats:
                            # Get historical stats if available
                            hist_key = f'hist_{window}_{stat}'
                            if hist_key in edge_data:
                                # Sum across periods
                                pair_embedding[hist_key] = sum(edge_data.get(hist_key, [0, 0, 0]))

                            # Historical average if available
                            avg_key = f'hist_{window}_{stat}_avg'
                            if avg_key in edge_data:
                                # Average across periods
                                pair_embedding[avg_key] = sum(edge_data.get(avg_key, [0, 0, 0])) / 3

                        # Games played together in window
                        games_key = f'hist_{window}_games'
                        if games_key in edge_data:
                            pair_embedding[games_key] = sum(edge_data.get(games_key, [0, 0, 0]))

                    pair_embeddings.append(pair_embedding)

    return pair_embeddings


def extract_team_embedding(graph, tgp_id, window_sizes, config):
    """
    Extract team-level embedding from a Team Game Performance (TGP) node.
    Uses config's stat_attributes for team_stats.
    """
    if tgp_id not in graph.nodes:
        return {}  # Return empty dict if TGP node doesn't exist

    tgp_data = graph.nodes[tgp_id]
    team_id = tgp_id.split('_')[1]  # Extract team ID from TGP node ID
    game_id = int(tgp_id.split('_')[0])  # Extract game ID from TGP node ID

    # Get team stats from config
    team_stats = config.stat_attributes['team_stats']

    # Basic team information
    team_embedding = {
        'team_id': team_id,
        'game_id': game_id,
        'home': tgp_data.get('home', 0),  # 1 if home team, 0 if away
        'days_since_last_game': tgp_data.get('days_since_last_game', 30)
    }

    # Current game team statistics - use stats from config
    for stat in team_stats:
        if stat in tgp_data:
            # Sum across periods
            team_embedding[f'current_{stat}'] = sum(tgp_data.get(stat, [0, 0, 0]))

            # Period-specific data
            for period_idx, period_name in enumerate(['regulation', 'overtime', 'shootout']):
                try:
                    team_embedding[f'current_{stat}_{period_name}'] = tgp_data.get(stat, [0, 0, 0])[period_idx]
                except IndexError:
                    team_embedding[f'current_{stat}_{period_name}'] = 0

    # Historical window statistics - use window sizes from config
    for window in window_sizes:
        for stat in team_stats:
            # Historical stats
            hist_key = f'hist_{window}_{stat}'
            if hist_key in tgp_data:
                # Sum across periods
                team_embedding[hist_key] = sum(tgp_data.get(hist_key, [0, 0, 0]))

            # Historical average
            avg_key = f'hist_{window}_{stat}_avg'
            if avg_key in tgp_data:
                # Average across periods
                team_embedding[avg_key] = sum(tgp_data.get(avg_key, [0, 0, 0])) / 3

        # Games played in window
        games_key = f'hist_{window}_games'
        if games_key in tgp_data:
            team_embedding[games_key] = sum(tgp_data.get(games_key, [0, 0, 0]))

    return team_embedding


def extract_game_outcome(graph, game_id, home_team, away_team):
    """
    Extract the game outcome (win/loss/tie) from TGP nodes.
    """
    home_tgp = f"{game_id}_{home_team}"
    away_tgp = f"{game_id}_{away_team}"

    if home_tgp in graph.nodes and away_tgp in graph.nodes:
        home_node = graph.nodes[home_tgp]
        away_node = graph.nodes[away_tgp]

        home_win = sum(home_node.get('win', [0, 0, 0]))
        away_win = sum(away_node.get('win', [0, 0, 0]))

        if home_win > 0:
            return 'home_win'
        elif away_win > 0:
            return 'away_win'
        else:
            return 'tie'

    return 'unknown'


def build_team_sequences(embeddings, window_sizes, max_sequence_length=10):
    """
    Build chronological game sequences for each team.
    """
    team_sequences = {}

    # First, organize games by team
    team_games = defaultdict(list)

    for game in embeddings['games']:
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        date = game['date']

        # Add game to home team's list
        team_games[home_team].append({
            'game_id': game_id,
            'date': date,
            'is_home': True,
            'opponent': away_team,
            'team_key': f"{game_id}_{home_team}",
            'outcome': game['outcome']
        })

        # Add game to away team's list
        team_games[away_team].append({
            'game_id': game_id,
            'date': date,
            'is_home': False,
            'opponent': home_team,
            'team_key': f"{game_id}_{away_team}",
            'outcome': game['outcome']
        })

    # Sort each team's games chronologically
    for team, games in team_games.items():
        games.sort(key=lambda x: x['date'])

    # Build sequences for each team
    for team, games in team_games.items():
        team_sequences[team] = []

        # Process each game as a potential sequence end point
        for i in range(len(games)):
            # Get preceding games up to max_sequence_length
            start_idx = max(0, i - max_sequence_length + 1)
            sequence = games[start_idx:i + 1]

            if len(sequence) > 0:
                # Create sequence entry
                seq_entry = {
                    'team_id': team,
                    'sequence_length': len(sequence),
                    'end_game_id': sequence[-1]['game_id'],
                    'end_date': sequence[-1]['date'],
                    'games': sequence
                }

                team_sequences[team].append(seq_entry)

    return team_sequences


def export_to_csv(embeddings, output_dir):
    """
    Export hierarchical embeddings to flattened CSV files.
    """
    csv_paths = {}

    # 1. Export player embeddings
    player_rows = []
    for game_key, players in embeddings['players'].items():
        for player in players:
            player_rows.append(player)

    if player_rows:
        player_df = pd.DataFrame(player_rows)
        player_path = os.path.join(output_dir, 'player_embeddings.csv')
        player_df.to_csv(player_path, index=False)
        csv_paths['players'] = player_path

    # 2. Export player-pair embeddings
    pair_rows = []
    for game_key, pairs in embeddings['player_pairs'].items():
        for pair in pairs:
            pair_rows.append(pair)

    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_path = os.path.join(output_dir, 'player_pair_embeddings.csv')
        pair_df.to_csv(pair_path, index=False)
        csv_paths['player_pairs'] = pair_path

    # 3. Export team embeddings
    team_rows = []
    for game_key, team in embeddings['teams'].items():
        team_rows.append(team)

    if team_rows:
        team_df = pd.DataFrame(team_rows)
        team_path = os.path.join(output_dir, 'team_embeddings.csv')
        team_df.to_csv(team_path, index=False)
        csv_paths['teams'] = team_path

    # 4. Export game embeddings
    game_df = pd.DataFrame(embeddings['games'])
    game_path = os.path.join(output_dir, 'game_embeddings.csv')
    game_df.to_csv(game_path, index=False)
    csv_paths['games'] = game_path

    # 5. Export sequences (more complex - flatten hierarchical structure)
    seq_rows = []
    for team, sequences in embeddings['team_sequences'].items():
        for seq in sequences:
            # For each game in the sequence
            for i, game in enumerate(seq['games']):
                seq_row = {
                    'team_id': team,
                    'sequence_id': f"{team}_{seq['end_game_id']}",
                    'sequence_position': i,
                    'sequence_length': seq['sequence_length'],
                    'game_id': game['game_id'],
                    'is_home': game['is_home'],
                    'opponent': game['opponent'],
                    'game_date': game['date'],
                    'game_outcome': game['outcome'],
                    'is_sequence_end': (i == seq['sequence_length'] - 1)
                }
                seq_rows.append(seq_row)

    if seq_rows:
        seq_df = pd.DataFrame(seq_rows)
        seq_path = os.path.join(output_dir, 'sequence_embeddings.csv')
        seq_df.to_csv(seq_path, index=False)
        csv_paths['sequences'] = seq_path

    return csv_paths