import json
import os
import pandas as pd


def load_hockey_embeddings(config, format='both'):
    """
    Load hockey embeddings prepared by the prepare_hockey_embeddings function.

    Args:
        config: Config object containing file paths
        format: Which format to load ('json', 'csv', or 'both')

    Returns:
        Dictionary containing the loaded embeddings
    """
    embeddings = {}

    # JSON format contains all data in a single file
    if format in ['json', 'both']:
        json_path = config.file_paths['embedding_paths'] + '.json'
        if os.path.exists(json_path):
            print(f"Loading JSON embeddings from {json_path}")
            with open(json_path, 'r') as f:
                embeddings['json'] = json.load(f)
                print(f"Successfully loaded JSON embeddings with {len(embeddings['json']['games'])} games")
        else:
            print(f"Warning: JSON embeddings file not found at {json_path}")

    # CSV format has multiple files with flattened data structure
    if format in ['csv', 'both']:
        csv_dir = config.file_paths['embedding_paths']
        embeddings['csv'] = {}

        # Check if directory exists
        if not os.path.exists(csv_dir):
            print(f"Warning: CSV embeddings directory not found at {csv_dir}")
            os.makedirs(csv_dir, exist_ok=True)

        # Load each CSV file
        csv_files = {
            'players': 'player_embeddings.csv',
            'player_pairs': 'player_pair_embeddings.csv',
            'teams': 'team_embeddings.csv',
            'games': 'game_embeddings.csv',
            'sequences': 'sequence_embeddings.csv'
        }

        for key, filename in csv_files.items():
            filepath = os.path.join(csv_dir, filename)
            if os.path.exists(filepath):
                embeddings['csv'][key] = pd.read_csv(filepath)
                print(f"Loaded {key} embeddings: {len(embeddings['csv'][key])} records")
            else:
                print(f"Warning: {key} embeddings file not found at {filepath}")

    return embeddings

