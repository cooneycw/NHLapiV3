import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import networkx as nx
import pickle
import os
from datetime import datetime, timedelta
import gc


def extract_features_from_graph(graph, config, cutoff_date=None):
    """
    Extract features from a NetworkX graph into a wide table format suitable for ML.

    Args:
        graph: NetworkX graph containing hockey game data
        config: Configuration object with settings
        cutoff_date: Optional datetime to split data

    Returns:
        DataFrame with one row per game and features as columns
    """
    print("Extracting features from graph...")

    # Create list to hold game data rows
    game_rows = []

    # Get sorted list of games by date
    game_nodes = [
        (game_id, data) for game_id, data in graph.nodes(data=True)
        if data.get('type') == 'game'
    ]

    # Filter by date if needed
    if cutoff_date:
        game_nodes = [
            (game_id, data) for game_id, data in game_nodes
            if data.get('game_date') and data['game_date'] < cutoff_date
        ]

    # Process each game
    for game_idx, (game_id, game_data) in enumerate(game_nodes):
        if game_idx % 100 == 0:
            print(f"Processing game {game_idx + 1}/{len(game_nodes)}")
            gc.collect()  # Periodic garbage collection

        # Basic game information
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        game_date = game_data.get('game_date')

        if not all([home_team, away_team, game_date]):
            print(f"Skipping game {game_id}: missing basic information")
            continue

        # Get TGP nodes
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        # Check if TGP nodes exist
        if home_tgp not in graph.nodes or away_tgp not in graph.nodes:
            print(f"Skipping game {game_id}: missing TGP nodes")
            continue

        # Get team performance data
        home_tgp_data = graph.nodes[home_tgp]
        away_tgp_data = graph.nodes[away_tgp]

        # Extract target variable: home team win
        home_win = sum(home_tgp_data.get('win', [0, 0, 0])) > 0

        # Create a row for this game
        row = {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_win': 1 if home_win else 0,
        }

        # Extract features from TGP nodes

        # 1. Game-specific features
        for prefix, tgp_data in [('home_', home_tgp_data), ('away_', away_tgp_data)]:
            # Direct game stats
            stat_attributes = config.stat_attributes['team_stats']
            for stat in stat_attributes:
                if stat in tgp_data:
                    # Sum across periods (regulation, overtime, shootout)
                    row[f'{prefix}{stat}'] = sum(tgp_data[stat])

            # Days since last game
            row[f'{prefix}days_since_last_game'] = tgp_data.get('days_since_last_game', 0)

            # Add home/away indicator
            row[f'{prefix}is_home'] = 1 if prefix == 'home_' else 0

        # 2. Historical features for each team (using window sizes from config)
        for window_size in config.stat_window_sizes:
            for prefix, tgp_data in [('home_', home_tgp_data), ('away_', away_tgp_data)]:
                # Add historical stats
                for stat in stat_attributes:
                    hist_key = f'hist_{window_size}_{stat}'
                    hist_avg_key = f'{hist_key}_avg'

                    if hist_key in tgp_data:
                        # Sum across periods
                        row[f'{prefix}hist_{window_size}_{stat}'] = sum(tgp_data[hist_key])

                    if hist_avg_key in tgp_data:
                        # Average (already calculated, but need the period-weighted average)
                        games_key = f'hist_{window_size}_games'
                        games = tgp_data.get(games_key, [0, 0, 0])
                        total_games = sum(games)

                        if total_games > 0:
                            # Calculate weighted average across periods
                            weighted_avg = sum(tgp_data[hist_avg_key][i] * games[i]
                                               for i in range(3)) / total_games
                            row[f'{prefix}hist_{window_size}_{stat}_avg'] = weighted_avg
                        else:
                            row[f'{prefix}hist_{window_size}_{stat}_avg'] = 0

        # 3. Team strength features (aggregating across player performance)
        for prefix, team in [('home_', home_team), ('away_', away_team)]:
            # Get all player-game-performance nodes for this team in this game
            pgp_nodes = [
                node for node in graph.nodes()
                if isinstance(node, str) and
                   node.startswith(f"{game_id}_") and
                   graph.nodes[node].get('type') == 'player_game_performance' and
                   graph.nodes[node].get('player_team') == team
            ]

            # Count players by position
            position_counts = {'F': 0, 'D': 0, 'G': 0}
            for pgp in pgp_nodes:
                pos = graph.nodes[pgp].get('player_position', '')
                if pos in ['C', 'R', 'L']:  # Forward positions
                    position_counts['F'] += 1
                elif pos == 'D':
                    position_counts['D'] += 1
                elif pos == 'G':
                    position_counts['G'] += 1

            for pos, count in position_counts.items():
                row[f'{prefix}count_{pos}'] = count

            # Aggregated player stats
            player_stats = config.stat_attributes['player_stats']

            # Initialize aggregates
            for stat in player_stats:
                row[f'{prefix}player_total_{stat}'] = 0
                row[f'{prefix}player_avg_{stat}'] = 0
                row[f'{prefix}player_max_{stat}'] = 0

            # Collect values
            stat_values = {stat: [] for stat in player_stats}

            # Aggregate across players
            for pgp in pgp_nodes:
                pgp_data = graph.nodes[pgp]

                for stat in player_stats:
                    if stat in pgp_data:
                        stat_total = sum(pgp_data[stat])
                        row[f'{prefix}player_total_{stat}'] += stat_total
                        stat_values[stat].append(stat_total)

            # Calculate averages and maximums
            for stat in player_stats:
                if stat_values[stat]:
                    row[f'{prefix}player_avg_{stat}'] = np.mean(stat_values[stat])
                    row[f'{prefix}player_max_{stat}'] = max(stat_values[stat])

        # 4. Player interaction features (from pgp-pgp edges)
        for prefix, team in [('home_', home_team), ('away_', away_team)]:
            # Get pair edges for this team in this game
            pgp_edges = []
            for u, v, data in graph.edges(data=True):
                if (isinstance(u, str) and isinstance(v, str) and
                        u.startswith(f"{game_id}_") and v.startswith(f"{game_id}_") and
                        data.get('type') == 'pgp_pgp_edge' and
                        graph.nodes[u].get('player_team') == team and
                        graph.nodes[v].get('player_team') == team):
                    pgp_edges.append((u, v, data))

            # Collect pair statistics
            pair_stats = config.stat_attributes['player_pair_stats']

            for stat in pair_stats:
                pair_values = []
                for u, v, data in pgp_edges:
                    if stat in data:
                        pair_values.append(sum(data[stat]))

                if pair_values:
                    row[f'{prefix}pair_avg_{stat}'] = np.mean(pair_values)
                    row[f'{prefix}pair_max_{stat}'] = max(pair_values)
                    row[f'{prefix}pair_sum_{stat}'] = sum(pair_values)
                else:
                    row[f'{prefix}pair_avg_{stat}'] = 0
                    row[f'{prefix}pair_max_{stat}'] = 0
                    row[f'{prefix}pair_sum_{stat}'] = 0

        # Add this game's row to our dataset
        game_rows.append(row)

    # Create DataFrame from rows
    df = pd.DataFrame(game_rows)

    # Set game_id as index but keep it as a column too
    df = df.set_index('game_id', drop=False)

    # Add derived features
    print("Adding derived features...")

    # Sort by date for team sequences
    df = df.sort_values('game_date')

    # Team matchup features
    df['team_matchup'] = df['home_team'] + '_vs_' + df['away_team']

    # Difference features
    for stat in config.stat_attributes['team_stats']:
        if f'home_{stat}' in df.columns and f'away_{stat}' in df.columns:
            df[f'diff_{stat}'] = df[f'home_{stat}'] - df[f'away_{stat}']

    # Historical win rate difference (using the largest window)
    largest_window = max(config.stat_window_sizes)
    if f'home_hist_{largest_window}_win' in df.columns and f'home_hist_{largest_window}_games' in df.columns:
        df['home_win_rate'] = df[f'home_hist_{largest_window}_win'] / df[f'home_hist_{largest_window}_games'].clip(1)
    else:
        df['home_win_rate'] = 0.5

    if f'away_hist_{largest_window}_win' in df.columns and f'away_hist_{largest_window}_games' in df.columns:
        df['away_win_rate'] = df[f'away_hist_{largest_window}_win'] / df[f'away_hist_{largest_window}_games'].clip(1)
    else:
        df['away_win_rate'] = 0.5

    df['win_rate_diff'] = df['home_win_rate'] - df['away_win_rate']

    # Fill NAs with zeros
    df = df.fillna(0)

    print(f"Created dataset with {len(df)} games and {len(df.columns)} features")
    return df


def train_logistic_regression_model(df, config):
    """
    Train a logistic regression model on the prepared dataset.

    Args:
        df: DataFrame with features and target variable
        config: Configuration object with settings

    Returns:
        Trained model, feature list, and evaluation metrics
    """
    print("Training logistic regression model...")

    # Define target and features
    y = df['home_win']

    # Drop columns that shouldn't be features
    cols_to_drop = ['game_id', 'game_date', 'home_team', 'away_team',
                    'home_win', 'team_matchup']
    X = df.drop(columns=cols_to_drop, errors='ignore')

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Create pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Define hyperparameter grid
    param_grid = {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__solver': ['liblinear', 'saga']
    }

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Print metrics
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Get feature importances
    logreg = best_model.named_steps['logreg']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(logreg.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print("\nTop 20 most important features:")
    print(feature_importance.head(20))

    return best_model, X.columns, {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'feature_importance': feature_importance
    }


def create_and_train_hockey_prediction_model(config_file_path):
    """
    Main function to load graph data, create features, and train model.

    Args:
        config_file_path: Path to the configuration file
    """
    # Load config
    with open(config_file_path, 'rb') as f:
        config = pickle.load(f)

    # Set training data cutoff date (if needed)
    cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    # Load graph
    print(f"Loading graph from {config.file_paths['graph']}...")
    with open(config.file_paths['graph'], 'rb') as f:
        graph = pickle.load(f)

    print(f"Graph loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Extract features
    df = extract_features_from_graph(graph, config, cutoff_date)

    # Save features to CSV for inspection
    features_file = os.path.join(config.file_paths['gnn_analysis'], 'hockey_features.csv')
    df.to_csv(features_file)
    print(f"Features saved to {features_file}")

    # Train model
    model, features, metrics = train_logistic_regression_model(df, config)

    # Save model and results
    model_file = os.path.join(config.file_paths['gnn_analysis'], 'hockey_logreg_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'features': features, 'metrics': metrics}, f)

    print(f"Model saved to {model_file}")

    # Save feature importance plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    top_features = metrics['feature_importance'].head(20)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plot_file = os.path.join(config.file_paths['gnn_analysis'], 'feature_importance.png')
    plt.savefig(plot_file)
    print(f"Feature importance plot saved to {plot_file}")

    return df, model, metrics


def make_game_predictions(model_path, config_path, games_to_predict=None):
    """
    Make predictions for upcoming or specific games.

    Args:
        model_path: Path to the saved model file
        config_path: Path to the configuration file
        games_to_predict: Optional list of game IDs to predict

    Returns:
        DataFrame with game predictions
    """
    # Load model and config
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    model = model_data['model']
    required_features = model_data['features']

    # Load graph
    with open(config.file_paths['graph'], 'rb') as f:
        graph = pickle.load(f)

    # If no specific games provided, get upcoming games or most recent games
    if not games_to_predict:
        # Assuming we want the most recent 10 games in the graph
        game_nodes = [(game_id, data) for game_id, data in graph.nodes(data=True)
                      if data.get('type') == 'game']
        game_nodes.sort(key=lambda x: x[1].get('game_date', datetime.min), reverse=True)
        games_to_predict = [game_id for game_id, _ in game_nodes[:10]]

    # Extract features for these games
    prediction_features = []
    for game_id in games_to_predict:
        if game_id in graph.nodes and graph.nodes[game_id].get('type') == 'game':
            game_data = graph.nodes[game_id]

            # Basic game information
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')
            game_date = game_data.get('game_date')

            if not all([home_team, away_team, game_date]):
                print(f"Skipping game {game_id}: missing basic information")
                continue

            # Get TGP nodes
            home_tgp = f"{game_id}_{home_team}"
            away_tgp = f"{game_id}_{away_team}"

            # Check if TGP nodes exist
            if home_tgp not in graph.nodes or away_tgp not in graph.nodes:
                print(f"Skipping game {game_id}: missing TGP nodes")
                continue

            # Extract features (same logic as in extract_features_from_graph)
            # ... (implement same feature extraction logic from above)

            prediction_features.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                # Add all extracted features
            })

    # Create DataFrame
    pred_df = pd.DataFrame(prediction_features)

    # Ensure all required features are present
    for feature in required_features:
        if feature not in pred_df.columns:
            pred_df[feature] = 0

    # Make predictions
    pred_df['home_win_prob'] = model.predict_proba(pred_df[required_features])[:, 1]
    pred_df['prediction'] = model.predict(pred_df[required_features])

    return pred_df[['game_id', 'game_date', 'home_team', 'away_team',
                    'home_win_prob', 'prediction']]


# Example usage:
if __name__ == "__main__":
    config_path = "path/to/your/config.pkl"
    df, model, metrics = create_and_train_hockey_prediction_model(config_path)

    # Make predictions for upcoming games
    predictions = make_game_predictions(
        "path/to/your/saved/model.pkl",
        config_path
    )
    print(predictions)