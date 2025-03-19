import copy
import numpy as np
import os
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, f1_score, roc_auc_score, confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from src_code.utils.save_graph_utils import load_filtered_graph


def run_gnn_enhanced(config, config_model):
    """
    Enhanced version of run_gnn with improved model complexity, evaluation, and visualization.
    Now supports multiple window sizes for feature extraction.

    Args:
        config: Configuration object with file paths
        config_model: Configuration object with model parameters
    """
    # Force matplotlib to save files instead of trying to display them
    mpl.use('Agg')

    print("====== Starting Enhanced GNN Training ======")
    print(f"Loading graph from {config.file_paths['graph']}")
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None
    # Load the graph with date filtering
    data_graph = load_filtered_graph(config.file_paths["graph"], training_cutoff_date)

    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")
    stats = get_simple_node_stats(data_graph)
    feature_stats = examine_tgp_feature_stats(data_graph, window_sizes=config.stat_window_sizes)



    features, edge_list, labels_dict, node_mapping, diagnostics = extract_features_for_multi_task(
        data_graph, window_sizes=config.stat_window_sizes
    )
    # Examine detailed diagnostics
    print(f"Labeled games: {diagnostics['labeled_games']}/{diagnostics['total_games']}")
    print(f"Home win rate: {diagnostics['home_wins'] / diagnostics['labeled_games'] * 100:.1f}%")


    # Prepare data for multi-task training
    model_data = prepare_multi_task_data(features, edge_list, labels_dict)

    # Create multi-task model
    model = MultiTaskHockeyGNN(in_channels=model_data['x'].shape[1], hidden_channels=128)

    # Training and predictions will return dictionaries with all five tasks

    # Create output directory for visualizations
    output_dir = config.file_paths["gnn_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis outputs will be saved to {output_dir}")

    # Define feature names for interpretability
    # Base features that are included for each window size
    base_feature_names = [
        "Days Since Last Game (normalized)",
        "Home/Away Indicator"
    ]

    # Extract parameters from config_model with defaults
    epochs = config_model.num_epochs if hasattr(config_model, 'num_epochs') else 150
    hidden_channels = config_model.hidden_channels if hasattr(config_model, 'hidden_channels') else 128
    window_sizes = config.stat_window_sizes if hasattr(config, 'stat_window_sizes') else [5]
    lr = config_model.learning_rate if hasattr(config_model, 'learning_rate') else 0.005
    dropout_rate1 = config_model.dropout_rate1 if hasattr(config_model, 'dropout_rate1') else 0.4
    dropout_rate2 = config_model.dropout_rate2 if hasattr(config_model, 'dropout_rate2') else 0.3
    patience = config_model.patience if hasattr(config_model, 'patience') else 8
    weight_decay = config_model.weight_decay if hasattr(config_model, 'weight_decay') else 1e-4
    lr_reduce_factor = config_model.lr_reduce_factor if hasattr(config_model, 'lr_reduce_factor') else 0.3
    lr_reduce_patience = config_model.lr_reduce_patience if hasattr(config_model, 'lr_reduce_patience') else 5

    # Generate feature names for all window sizes
    feature_names = []
    for win_size in window_sizes:
        feature_names.extend([
            f"Historical Win Rate (window={win_size})",
            f"Historical Goal Rate (window={win_size})",
            f"Historical Goals Against Rate (window={win_size})"
        ])
    feature_names.extend(base_feature_names)  # Add base features

    # Train and evaluate the enhanced GNN with all parameters
    print(f"Training enhanced GNN with up to {epochs} epochs (patience={patience}) and window sizes {window_sizes}")
    model, metrics, history = train_and_evaluate_gnn_enhanced(
        data_graph,
        epochs=epochs,
        hidden_channels=hidden_channels,
        window_sizes=window_sizes,
        lr=lr,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2,
        patience=patience,
        weight_decay=weight_decay,
        lr_reduce_factor=lr_reduce_factor,
        lr_reduce_patience=lr_reduce_patience
    )

    # Analyze feature importance using SHAP
    print("\nAnalyzing feature importance...")
    try:
        # Check if SHAP is installed
        try:
            import shap
            # Get feature data for SHAP analysis
            features, edge_list, labels, _ = extract_features_from_graph(data_graph, window_sizes)
            model_data = prepare_train_test_data(features, edge_list, labels)

            # Run feature importance analysis
            feature_importance = analyze_feature_importance(model, model_data, feature_names)
        except ImportError:
            print("SHAP package not installed. Skipping feature importance analysis.")
            print("To install SHAP, run: pip install shap")
            feature_importance = None
    except Exception as e:
        print(f"Warning: Feature importance analysis failed: {str(e)}")
        feature_importance = None

    # Create comprehensive visualization report
    print("\nGenerating comprehensive visualization report...")
    create_comprehensive_visualization_report(
        model, metrics, history, feature_importance, output_dir
    )

    # Make predictions for sample games
    print("\n====== Making predictions with enhanced model ======")
    teams_to_predict = [
        ('TOR', 'MTL'),
        ('BOS', 'TBL'),
        ('EDM', 'CGY'),
        ('NYR', 'NYI'),
        ('PIT', 'WSH')
    ]

    results = []
    for home_team, away_team in teams_to_predict:
        home_win_probability = predict_game_outcome(model, data_graph, home_team, away_team, window_sizes)
        print(f'Probability of {home_team} winning at home against {away_team}: {home_win_probability:.4f}')
        results.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': home_win_probability
        })

    # Create a summary bar chart of predictions
    try:
        plt.figure(figsize=(12, 6))
        teams = [f"{r['home_team']} vs {r['away_team']}" for r in results]
        probs = [r['home_win_prob'] for r in results]

        # Create colormap based on probability
        colors = ['#7f0000' if p < 0.4 else '#ffcc00' if p < 0.6 else '#006600' for p in probs]

        plt.bar(teams, probs, color=colors)
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Matchup (Home vs Away)')
        plt.ylabel('Probability of Home Team Win')
        plt.title('Predicted Game Outcomes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/game_predictions.png")
        print(f"Game predictions chart saved to {output_dir}/game_predictions.png")
    except Exception as e:
        print(f"Warning: Could not create predictions chart: {str(e)}")

    print("\n====== GNN Analysis Complete ======")
    print(f"All analysis outputs saved to {output_dir}")
    print(f"View the confusion matrix at: {output_dir}/confusion_matrix.png")
    if feature_importance is not None:
        print(f"View the SHAP feature importance at: {output_dir}/feature_importance_summary.png")

    return model, metrics, history, feature_importance


def extract_features_for_multi_task(data_graph, window_sizes=[5]):
    """
    Extract features from the graph for multi-task GNN input.
    Creates five separate prediction tasks:
    1. Regulation win probability
    2. Overtime win probability
    3. Shootout win probability
    4. Probability of game going to overtime
    5. Probability of game going to shootout

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to multi-task labels
        node_mapping: Dictionary mapping node IDs to their indices in features_list
        diagnostics: Dictionary with diagnostic information
    """
    import numpy as np
    print(f"Extracting features for multi-task prediction with window sizes {window_sizes}...")

    # Initialize containers
    features_list = []
    node_mapping = {}

    # For multi-task learning, we need separate label dictionaries
    labels_dict = {
        'regulation_win': {},  # 1 for home win, 0 for away win in regulation
        'overtime_win': {},  # 1 for home win, 0 for away win in overtime
        'shootout_win': {},  # 1 for home win, 0 for away win in shootout
        'goes_to_overtime': {},  # 1 if game goes to overtime, 0 if ends in regulation
        'goes_to_shootout': {}  # 1 if game goes to shootout, 0 if not
    }

    # Diagnostic information
    diagnostics = {
        'feature_stats': {},
        'missing_values': {},
        'total_games': 0,
        'game_outcomes': {
            'regulation_home_win': 0,
            'regulation_away_win': 0,
            'overtime_home_win': 0,
            'overtime_away_win': 0,
            'shootout_home_win': 0,
            'shootout_away_win': 0,
            'games_to_overtime': 0,
            'games_to_shootout': 0,
            'unlabeled': 0
        },
        'window_sizes': window_sizes,
        'node_types': {
            'home_team': 0,
            'away_team': 0,
            'game': 0
        }
    }

    # Track missing values
    missing_values = {
        'hist_win_avg': 0,
        'hist_goal_avg': 0,
        'hist_goal_against_avg': 0,
        'hist_regulation_win_avg': 0,
        'hist_overtime_win_avg': 0,
        'hist_shootout_win_avg': 0,
        'days_since_last_game': 0,
        'win_label': 0
    }

    # Get all game nodes
    game_nodes = [node for node, data in data_graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"Found {len(game_nodes)} game nodes in the graph")
    diagnostics['total_games'] = len(game_nodes)

    feature_count = 0
    labeled_count = 0

    # Process each game
    for game_id in game_nodes:
        game_data = data_graph.nodes[game_id]
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        if not home_team or not away_team:
            print(f"Warning: Game {game_id} missing team information")
            continue

        # Get TGP nodes for this game
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        if home_tgp in data_graph.nodes and away_tgp in data_graph.nodes:
            home_tgp_data = data_graph.nodes[home_tgp]
            away_tgp_data = data_graph.nodes[away_tgp]

            # Extract home team features for all window sizes
            home_features = []
            for window_size in window_sizes:
                # Use pre-calculated average values where available

                # Win rates - overall
                win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_win_avg'], list):
                        win_rate = sum(home_tgp_data[f'hist_{window_size}_win_avg']) / 3
                    else:
                        win_rate = home_tgp_data[f'hist_{window_size}_win_avg']
                else:
                    missing_values['hist_win_avg'] += 1
                home_features.append(win_rate)

                # Win rates - specific for regulation (index 0)
                regulation_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            home_tgp_data[f'hist_{window_size}_win_avg']) > 0:
                        regulation_win_rate = home_tgp_data[f'hist_{window_size}_win_avg'][0]
                    elif not isinstance(home_tgp_data[f'hist_{window_size}_win_avg'], list):
                        regulation_win_rate = home_tgp_data[f'hist_{window_size}_win_avg']
                else:
                    missing_values['hist_regulation_win_avg'] += 1
                home_features.append(regulation_win_rate)

                # Win rates - specific for overtime (index 1)
                overtime_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            home_tgp_data[f'hist_{window_size}_win_avg']) > 1:
                        overtime_win_rate = home_tgp_data[f'hist_{window_size}_win_avg'][1]
                    # If it's not a list, we'll use the overall win rate
                else:
                    missing_values['hist_overtime_win_avg'] += 1
                home_features.append(overtime_win_rate)

                # Win rates - specific for shootout (index 2)
                shootout_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            home_tgp_data[f'hist_{window_size}_win_avg']) > 2:
                        shootout_win_rate = home_tgp_data[f'hist_{window_size}_win_avg'][2]
                    # If it's not a list, we'll use the overall win rate
                else:
                    missing_values['hist_shootout_win_avg'] += 1
                home_features.append(shootout_win_rate)

                # Historical goal rates - overall
                goal_rate = 0.0  # Default
                if f'hist_{window_size}_goal_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_goal_avg'], list):
                        goal_rate = sum(home_tgp_data[f'hist_{window_size}_goal_avg']) / 3
                    else:
                        goal_rate = home_tgp_data[f'hist_{window_size}_goal_avg']
                else:
                    missing_values['hist_goal_avg'] += 1
                home_features.append(goal_rate)

                # Goal rates - specific for regulation (index 0)
                regulation_goal_rate = 0.0  # Default
                if f'hist_{window_size}_goal_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_goal_avg'], list) and len(
                            home_tgp_data[f'hist_{window_size}_goal_avg']) > 0:
                        regulation_goal_rate = home_tgp_data[f'hist_{window_size}_goal_avg'][0]
                    elif not isinstance(home_tgp_data[f'hist_{window_size}_goal_avg'], list):
                        regulation_goal_rate = home_tgp_data[f'hist_{window_size}_goal_avg']
                home_features.append(regulation_goal_rate)

                # Historical goals against rates - overall
                goals_against_rate = 0.0  # Default
                if f'hist_{window_size}_goal_against_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_goal_against_avg'], list):
                        goals_against_rate = sum(home_tgp_data[f'hist_{window_size}_goal_against_avg']) / 3
                    else:
                        goals_against_rate = home_tgp_data[f'hist_{window_size}_goal_against_avg']
                else:
                    missing_values['hist_goal_against_avg'] += 1
                home_features.append(goals_against_rate)

                # Goals against rates - specific for regulation (index 0)
                regulation_goals_against_rate = 0.0  # Default
                if f'hist_{window_size}_goal_against_avg' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_goal_against_avg'], list) and len(
                            home_tgp_data[f'hist_{window_size}_goal_against_avg']) > 0:
                        regulation_goals_against_rate = home_tgp_data[f'hist_{window_size}_goal_against_avg'][0]
                    elif not isinstance(home_tgp_data[f'hist_{window_size}_goal_against_avg'], list):
                        regulation_goals_against_rate = home_tgp_data[f'hist_{window_size}_goal_against_avg']
                home_features.append(regulation_goals_against_rate)

                # Add historical frequency of games going to overtime
                overtime_freq = 0.0  # Default
                if f'hist_{window_size}_games' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_games'], list) and len(
                            home_tgp_data[f'hist_{window_size}_games']) > 1:
                        total_games = sum(home_tgp_data[f'hist_{window_size}_games'])
                        overtime_games = home_tgp_data[f'hist_{window_size}_games'][1] + \
                                         home_tgp_data[f'hist_{window_size}_games'][2]
                        overtime_freq = overtime_games / total_games if total_games > 0 else 0.0
                home_features.append(overtime_freq)

                # Add historical frequency of games going to shootout
                shootout_freq = 0.0  # Default
                if f'hist_{window_size}_games' in home_tgp_data:
                    if isinstance(home_tgp_data[f'hist_{window_size}_games'], list) and len(
                            home_tgp_data[f'hist_{window_size}_games']) > 2:
                        total_games = sum(home_tgp_data[f'hist_{window_size}_games'])
                        shootout_games = home_tgp_data[f'hist_{window_size}_games'][2]
                        shootout_freq = shootout_games / total_games if total_games > 0 else 0.0
                home_features.append(shootout_freq)

            # Days since last game
            if 'days_since_last_game' in home_tgp_data:
                days_value = min(home_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                home_features.append(days_value)
            else:
                home_features.append(1.0)  # Default (max) days since last game
                missing_values['days_since_last_game'] += 1

            # Home advantage indicator
            home_features.append(1.0)  # Home team

            # Extract away team features using the same approach
            away_features = []
            for window_size in window_sizes:
                # Use pre-calculated average values where available

                # Win rates - overall
                win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_win_avg'], list):
                        win_rate = sum(away_tgp_data[f'hist_{window_size}_win_avg']) / 3
                    else:
                        win_rate = away_tgp_data[f'hist_{window_size}_win_avg']
                else:
                    missing_values['hist_win_avg'] += 1
                away_features.append(win_rate)

                # Win rates - specific for regulation (index 0)
                regulation_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            away_tgp_data[f'hist_{window_size}_win_avg']) > 0:
                        regulation_win_rate = away_tgp_data[f'hist_{window_size}_win_avg'][0]
                    elif not isinstance(away_tgp_data[f'hist_{window_size}_win_avg'], list):
                        regulation_win_rate = away_tgp_data[f'hist_{window_size}_win_avg']
                else:
                    missing_values['hist_regulation_win_avg'] += 1
                away_features.append(regulation_win_rate)

                # Win rates - specific for overtime (index 1)
                overtime_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            away_tgp_data[f'hist_{window_size}_win_avg']) > 1:
                        overtime_win_rate = away_tgp_data[f'hist_{window_size}_win_avg'][1]
                    # If it's not a list, we'll use the overall win rate
                else:
                    missing_values['hist_overtime_win_avg'] += 1
                away_features.append(overtime_win_rate)

                # Win rates - specific for shootout (index 2)
                shootout_win_rate = 0.5  # Default
                if f'hist_{window_size}_win_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_win_avg'], list) and len(
                            away_tgp_data[f'hist_{window_size}_win_avg']) > 2:
                        shootout_win_rate = away_tgp_data[f'hist_{window_size}_win_avg'][2]
                    # If it's not a list, we'll use the overall win rate
                else:
                    missing_values['hist_shootout_win_avg'] += 1
                away_features.append(shootout_win_rate)

                # Historical goal rates - overall
                goal_rate = 0.0  # Default
                if f'hist_{window_size}_goal_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_goal_avg'], list):
                        goal_rate = sum(away_tgp_data[f'hist_{window_size}_goal_avg']) / 3
                    else:
                        goal_rate = away_tgp_data[f'hist_{window_size}_goal_avg']
                else:
                    missing_values['hist_goal_avg'] += 1
                away_features.append(goal_rate)

                # Goal rates - specific for regulation (index 0)
                regulation_goal_rate = 0.0  # Default
                if f'hist_{window_size}_goal_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_goal_avg'], list) and len(
                            away_tgp_data[f'hist_{window_size}_goal_avg']) > 0:
                        regulation_goal_rate = away_tgp_data[f'hist_{window_size}_goal_avg'][0]
                    elif not isinstance(away_tgp_data[f'hist_{window_size}_goal_avg'], list):
                        regulation_goal_rate = away_tgp_data[f'hist_{window_size}_goal_avg']
                away_features.append(regulation_goal_rate)

                # Historical goals against rates - overall
                goals_against_rate = 0.0  # Default
                if f'hist_{window_size}_goal_against_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_goal_against_avg'], list):
                        goals_against_rate = sum(away_tgp_data[f'hist_{window_size}_goal_against_avg']) / 3
                    else:
                        goals_against_rate = away_tgp_data[f'hist_{window_size}_goal_against_avg']
                else:
                    missing_values['hist_goal_against_avg'] += 1
                away_features.append(goals_against_rate)

                # Goals against rates - specific for regulation (index 0)
                regulation_goals_against_rate = 0.0  # Default
                if f'hist_{window_size}_goal_against_avg' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_goal_against_avg'], list) and len(
                            away_tgp_data[f'hist_{window_size}_goal_against_avg']) > 0:
                        regulation_goals_against_rate = away_tgp_data[f'hist_{window_size}_goal_against_avg'][0]
                    elif not isinstance(away_tgp_data[f'hist_{window_size}_goal_against_avg'], list):
                        regulation_goals_against_rate = away_tgp_data[f'hist_{window_size}_goal_against_avg']
                away_features.append(regulation_goals_against_rate)

                # Add historical frequency of games going to overtime
                overtime_freq = 0.0  # Default
                if f'hist_{window_size}_games' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_games'], list) and len(
                            away_tgp_data[f'hist_{window_size}_games']) > 1:
                        total_games = sum(away_tgp_data[f'hist_{window_size}_games'])
                        overtime_games = away_tgp_data[f'hist_{window_size}_games'][1] + \
                                         away_tgp_data[f'hist_{window_size}_games'][2]
                        overtime_freq = overtime_games / total_games if total_games > 0 else 0.0
                away_features.append(overtime_freq)

                # Add historical frequency of games going to shootout
                shootout_freq = 0.0  # Default
                if f'hist_{window_size}_games' in away_tgp_data:
                    if isinstance(away_tgp_data[f'hist_{window_size}_games'], list) and len(
                            away_tgp_data[f'hist_{window_size}_games']) > 2:
                        total_games = sum(away_tgp_data[f'hist_{window_size}_games'])
                        shootout_games = away_tgp_data[f'hist_{window_size}_games'][2]
                        shootout_freq = shootout_games / total_games if total_games > 0 else 0.0
                away_features.append(shootout_freq)

            # Days since last game
            if 'days_since_last_game' in away_tgp_data:
                days_value = min(away_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                away_features.append(days_value)
            else:
                away_features.append(1.0)  # Default (max) days since last game
                missing_values['days_since_last_game'] += 1

            # Away team indicator
            away_features.append(0.0)  # Away team

            # Game node features (placeholder)
            # Create a feature vector with the same dimension as team features
            game_features = np.zeros(len(home_features), dtype=np.float32)
            game_features[-1] = 0.5  # Mark as game node with a middle value

            # Add features to list and update mapping
            home_idx = len(features_list)
            node_mapping[home_tgp] = home_idx
            features_list.append(np.array(home_features, dtype=np.float32))
            diagnostics['node_types']['home_team'] += 1

            away_idx = len(features_list)
            node_mapping[away_tgp] = away_idx
            features_list.append(np.array(away_features, dtype=np.float32))
            diagnostics['node_types']['away_team'] += 1

            game_idx = len(features_list)
            node_mapping[game_id] = game_idx
            features_list.append(game_features)
            diagnostics['node_types']['game'] += 1

            feature_count += 3  # Added 3 nodes with features

            # Create labels for each prediction task
            labeled = False

            # Extract game outcome information
            if 'games' in home_tgp_data and 'win' in home_tgp_data and 'games' in away_tgp_data and 'win' in away_tgp_data:
                # Check what period the game ended in (regulation, OT, shootout)
                home_games = [0, 0, 0]
                if isinstance(home_tgp_data['games'], list):
                    home_games = home_tgp_data['games']

                home_wins = [0, 0, 0]
                if isinstance(home_tgp_data['win'], list):
                    home_wins = home_tgp_data['win']

                away_wins = [0, 0, 0]
                if isinstance(away_tgp_data['win'], list):
                    away_wins = away_tgp_data['win']

                # Determine where the game ended based on games array
                if home_games[0] == 1 and home_games[1] == 0 and home_games[2] == 0:
                    # Game ended in regulation
                    if home_wins[0] == 1:
                        # Home team won in regulation
                        labels_dict['regulation_win'][game_idx] = 1
                        diagnostics['game_outcomes']['regulation_home_win'] += 1
                        labeled = True
                    elif away_wins[0] == 1:
                        # Away team won in regulation
                        labels_dict['regulation_win'][game_idx] = 0
                        diagnostics['game_outcomes']['regulation_away_win'] += 1
                        labeled = True

                    # Game did not go to overtime
                    labels_dict['goes_to_overtime'][game_idx] = 0

                elif home_games[1] == 1 and home_games[2] == 0:
                    # Game ended in overtime
                    if home_wins[1] == 1:
                        # Home team won in overtime
                        labels_dict['overtime_win'][game_idx] = 1
                        diagnostics['game_outcomes']['overtime_home_win'] += 1
                    elif away_wins[1] == 1:
                        # Away team won in overtime
                        labels_dict['overtime_win'][game_idx] = 0
                        diagnostics['game_outcomes']['overtime_away_win'] += 1

                    # Game went to overtime but not shootout
                    labels_dict['goes_to_overtime'][game_idx] = 1
                    labels_dict['goes_to_shootout'][game_idx] = 0
                    diagnostics['game_outcomes']['games_to_overtime'] += 1
                    labeled = True

                elif home_games[2] == 1:
                    # Game ended in shootout
                    if home_wins[2] == 1:
                        # Home team won in shootout
                        labels_dict['shootout_win'][game_idx] = 1
                        diagnostics['game_outcomes']['shootout_home_win'] += 1
                    elif away_wins[2] == 1:
                        # Away team won in shootout
                        labels_dict['shootout_win'][game_idx] = 0
                        diagnostics['game_outcomes']['shootout_away_win'] += 1

                    # Game went to overtime and shootout
                    labels_dict['goes_to_overtime'][game_idx] = 1
                    labels_dict['goes_to_shootout'][game_idx] = 1
                    diagnostics['game_outcomes']['games_to_overtime'] += 1
                    diagnostics['game_outcomes']['games_to_shootout'] += 1
                    labeled = True

            if labeled:
                labeled_count += 1
            else:
                diagnostics['game_outcomes']['unlabeled'] += 1
                missing_values['win_label'] += 1

    # Update diagnostic information
    diagnostics['labeled_games'] = labeled_count
    diagnostics['missing_values'] = missing_values
    diagnostics['node_count'] = len(features_list)

    # Calculate feature dimension
    if features_list:
        feature_dim = len(features_list[0])
        diagnostics['feature_dim'] = feature_dim
        print(f"Feature dimension: {feature_dim} (includes {len(window_sizes)} window sizes)")
    else:
        feature_dim = 0
        diagnostics['feature_dim'] = 0
        print("Warning: No features extracted")

    print(f"Extracted features for {feature_count} nodes ({labeled_count} labeled games)")

    # Create edge list - connect games to their TGPs
    edge_list = []
    for game_id in game_nodes:
        if game_id in data_graph.nodes and game_id in node_mapping:
            game_idx = node_mapping[game_id]

            # Connect game to home and away TGP nodes
            home_team = data_graph.nodes[game_id]['home_team']
            away_team = data_graph.nodes[game_id]['away_team']
            home_tgp = f"{game_id}_{home_team}"
            away_tgp = f"{game_id}_{away_team}"

            if home_tgp in node_mapping:
                home_tgp_idx = node_mapping[home_tgp]
                edge_list.append((game_idx, home_tgp_idx))
                edge_list.append((home_tgp_idx, game_idx))  # Bidirectional

            if away_tgp in node_mapping:
                away_tgp_idx = node_mapping[away_tgp]
                edge_list.append((game_idx, away_tgp_idx))
                edge_list.append((away_tgp_idx, game_idx))  # Bidirectional

    diagnostics['edge_count'] = len(edge_list)
    print(f"Created {len(edge_list)} edges")

    # Print summary statistics
    print("\n=== Multi-Task Feature Extraction Summary ===")
    print(f"Total games: {diagnostics['total_games']}")
    print(
        f"Labeled games: {diagnostics['labeled_games']} ({diagnostics['labeled_games'] / diagnostics['total_games'] * 100:.1f}%)")

    # Print game outcome distribution
    print("\nGame Outcome Distribution:")
    reg_games = diagnostics['game_outcomes']['regulation_home_win'] + diagnostics['game_outcomes'][
        'regulation_away_win']
    ot_games = diagnostics['game_outcomes']['overtime_home_win'] + diagnostics['game_outcomes']['overtime_away_win']
    so_games = diagnostics['game_outcomes']['shootout_home_win'] + diagnostics['game_outcomes']['shootout_away_win']

    print(f"Regulation: {reg_games} games ({reg_games / labeled_count * 100:.1f}%)")
    print(
        f"  Home wins: {diagnostics['game_outcomes']['regulation_home_win']} ({diagnostics['game_outcomes']['regulation_home_win'] / reg_games * 100:.1f}% of regulation games)")
    print(
        f"  Away wins: {diagnostics['game_outcomes']['regulation_away_win']} ({diagnostics['game_outcomes']['regulation_away_win'] / reg_games * 100:.1f}% of regulation games)")

    if ot_games > 0:
        print(f"Overtime: {ot_games} games ({ot_games / labeled_count * 100:.1f}%)")
        print(
            f"  Home wins: {diagnostics['game_outcomes']['overtime_home_win']} ({diagnostics['game_outcomes']['overtime_home_win'] / ot_games * 100:.1f}% of OT games)")
        print(
            f"  Away wins: {diagnostics['game_outcomes']['overtime_away_win']} ({diagnostics['game_outcomes']['overtime_away_win'] / ot_games * 100:.1f}% of OT games)")

    if so_games > 0:
        print(f"Shootout: {so_games} games ({so_games / labeled_count * 100:.1f}%)")
        print(
            f"  Home wins: {diagnostics['game_outcomes']['shootout_home_win']} ({diagnostics['game_outcomes']['shootout_home_win'] / so_games * 100:.1f}% of SO games)")
        print(
            f"  Away wins: {diagnostics['game_outcomes']['shootout_away_win']} ({diagnostics['game_outcomes']['shootout_away_win'] / so_games * 100:.1f}% of SO games)")

    print(
        f"\nGames to overtime: {diagnostics['game_outcomes']['games_to_overtime']} ({diagnostics['game_outcomes']['games_to_overtime'] / labeled_count * 100:.1f}%)")
    print(
        f"Games to shootout: {diagnostics['game_outcomes']['games_to_shootout']} ({diagnostics['game_outcomes']['games_to_shootout'] / labeled_count * 100:.1f}%)")

    print(f"\nFeature dimension: {diagnostics['feature_dim']}")

    if missing_values['win_label'] > 0:
        print(
            f"Warning: {missing_values['win_label']} games ({missing_values['win_label'] / diagnostics['total_games'] * 100:.1f}%) have no outcome labels")

    # Check for any missing values
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        print(f"\nWarning: Found {total_missing} missing values that were filled with defaults:")
        for key, count in missing_values.items():
            if count > 0:
                print(f"  {key}: {count} instances")

    return features_list, edge_list, labels_dict, node_mapping, diagnostics


def normalize_features(features_list):
    """
    Normalize feature vectors to improve GNN training.

    Args:
        features_list: List of feature vectors

    Returns:
        normalized_features: List of normalized feature vectors
    """
    import numpy as np

    # Convert to numpy array for easier processing
    features_array = np.array(features_list)

    # Check for NaN or inf values
    bad_values = np.isnan(features_array) | np.isinf(features_array)
    if np.any(bad_values):
        num_bad = np.sum(bad_values)
        print(f"Warning: Found {num_bad} NaN/inf values. Replacing with zeros.")
        features_array[bad_values] = 0.0

    # Get feature dimension
    num_samples, num_features = features_array.shape

    # Skip the last feature (home/away indicator) which is already normalized
    normalized_array = features_array.copy()

    # Normalize each feature column except the last one
    for i in range(num_features - 1):
        col = features_array[:, i]
        col_min = np.min(col)
        col_max = np.max(col)

        # Only normalize if there's a range of values
        if col_max > col_min:
            normalized_array[:, i] = (col - col_min) / (col_max - col_min)

    # Convert back to list of numpy arrays
    normalized_features = [normalized_array[i] for i in range(num_samples)]

    print(f"Features normalized. Shape: {normalized_array.shape}")
    return normalized_features


class MultiTaskHockeyGNN(nn.Module):
    """
    Multi-task GNN model for hockey game prediction.
    Predicts 5 separate outcomes:
    1. Regulation win probability
    2. Overtime win probability
    3. Shootout win probability
    4. Probability of game going to overtime
    5. Probability of game going to shootout
    """

    def __init__(self, in_channels, hidden_channels=128, dropout_rate1=0.4, dropout_rate2=0.3):
        super(MultiTaskHockeyGNN, self).__init__()
        from torch_geometric.nn import GCNConv
        import torch.nn as nn
        import torch.nn.functional as F

        # Shared layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Task-specific output layers
        self.regulation_win_fc = nn.Linear(hidden_channels, 2)  # Binary classification
        self.overtime_win_fc = nn.Linear(hidden_channels, 2)  # Binary classification
        self.shootout_win_fc = nn.Linear(hidden_channels, 2)  # Binary classification
        self.goes_to_overtime_fc = nn.Linear(hidden_channels, 2)  # Binary classification
        self.goes_to_shootout_fc = nn.Linear(hidden_channels, 2)  # Binary classification

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN with shared layers and task-specific outputs.

        Args:
            x: Node features
            edge_index: Edge indices
            game_indices: Indices of game nodes

        Returns:
            Dictionary of task-specific predictions
        """
        import torch.nn.functional as F

        # Shared layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate1, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate2, training=self.training)

        # Select only game nodes for prediction
        x_games = x[game_indices]

        # Task-specific predictions
        regulation_win_logits = self.regulation_win_fc(x_games)
        overtime_win_logits = self.overtime_win_fc(x_games)
        shootout_win_logits = self.shootout_win_fc(x_games)
        goes_to_overtime_logits = self.goes_to_overtime_fc(x_games)
        goes_to_shootout_logits = self.goes_to_shootout_fc(x_games)

        # Apply softmax to get probabilities
        regulation_win_probs = F.log_softmax(regulation_win_logits, dim=1)
        overtime_win_probs = F.log_softmax(overtime_win_logits, dim=1)
        shootout_win_probs = F.log_softmax(shootout_win_logits, dim=1)
        goes_to_overtime_probs = F.log_softmax(goes_to_overtime_logits, dim=1)
        goes_to_shootout_probs = F.log_softmax(goes_to_shootout_logits, dim=1)

        return {
            'regulation_win': regulation_win_probs,
            'overtime_win': overtime_win_probs,
            'shootout_win': shootout_win_probs,
            'goes_to_overtime': goes_to_overtime_probs,
            'goes_to_shootout': goes_to_shootout_probs
        }


def prepare_multi_task_data(features, edge_list, labels_dict, test_size=0.2):
    """
    Prepare data for multi-task GNN training and testing.

    Args:
        features: List of feature vectors
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to multi-task labels
        test_size: Proportion of data to use for testing

    Returns:
        model_data: Dictionary containing all data needed for training and evaluation
    """
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split

    print(f"Preparing multi-task train/test split with test_size={test_size}...")

    # Convert features to tensor
    x = torch.tensor(np.array(features), dtype=torch.float)

    # Convert edge list to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Create a 2x0 empty tensor as a valid but empty edge_index
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Get all game indices that have at least one label
    all_game_indices = set()
    for task_labels in labels_dict.values():
        all_game_indices.update(task_labels.keys())

    game_indices = sorted(list(all_game_indices))
    game_mask = torch.zeros(len(features), dtype=torch.bool)
    game_mask[game_indices] = True

    # Create separate label tensors for each task
    task_labels = {}
    for task, task_dict in labels_dict.items():
        # Create a tensor with -1 for games without a label for this task
        y = torch.full((len(game_indices),), -1, dtype=torch.long)

        # Fill in the labels for games that have them
        for i, idx in enumerate(game_indices):
            if idx in task_dict:
                y[i] = task_dict[idx]

        task_labels[task] = y

    # Get games that have labels for the regulation win task (most common)
    # for train/test splitting
    labeled_indices = []
    for i, idx in enumerate(game_indices):
        if idx in labels_dict['regulation_win']:
            labeled_indices.append(i)

    # Create stratified splits based on regulation win outcomes
    if labeled_indices:
        target_for_split = [labels_dict['regulation_win'][game_indices[i]] for i in labeled_indices]
        train_indices, test_indices = train_test_split(
            labeled_indices, test_size=test_size, random_state=42,
            stratify=target_for_split
        )
    else:
        # Fallback if no regulation win labels
        total = len(game_indices)
        train_size = int((1 - test_size) * total)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, total))

    # Map local indices back to global indices
    train_games = [game_indices[i] for i in train_indices]
    test_games = [game_indices[i] for i in test_indices]

    # Create train and test masks
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[train_games] = True

    test_mask = torch.zeros(len(features), dtype=torch.bool)
    test_mask[test_games] = True

    # Print task-specific statistics
    for task, y in task_labels.items():
        labeled_count = (y >= 0).sum().item()
        if labeled_count > 0:
            class_counts = torch.bincount(y[y >= 0], minlength=2)
            print(f"Task '{task}': {labeled_count} labeled games")
            if task.endswith('_win'):
                print(f"  Class 0 (Away win): {class_counts[0]} ({class_counts[0] / labeled_count * 100:.1f}%)")
                print(f"  Class 1 (Home win): {class_counts[1]} ({class_counts[1] / labeled_count * 100:.1f}%)")
            else:
                print(f"  Class 0 (No): {class_counts[0]} ({class_counts[0] / labeled_count * 100:.1f}%)")
                print(f"  Class 1 (Yes): {class_counts[1]} ({class_counts[1] / labeled_count * 100:.1f}%)")

    print(f"Data prepared with {len(game_indices)} games. "
          f"Training on {len(train_indices)} games, "
          f"testing on {len(test_indices)} games.")

    # Create dictionary with all training data
    model_data = {
        'x': x,
        'edge_index': edge_index,
        'task_labels': task_labels,
        'game_mask': game_mask,
        'game_indices': torch.tensor(game_indices, dtype=torch.long),
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_games': train_games,
        'test_games': test_games,
        'train_indices': train_indices,
        'test_indices': test_indices
    }

    return model_data


def extract_features_from_graph(data_graph, window_sizes=[5]):
    """
    Extract features from the graph for multi-task GNN input.
    This is a wrapper around the multi-task version that maintains the original interface.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to overall outcome labels (1 for home win, 0 for away win)
        node_mapping: Dictionary mapping node IDs to their indices in features_list
    """
    import numpy as np

    # Call the multi-task version
    features_list, edge_list, multi_task_labels, node_mapping, diagnostics = extract_features_for_multi_task(
        data_graph, window_sizes
    )

    # Normalize features
    features_list = normalize_features(features_list)

    # To maintain backward compatibility, convert multi-task labels to overall win/loss
    labels_dict = {}

    # Combine labels from regulation, overtime, and shootout wins
    for game_idx in set().union(
            multi_task_labels['regulation_win'].keys(),
            multi_task_labels['overtime_win'].keys(),
            multi_task_labels['shootout_win'].keys()
    ):
        # Check regulation wins
        if game_idx in multi_task_labels['regulation_win']:
            labels_dict[game_idx] = multi_task_labels['regulation_win'][game_idx]
        # Check overtime wins
        elif game_idx in multi_task_labels['overtime_win']:
            labels_dict[game_idx] = multi_task_labels['overtime_win'][game_idx]
        # Check shootout wins
        elif game_idx in multi_task_labels['shootout_win']:
            labels_dict[game_idx] = multi_task_labels['shootout_win'][game_idx]

    # Print overall win/loss balance
    home_wins = sum(1 for label in labels_dict.values() if label == 1)
    away_wins = sum(1 for label in labels_dict.values() if label == 0)
    labeled_count = home_wins + away_wins

    if labeled_count > 0:
        print("\n=== Overall Win/Loss Labels ===")
        print(f"Labeled games: {labeled_count}")
        print(f"Home wins: {home_wins} ({home_wins / labeled_count * 100:.1f}%)")
        print(f"Away wins: {away_wins} ({away_wins / labeled_count * 100:.1f}%)")

    return features_list, edge_list, labels_dict, node_mapping


def extract_features_from_graph_with_diagnostics(data_graph, window_sizes=[5]):
    """
    Extract features from the graph for GNN input with improved diagnostics.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to outcome labels (1 for home win, 0 for away win)
        node_mapping: Dictionary mapping node IDs to their indices in features_list
        diagnostics: Dictionary with diagnostic information
    """
    print(f"Extracting features with window sizes {window_sizes}...")

    # Initialize containers
    features_list = []
    node_mapping = {}
    labels_dict = {}

    # Diagnostic information
    diagnostics = {
        'feature_stats': {},
        'missing_values': {},
        'total_games': 0,
        'labeled_games': 0,
        'unlabeled_games': 0,
        'home_wins': 0,
        'away_wins': 0,
        'window_sizes': window_sizes,
        'node_types': {
            'home_team': 0,
            'away_team': 0,
            'game': 0
        }
    }

    # List to store all feature values for later statistics
    feature_values = {}
    for window in window_sizes:
        feature_values[f'win_rate_{window}'] = []
        feature_values[f'goal_rate_{window}'] = []
        feature_values[f'goals_against_rate_{window}'] = []
    feature_values['days_since_last_game'] = []
    feature_values['home_indicator'] = []

    # Track missing values
    missing_values = {
        'hist_win': 0,
        'hist_goal': 0,
        'hist_goal_against': 0,
        'hist_games': 0,
        'days_since_last_game': 0,
        'win_label': 0
    }

    # Get all game nodes
    game_nodes = [node for node, data in data_graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"Found {len(game_nodes)} game nodes in the graph")
    diagnostics['total_games'] = len(game_nodes)

    feature_count = 0
    labeled_count = 0

    # Process each game
    for game_id in game_nodes:
        game_data = data_graph.nodes[game_id]
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        if not home_team or not away_team:
            print(f"Warning: Game {game_id} missing team information")
            continue

        # Get TGP nodes for this game
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        if home_tgp in data_graph.nodes and away_tgp in data_graph.nodes:
            home_tgp_data = data_graph.nodes[home_tgp]
            away_tgp_data = data_graph.nodes[away_tgp]

            # Extract home team features for all window sizes
            home_features = []
            for window_size in window_sizes:
                # Historical win rate
                if (f'hist_{window_size}_win' in home_tgp_data and
                        f'hist_{window_size}_games' in home_tgp_data):

                    # Handle list or scalar values
                    if isinstance(home_tgp_data[f'hist_{window_size}_win'], list):
                        wins = sum(home_tgp_data[f'hist_{window_size}_win'])
                    else:
                        wins = home_tgp_data[f'hist_{window_size}_win']

                    if isinstance(home_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = home_tgp_data[f'hist_{window_size}_games']

                    win_rate = wins / games if games > 0 else 0.5
                    home_features.append(win_rate)
                    feature_values[f'win_rate_{window_size}'].append(win_rate)
                else:
                    home_features.append(0.5)  # Default win rate
                    missing_values['hist_win'] += 1

                # Historical goal rate
                if (f'hist_{window_size}_goal' in home_tgp_data and
                        f'hist_{window_size}_games' in home_tgp_data):

                    if isinstance(home_tgp_data[f'hist_{window_size}_goal'], list):
                        goals = sum(home_tgp_data[f'hist_{window_size}_goal'])
                    else:
                        goals = home_tgp_data[f'hist_{window_size}_goal']

                    if isinstance(home_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = home_tgp_data[f'hist_{window_size}_games']

                    goal_rate = goals / games if games > 0 else 0
                    home_features.append(goal_rate)
                    feature_values[f'goal_rate_{window_size}'].append(goal_rate)
                else:
                    home_features.append(0)  # Default goal rate
                    missing_values['hist_goal'] += 1

                # Recent goals against
                if (f'hist_{window_size}_goal_against' in home_tgp_data and
                        f'hist_{window_size}_games' in home_tgp_data):

                    if isinstance(home_tgp_data[f'hist_{window_size}_goal_against'], list):
                        goals_against = sum(home_tgp_data[f'hist_{window_size}_goal_against'])
                    else:
                        goals_against = home_tgp_data[f'hist_{window_size}_goal_against']

                    if isinstance(home_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = home_tgp_data[f'hist_{window_size}_games']

                    goals_against_rate = goals_against / games if games > 0 else 0
                    home_features.append(goals_against_rate)
                    feature_values[f'goals_against_rate_{window_size}'].append(goals_against_rate)
                else:
                    home_features.append(0)  # Default goals against rate
                    missing_values['hist_goal_against'] += 1

            # Days since last game
            if 'days_since_last_game' in home_tgp_data:
                days_value = min(home_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                home_features.append(days_value)
                feature_values['days_since_last_game'].append(days_value)
            else:
                home_features.append(1.0)  # Default (max) days since last game
                missing_values['days_since_last_game'] += 1

            # Home advantage indicator
            home_features.append(1.0)  # Home team
            feature_values['home_indicator'].append(1.0)

            # Extract away team features for all window sizes with the same approach
            away_features = []
            for window_size in window_sizes:
                # Historical win rate
                if (f'hist_{window_size}_win' in away_tgp_data and
                        f'hist_{window_size}_games' in away_tgp_data):

                    if isinstance(away_tgp_data[f'hist_{window_size}_win'], list):
                        wins = sum(away_tgp_data[f'hist_{window_size}_win'])
                    else:
                        wins = away_tgp_data[f'hist_{window_size}_win']

                    if isinstance(away_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = away_tgp_data[f'hist_{window_size}_games']

                    win_rate = wins / games if games > 0 else 0.5
                    away_features.append(win_rate)
                    feature_values[f'win_rate_{window_size}'].append(win_rate)
                else:
                    away_features.append(0.5)  # Default win rate
                    missing_values['hist_win'] += 1

                # Historical goal rate
                if (f'hist_{window_size}_goal' in away_tgp_data and
                        f'hist_{window_size}_games' in away_tgp_data):

                    if isinstance(away_tgp_data[f'hist_{window_size}_goal'], list):
                        goals = sum(away_tgp_data[f'hist_{window_size}_goal'])
                    else:
                        goals = away_tgp_data[f'hist_{window_size}_goal']

                    if isinstance(away_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = away_tgp_data[f'hist_{window_size}_games']

                    goal_rate = goals / games if games > 0 else 0
                    away_features.append(goal_rate)
                    feature_values[f'goal_rate_{window_size}'].append(goal_rate)
                else:
                    away_features.append(0)  # Default goal rate
                    missing_values['hist_goal'] += 1

                # Recent goals against
                if (f'hist_{window_size}_goal_against' in away_tgp_data and
                        f'hist_{window_size}_games' in away_tgp_data):

                    if isinstance(away_tgp_data[f'hist_{window_size}_goal_against'], list):
                        goals_against = sum(away_tgp_data[f'hist_{window_size}_goal_against'])
                    else:
                        goals_against = away_tgp_data[f'hist_{window_size}_goal_against']

                    if isinstance(away_tgp_data[f'hist_{window_size}_games'], list):
                        games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    else:
                        games = away_tgp_data[f'hist_{window_size}_games']

                    goals_against_rate = goals_against / games if games > 0 else 0
                    away_features.append(goals_against_rate)
                    feature_values[f'goals_against_rate_{window_size}'].append(goals_against_rate)
                else:
                    away_features.append(0)  # Default goals against rate
                    missing_values['hist_goal_against'] += 1

            # Days since last game
            if 'days_since_last_game' in away_tgp_data:
                days_value = min(away_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                away_features.append(days_value)
                feature_values['days_since_last_game'].append(days_value)
            else:
                away_features.append(1.0)  # Default (max) days since last game
                missing_values['days_since_last_game'] += 1

            # Away team indicator
            away_features.append(0.0)  # Away team
            feature_values['home_indicator'].append(0.0)

            # Game node features (placeholder)
            # Create a feature vector with the same dimension as team features
            game_features = np.zeros(len(home_features), dtype=np.float32)
            game_features[-1] = 0.5  # Mark as game node with a middle value

            # Add features to list and update mapping
            home_idx = len(features_list)
            node_mapping[home_tgp] = home_idx
            features_list.append(np.array(home_features, dtype=np.float32))
            diagnostics['node_types']['home_team'] += 1

            away_idx = len(features_list)
            node_mapping[away_tgp] = away_idx
            features_list.append(np.array(away_features, dtype=np.float32))
            diagnostics['node_types']['away_team'] += 1

            game_idx = len(features_list)
            node_mapping[game_id] = game_idx
            features_list.append(game_features)
            diagnostics['node_types']['game'] += 1

            feature_count += 3  # Added 3 nodes with features

            # Create label based on win/loss
            labeled = False

            # First check win attribute
            if 'win' in home_tgp_data and 'win' in away_tgp_data:
                home_wins = 0
                away_wins = 0

                # Handle win data which could be a list or scalar
                if isinstance(home_tgp_data['win'], list):
                    home_wins = sum(home_tgp_data['win'])
                else:
                    home_wins = home_tgp_data['win']

                if isinstance(away_tgp_data['win'], list):
                    away_wins = sum(away_tgp_data['win'])
                else:
                    away_wins = away_tgp_data['win']

                if home_wins > 0:
                    labels_dict[game_idx] = 1  # Home win
                    labeled_count += 1
                    diagnostics['home_wins'] += 1
                    labeled = True
                elif away_wins > 0:
                    labels_dict[game_idx] = 0  # Away win
                    labeled_count += 1
                    diagnostics['away_wins'] += 1
                    labeled = True

            # If no win data, try to infer from goals
            if not labeled and 'goal' in home_tgp_data and 'goal' in away_tgp_data:
                home_goals = 0
                away_goals = 0

                # Handle goal data which could be a list or scalar
                if isinstance(home_tgp_data['goal'], list):
                    home_goals = sum(home_tgp_data['goal'])
                else:
                    home_goals = home_tgp_data['goal']

                if isinstance(away_tgp_data['goal'], list):
                    away_goals = sum(away_tgp_data['goal'])
                else:
                    away_goals = away_tgp_data['goal']

                if home_goals > away_goals:
                    labels_dict[game_idx] = 1  # Home win
                    labeled_count += 1
                    diagnostics['home_wins'] += 1
                    labeled = True
                elif away_goals > home_goals:
                    labels_dict[game_idx] = 0  # Away win
                    labeled_count += 1
                    diagnostics['away_wins'] += 1
                    labeled = True

            if not labeled:
                diagnostics['unlabeled_games'] += 1
                missing_values['win_label'] += 1

    # Update diagnostic information
    diagnostics['labeled_games'] = labeled_count
    diagnostics['missing_values'] = missing_values
    diagnostics['node_count'] = len(features_list)

    # Calculate feature statistics
    for feature, values in feature_values.items():
        if values:
            diagnostics['feature_stats'][feature] = {
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'zeros': sum(1 for v in values if v == 0),
                'zeros_pct': (sum(1 for v in values if v == 0) / len(values)) * 100
            }

    # Calculate feature dimension for debugging
    if features_list:
        feature_dim = len(features_list[0])
        diagnostics['feature_dim'] = feature_dim
        print(f"Feature dimension: {feature_dim} (includes {len(window_sizes)} window sizes)")
    else:
        feature_dim = 0
        diagnostics['feature_dim'] = 0
        print("Warning: No features extracted")

    print(f"Extracted features for {feature_count} nodes ({labeled_count} labeled games)")

    # Create edge list - connect games to their TGPs
    edge_list = []
    for game_id in game_nodes:
        if game_id in data_graph.nodes and game_id in node_mapping:
            game_idx = node_mapping[game_id]

            # Connect game to home and away TGP nodes
            home_team = data_graph.nodes[game_id]['home_team']
            away_team = data_graph.nodes[game_id]['away_team']
            home_tgp = f"{game_id}_{home_team}"
            away_tgp = f"{game_id}_{away_team}"

            if home_tgp in node_mapping:
                home_tgp_idx = node_mapping[home_tgp]
                edge_list.append((game_idx, home_tgp_idx))
                edge_list.append((home_tgp_idx, game_idx))  # Bidirectional

            if away_tgp in node_mapping:
                away_tgp_idx = node_mapping[away_tgp]
                edge_list.append((game_idx, away_tgp_idx))
                edge_list.append((away_tgp_idx, game_idx))  # Bidirectional

    diagnostics['edge_count'] = len(edge_list)
    print(f"Created {len(edge_list)} edges")

    # Print summary statistics
    print("\n=== Feature Extraction Summary ===")
    print(f"Total games: {diagnostics['total_games']}")
    print(
        f"Labeled games: {diagnostics['labeled_games']} ({diagnostics['labeled_games'] / diagnostics['total_games'] * 100:.1f}%)")
    print(
        f"Home wins: {diagnostics['home_wins']} ({diagnostics['home_wins'] / diagnostics['labeled_games'] * 100:.1f}%)")
    print(
        f"Away wins: {diagnostics['away_wins']} ({diagnostics['away_wins'] / diagnostics['labeled_games'] * 100:.1f}%)")
    print(f"Feature dimension: {diagnostics['feature_dim']}")

    if missing_values['win_label'] > 0:
        print(
            f"Warning: {missing_values['win_label']} games ({missing_values['win_label'] / diagnostics['total_games'] * 100:.1f}%) have no win/loss label")

    # Check for any missing values
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        print(f"\nWarning: Found {total_missing} missing values that were filled with defaults:")
        for key, count in missing_values.items():
            if count > 0:
                print(f"  {key}: {count} instances")

    return features_list, edge_list, labels_dict, node_mapping, diagnostics


def normalize_features(features_list):
    """
    Normalize feature vectors to improve GNN training.

    Args:
        features_list: List of feature vectors

    Returns:
        normalized_features: List of normalized feature vectors
    """
    # Convert to numpy array for easier processing
    features_array = np.array(features_list)

    # Check for NaN or inf values
    bad_values = np.isnan(features_array) | np.isinf(features_array)
    if np.any(bad_values):
        num_bad = np.sum(bad_values)
        print(f"Warning: Found {num_bad} NaN/inf values. Replacing with zeros.")
        features_array[bad_values] = 0.0

    # Get feature dimension
    num_samples, num_features = features_array.shape

    # Skip the last feature (home/away indicator) which is already normalized
    normalized_array = features_array.copy()

    # Normalize each feature column except the last one
    for i in range(num_features - 1):
        col = features_array[:, i]
        col_min = np.min(col)
        col_max = np.max(col)

        # Only normalize if there's a range of values
        if col_max > col_min:
            normalized_array[:, i] = (col - col_min) / (col_max - col_min)

    # Convert back to list of numpy arrays
    normalized_features = [normalized_array[i] for i in range(num_samples)]

    print(f"Features normalized. Shape: {normalized_array.shape}")
    return normalized_features


def print_feature_distributions(feature_values, window_sizes):
    """
    Print distribution information for each feature type.

    Args:
        feature_values: Dictionary of feature values
        window_sizes: List of window sizes used
    """
    print("\n=== Feature Distributions ===")

    # Group features by type
    feature_types = {
        'Win Rate': [f'win_rate_{w}' for w in window_sizes],
        'Goal Rate': [f'goal_rate_{w}' for w in window_sizes],
        'Goals Against Rate': [f'goals_against_rate_{w}' for w in window_sizes],
        'Other': ['days_since_last_game', 'home_indicator']
    }

    for type_name, feature_keys in feature_types.items():
        print(f"\n{type_name} Features:")
        for key in feature_keys:
            if key in feature_values and feature_values[key]:
                values = feature_values[key]
                print(f"  {key}:")
                print(f"    Range: {np.min(values):.4f} to {np.max(values):.4f}")
                print(f"    Mean: {np.mean(values):.4f} (std: {np.std(values):.4f})")
                print(
                    f"    Zeros: {sum(1 for v in values if v == 0)} ({sum(1 for v in values if v == 0) / len(values) * 100:.1f}%)")


def extract_features_from_graph(data_graph, window_sizes=[5]):
    """
    Extract features from the graph for GNN input, supporting multiple window sizes.
    This is a wrapper around the diagnostic version that maintains the original interface.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to outcome labels (1 for home win, 0 for away win)
        node_mapping: Dictionary mapping node IDs to their indices in features_list
    """
    # Call the diagnostic version
    features_list, edge_list, labels_dict, node_mapping, diagnostics = extract_features_from_graph_with_diagnostics(
        data_graph, window_sizes
    )

    # Normalize features
    features_list = normalize_features(features_list)

    # Print feature distributions
    if 'feature_stats' in diagnostics:
        print_feature_distributions(diagnostics['feature_stats'], window_sizes)

    return features_list, edge_list, labels_dict, node_mapping


def get_simple_node_stats(graph):
    """
    Get simple statistics about nodes in the graph.

    Args:
        graph: NetworkX graph containing hockey data

    Returns:
        Dictionary with basic node statistics
    """
    # Count node types
    node_types = Counter()
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] += 1

    print(f"Node type counts:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")

    # Focus on TGP nodes (Team Game Performance)
    tgp_nodes = [node for node, data in graph.nodes(data=True)
                 if data.get('type') == 'team_game_performance']

    print(f"\nFound {len(tgp_nodes)} TGP nodes")

    # Sample a few TGP nodes to examine their structure
    if tgp_nodes:
        sample_size = min(3, len(tgp_nodes))
        print(f"\nExamining {sample_size} sample TGP nodes:")

        for i, node in enumerate(tgp_nodes[:sample_size]):
            print(f"\nSample TGP node {i + 1}: {node}")
            data = graph.nodes[node]

            # Count attribute types
            attribute_types = defaultdict(int)
            for key, value in data.items():
                attribute_types[type(value).__name__] += 1

            print("  Attribute type counts:")
            for attr_type, count in attribute_types.items():
                print(f"    {attr_type}: {count}")

            # Show sample attributes
            print("  Sample attributes:")
            for key, value in list(data.items())[:10]:
                print(f"    {key}: {type(value).__name__} = {value}")

        # Check for key attributes across all TGP nodes
        print("\nKey attribute availability across all TGP nodes:")
        key_attrs = ['game_date', 'days_since_last_game', 'home', 'win', 'loss', 'goal', 'goal_against']

        for attr in key_attrs:
            count = sum(1 for node in tgp_nodes if attr in graph.nodes[node])
            percentage = (count / len(tgp_nodes)) * 100
            print(f"  {attr}: {count} nodes ({percentage:.1f}%)")

        # Look for historical statistics
        hist_prefix = 'hist_'
        hist_attrs = set()

        for node in tgp_nodes:
            for attr in graph.nodes[node]:
                if attr.startswith(hist_prefix):
                    hist_attrs.add(attr)

        print(f"\nFound {len(hist_attrs)} historical attributes")
        if hist_attrs:
            print("Sample historical attributes:")
            for attr in sorted(list(hist_attrs))[:10]:
                print(f"  {attr}")

    # Basic statistics about game nodes
    game_nodes = [node for node, data in graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"\nFound {len(game_nodes)} game nodes")

    # Check edges between games and TGPs
    if game_nodes and tgp_nodes:
        game_tgp_edges = sum(1 for u, v, data in graph.edges(data=True)
                             if (u in game_nodes and v in tgp_nodes) or
                             (v in game_nodes and u in tgp_nodes))

        print(f"Game-to-TGP edges: {game_tgp_edges}")

    return {
        'node_type_counts': dict(node_types),
        'tgp_count': len(tgp_nodes),
        'game_count': len(game_nodes)
    }


def examine_tgp_feature_stats(graph, window_sizes=[5]):
    """
    Examine statistics and distributions of key features in TGP nodes.

    Args:
        graph: NetworkX graph containing hockey data
        window_sizes: List of window sizes for historical features

    Returns:
        Dictionary with feature statistics
    """
    # Find all TGP nodes
    tgp_nodes = [node for node, data in graph.nodes(data=True)
                 if data.get('type') == 'team_game_performance']

    if not tgp_nodes:
        print("No TGP nodes found in graph")
        return {}

    print(f"\nExamining key features for {len(tgp_nodes)} TGP nodes:")

    # Define key features to examine
    key_features = ['win', 'loss', 'goal', 'goal_against', 'days_since_last_game']

    # Add historical features for window sizes
    for window in window_sizes:
        for stat in ['win', 'goal', 'goal_against']:
            key_features.append(f'hist_{window}_{stat}')

    feature_stats = {}

    # Collect values for each feature
    for feature in key_features:
        values = []

        for node in tgp_nodes:
            data = graph.nodes[node]
            if feature in data:
                value = data[feature]

                # Handle list values
                if isinstance(value, list):
                    # Sum the values across periods
                    values.append(sum(value))
                else:
                    values.append(value)

        # Calculate basic statistics
        if values:
            feature_stats[feature] = {
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

            # Count zeros and missing values
            zeros = sum(1 for v in values if v == 0)
            feature_stats[feature]['zeros'] = zeros
            feature_stats[feature]['zeros_pct'] = (zeros / len(values)) * 100

            print(f"\n{feature}:")
            print(f"  Available in {len(values)}/{len(tgp_nodes)} nodes ({len(values) / len(tgp_nodes) * 100:.1f}%)")
            print(f"  Range: {feature_stats[feature]['min']} to {feature_stats[feature]['max']}")
            print(f"  Mean: {feature_stats[feature]['mean']:.4f} (std: {feature_stats[feature]['std']:.4f})")
            print(f"  Zeros: {feature_stats[feature]['zeros']} ({feature_stats[feature]['zeros_pct']:.1f}%)")

    return feature_stats


def extract_features_from_graph(data_graph, window_sizes=[5]):
    """
    Extract features from the graph for GNN input, supporting multiple window sizes.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to outcome labels (1 for home win, 0 for away win)
        node_mapping: Dictionary mapping node IDs to their indices in features_list
    """
    print(f"Extracting features with window sizes {window_sizes}...")

    # Initialize containers
    features_list = []
    node_mapping = {}
    labels_dict = {}

    # Get all game nodes
    game_nodes = [node for node, data in data_graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"Found {len(game_nodes)} game nodes in the graph")

    feature_count = 0
    labeled_count = 0

    # Process each game
    for game_id in game_nodes:
        game_data = data_graph.nodes[game_id]
        home_team = game_data['home_team']
        away_team = game_data['away_team']

        # Get TGP nodes for this game
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        if home_tgp in data_graph.nodes and away_tgp in data_graph.nodes:
            home_tgp_data = data_graph.nodes[home_tgp]
            away_tgp_data = data_graph.nodes[away_tgp]

            # Extract home team features for all window sizes
            home_features = []
            for window_size in window_sizes:
                # Historical win rate
                if f'hist_{window_size}_win' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                    wins = sum(home_tgp_data[f'hist_{window_size}_win'])
                    games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    win_rate = wins / games if games > 0 else 0.5
                    home_features.append(win_rate)
                else:
                    home_features.append(0.5)  # Default win rate

                # Historical goal rate
                if f'hist_{window_size}_goal' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                    goals = sum(home_tgp_data[f'hist_{window_size}_goal'])
                    games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    goal_rate = goals / games if games > 0 else 0
                    home_features.append(goal_rate)
                else:
                    home_features.append(0)  # Default goal rate

                # Recent goals against
                if f'hist_{window_size}_goal_against' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                    goals_against = sum(home_tgp_data[f'hist_{window_size}_goal_against'])
                    games = sum(home_tgp_data[f'hist_{window_size}_games'])
                    goals_against_rate = goals_against / games if games > 0 else 0
                    home_features.append(goals_against_rate)
                else:
                    home_features.append(0)  # Default goals against rate

            # Days since last game
            if 'days_since_last_game' in home_tgp_data:
                home_features.append(min(home_tgp_data['days_since_last_game'], 30) / 30)  # Normalize
            else:
                home_features.append(1.0)  # Default (max) days since last game

            # Home advantage indicator
            home_features.append(1.0)  # Home team

            # Extract away team features for all window sizes
            away_features = []
            for window_size in window_sizes:
                # Historical win rate
                if f'hist_{window_size}_win' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                    wins = sum(away_tgp_data[f'hist_{window_size}_win'])
                    games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    win_rate = wins / games if games > 0 else 0.5
                    away_features.append(win_rate)
                else:
                    away_features.append(0.5)  # Default win rate

                # Historical goal rate
                if f'hist_{window_size}_goal' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                    goals = sum(away_tgp_data[f'hist_{window_size}_goal'])
                    games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    goal_rate = goals / games if games > 0 else 0
                    away_features.append(goal_rate)
                else:
                    away_features.append(0)  # Default goal rate

                # Recent goals against
                if f'hist_{window_size}_goal_against' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                    goals_against = sum(away_tgp_data[f'hist_{window_size}_goal_against'])
                    games = sum(away_tgp_data[f'hist_{window_size}_games'])
                    goals_against_rate = goals_against / games if games > 0 else 0
                    away_features.append(goals_against_rate)
                else:
                    away_features.append(0)  # Default goals against rate

            # Days since last game
            if 'days_since_last_game' in away_tgp_data:
                away_features.append(min(away_tgp_data['days_since_last_game'], 30) / 30)  # Normalize
            else:
                away_features.append(1.0)  # Default (max) days since last game

            # Away team indicator
            away_features.append(0.0)  # Away team

            # Game node features (placeholder)
            # Create a feature vector with the same dimension as team features
            game_features = np.zeros(len(home_features), dtype=np.float32)
            game_features[-1] = 0.5  # Mark as game node with a middle value

            # Add features to list and update mapping
            home_idx = len(features_list)
            node_mapping[home_tgp] = home_idx
            features_list.append(np.array(home_features, dtype=np.float32))

            away_idx = len(features_list)
            node_mapping[away_tgp] = away_idx
            features_list.append(np.array(away_features, dtype=np.float32))

            game_idx = len(features_list)
            node_mapping[game_id] = game_idx
            features_list.append(game_features)

            feature_count += 3  # Added 3 nodes with features

            # Create label based on win/loss
            if 'win' in home_tgp_data and 'win' in away_tgp_data:
                if sum(home_tgp_data.get('win', [0, 0, 0])) > 0:
                    labels_dict[game_idx] = 1  # Home win
                    labeled_count += 1
                elif sum(away_tgp_data.get('win', [0, 0, 0])) > 0:
                    labels_dict[game_idx] = 0  # Away win
                    labeled_count += 1

    # Calculate feature dimension for debugging
    if features_list:
        feature_dim = len(features_list[0])
        print(f"Feature dimension: {feature_dim} (includes {len(window_sizes)} window sizes)")
    else:
        feature_dim = 0
        print("Warning: No features extracted")

    print(f"Extracted features for {feature_count} nodes ({labeled_count} labeled games)")

    # Create edge list - connect games to their TGPs
    edge_list = []
    for game_id in game_nodes:
        if game_id in data_graph.nodes:
            if game_id in node_mapping:
                game_idx = node_mapping[game_id]

                # Connect game to home and away TGP nodes
                home_team = data_graph.nodes[game_id]['home_team']
                away_team = data_graph.nodes[game_id]['away_team']
                home_tgp = f"{game_id}_{home_team}"
                away_tgp = f"{game_id}_{away_team}"

                if home_tgp in node_mapping:
                    home_tgp_idx = node_mapping[home_tgp]
                    edge_list.append((game_idx, home_tgp_idx))
                    edge_list.append((home_tgp_idx, game_idx))  # Bidirectional

                if away_tgp in node_mapping:
                    away_tgp_idx = node_mapping[away_tgp]
                    edge_list.append((game_idx, away_tgp_idx))
                    edge_list.append((away_tgp_idx, game_idx))  # Bidirectional

    print(f"Created {len(edge_list)} edges")

    return features_list, edge_list, labels_dict, node_mapping


def predict_game_outcome(model, data_graph, home_team, away_team, window_sizes=[5]):
    """
    Predict the outcome of a game between two teams, supporting multiple window sizes.

    Args:
        model: Trained GNN model
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_sizes: List of historical window sizes to use for features

    Returns:
        probability: Probability of home team winning
    """

    print(f"\nPredicting outcome for {home_team} (home) vs {away_team} (away)...")

    # Set model to evaluation mode
    model.eval()

    # Create a temporary game node for prediction
    temp_game_id = 'temp_prediction_game'
    temp_home_tgp = f'{temp_game_id}_{home_team}'
    temp_away_tgp = f'{temp_game_id}_{away_team}'

    # Extract historical stats for both teams
    # Find most recent TGP nodes for each team
    home_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{home_team}')
                 and data.get('type') == 'team_game_performance']

    away_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{away_team}')
                 and data.get('type') == 'team_game_performance']

    # Sort by game date (most recent first)
    home_tgps = sorted(home_tgps,
                       key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in data_graph.nodes[x] else 0,
                       reverse=True)

    away_tgps = sorted(away_tgps,
                       key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in data_graph.nodes[x] else 0,
                       reverse=True)

    # Create features
    home_features = []
    away_features = []

    # Extract home team features for all window sizes
    if home_tgps:
        home_tgp_data = data_graph.nodes[home_tgps[0]]

        for window_size in window_sizes:
            # Historical win rate
            if f'hist_{window_size}_win' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                wins = sum(home_tgp_data[f'hist_{window_size}_win'])
                games = sum(home_tgp_data[f'hist_{window_size}_games'])
                win_rate = wins / games if games > 0 else 0.5
                home_features.append(win_rate)
            else:
                home_features.append(0.5)  # Default win rate

            # Historical goal rate
            if f'hist_{window_size}_goal' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                goals = sum(home_tgp_data[f'hist_{window_size}_goal'])
                games = sum(home_tgp_data[f'hist_{window_size}_games'])
                goal_rate = goals / games if games > 0 else 0
                home_features.append(goal_rate)
            else:
                home_features.append(0)  # Default goal rate

            # Recent goals against
            if f'hist_{window_size}_goal_against' in home_tgp_data and f'hist_{window_size}_games' in home_tgp_data:
                goals_against = sum(home_tgp_data[f'hist_{window_size}_goal_against'])
                games = sum(home_tgp_data[f'hist_{window_size}_games'])
                goals_against_rate = goals_against / games if games > 0 else 0
                home_features.append(goals_against_rate)
            else:
                home_features.append(0)  # Default goals against rate
    else:
        # Default features if no historical data
        for _ in window_sizes:
            home_features.extend([0.5, 0, 0])  # Default values for each window size

    # Add days since last game (default to 3)
    home_features.append(3 / 30)

    # Home advantage indicator
    home_features.append(1.0)  # Home team

    # Extract away team features for all window sizes
    if away_tgps:
        away_tgp_data = data_graph.nodes[away_tgps[0]]

        for window_size in window_sizes:
            # Historical win rate
            if f'hist_{window_size}_win' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                wins = sum(away_tgp_data[f'hist_{window_size}_win'])
                games = sum(away_tgp_data[f'hist_{window_size}_games'])
                win_rate = wins / games if games > 0 else 0.5
                away_features.append(win_rate)
            else:
                away_features.append(0.5)  # Default win rate

            # Historical goal rate
            if f'hist_{window_size}_goal' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                goals = sum(away_tgp_data[f'hist_{window_size}_goal'])
                games = sum(away_tgp_data[f'hist_{window_size}_games'])
                goal_rate = goals / games if games > 0 else 0
                away_features.append(goal_rate)
            else:
                away_features.append(0)  # Default goal rate

            # Recent goals against
            if f'hist_{window_size}_goal_against' in away_tgp_data and f'hist_{window_size}_games' in away_tgp_data:
                goals_against = sum(away_tgp_data[f'hist_{window_size}_goal_against'])
                games = sum(away_tgp_data[f'hist_{window_size}_games'])
                goals_against_rate = goals_against / games if games > 0 else 0
                away_features.append(goals_against_rate)
            else:
                away_features.append(0)  # Default goals against rate
    else:
        # Default features if no historical data
        for _ in window_sizes:
            away_features.extend([0.5, 0, 0])  # Default values for each window size

    # Add days since last game (default to 3)
    away_features.append(3 / 30)

    # Away team indicator
    away_features.append(0.0)  # Away team

    # Create zero features for game node
    game_features = np.zeros(len(home_features), dtype=np.float32)
    game_features[-1] = 0.5  # Mark as game node

    # Convert features to tensors
    x = torch.tensor([
        home_features,
        away_features,
        game_features
    ], dtype=torch.float)

    # Create edges
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 0, 2, 1],  # Source nodes
        [2, 0, 2, 1, 0, 2, 1, 2]  # Target nodes
    ], dtype=torch.long)

    # Set game index
    game_indices = torch.tensor([2])  # Index of the game node

    # Make prediction
    with torch.no_grad():
        # Forward pass
        out = model(x, edge_index, game_indices)

        # Get probability of home team winning
        probabilities = torch.exp(out)
        home_win_prob = probabilities[0, 1].item()

    print(f"Prediction complete: {home_win_prob:.4f} probability of {home_team} winning")

    return home_win_prob


def train_and_evaluate_gnn_enhanced(data_graph, epochs=200, hidden_channels=64, window_sizes=[5],
                                    lr=0.01, dropout_rate1=0.5, dropout_rate2=0.3, patience=8,
                                    weight_decay=1e-4, lr_reduce_factor=0.3, lr_reduce_patience=5):
    """
    Enhanced training and evaluation of the GNN model with learning rate scheduling and early stopping.
    Now supports multiple window sizes for feature extraction.

    Args:
        data_graph: NetworkX graph containing hockey data
        epochs: Maximum number of training epochs
        hidden_channels: Number of hidden channels in the GNN
        window_sizes: List of historical window sizes to use for features
        lr: Initial learning rate for optimizer
        dropout_rate1: Dropout rate after first layer
        dropout_rate2: Dropout rate after second layer
        patience: Number of epochs to wait for improvement before stopping
        weight_decay: L2 regularization factor
        lr_reduce_factor: Factor by which to reduce learning rate
        lr_reduce_patience: Patience for learning rate scheduler

    Returns:
        model: Trained GNN model
        metrics: Dictionary with accuracy, F1, ROC AUC
        history: Dictionary with training and validation metrics
    """
    print("\n====== Starting Enhanced GNN Training Process ======")

    # Extract features and prepare data
    features, edge_list, labels, node_mapping = extract_features_from_graph(data_graph, window_sizes)
    model_data = prepare_train_test_data(features, edge_list, labels)

    # Get input dimension from features
    in_channels = model_data['x'].shape[1]

    # Create model
    print(f"Creating model with {in_channels} input features, {hidden_channels} hidden channels, "
          f"dropout rates of {dropout_rate1} and {dropout_rate2}...")
    model = EnhancedHockeyGNN(in_channels, hidden_channels, dropout_rate1, dropout_rate2)

    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_roc_auc': []
    }

    # Early stopping variables
    best_f1 = 0
    best_accuracy = 0
    best_model_state = None
    epochs_no_improve = 0

    # Train model
    print(f"\nStarting training for up to {epochs} epochs (with early stopping, patience={patience})...")

    for epoch in range(epochs):
        # Train one epoch
        loss = train_one_epoch(model, model_data, optimizer)
        history['train_loss'].append(loss)

        # Evaluate model
        metrics = evaluate_model_enhanced(model, model_data)
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1'])
        history['val_roc_auc'].append(metrics.get('roc_auc', 0))

        # Update learning rate based on validation performance
        scheduler.step(metrics['f1'])

        # Early stopping check
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_accuracy = metrics['accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1 or epochs_no_improve == patience:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch}/{epochs}, Loss: {loss:.4f}, LR: {current_lr:.6f}, '
                  f'Accuracy: {metrics["accuracy"]:.4f}, F1: {metrics["f1"]:.4f}, '
                  f'ROC AUC: {metrics.get("roc_auc", 0):.4f} '
                  f'({metrics["correct"]}/{metrics["total"]} correct)')

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    # Final evaluation
    final_metrics = evaluate_model_enhanced(model, model_data)

    # Return the trained model, metrics, and history
    return model, final_metrics, history


def plot_training_metrics(history, output_dir="."):
    """
    Plot training and validation metrics with enhanced visualizations.

    Args:
        history: Dictionary with training history
        output_dir: Directory to save plots
    """

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get epoch numbers for x-axis
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. ENHANCED: Loss and metrics on same plot with dual y-axis
    plt.figure(figsize=(12, 7))

    # Primary y-axis for loss
    ax1 = plt.gca()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, history['train_loss'], 'tab:red', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Secondary y-axis for metrics
    ax2 = ax1.twinx()
    ax2.set_ylabel('Performance Metrics', color='tab:blue')
    ax2.plot(epochs, history['val_accuracy'], 'tab:blue', label='Validation Accuracy')
    ax2.plot(epochs, history['val_f1'], 'tab:green', label='Validation F1')
    ax2.plot(epochs, history['val_roc_auc'], 'tab:purple', label='Validation ROC AUC')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Create a single legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('Training Progress: Loss vs. Performance Metrics')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_progress_combined.png")
    plt.close()

    # 2. Find the epoch with the best F1 score and mark it
    best_f1_epoch = np.argmax(history['val_f1']) + 1
    best_f1 = max(history['val_f1'])

    # 3. Create an "overfitting analysis" plot
    plt.figure(figsize=(12, 7))

    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'])
    plt.axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best F1 at epoch {best_f1_epoch}')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot F1 score (usually our primary metric)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['val_f1'], label='F1 Score')
    plt.plot(epochs, history['val_accuracy'], label='Accuracy')
    plt.axvline(x=best_f1_epoch, color='green', linestyle='--', alpha=0.7)
    plt.scatter([best_f1_epoch], [best_f1], color='green', s=100, zorder=5)
    plt.annotate(f'Best F1: {best_f1:.4f}',
                 xy=(best_f1_epoch, best_f1),
                 xytext=(best_f1_epoch + 1, best_f1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    plt.title('Validation Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overfitting_analysis.png")
    plt.close()

    # Also create the original individual plots
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_loss.png")
    plt.close()

    # Plot validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.savefig(f"{output_dir}/validation_accuracy.png")
    plt.close()

    # Plot validation F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'])
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True)
    plt.savefig(f"{output_dir}/validation_f1.png")
    plt.close()

    # Plot validation ROC AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_roc_auc'])
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('Validation ROC AUC')
    plt.grid(True)
    plt.savefig(f"{output_dir}/validation_roc_auc.png")
    plt.close()

    print(f"Training metric plots saved to {output_dir}")
    print(f"Best F1 score of {best_f1:.4f} achieved at epoch {best_f1_epoch}")


def plot_confusion_matrix(metrics, output_path="confusion_matrix.png"):
    """
    Plot and save a confusion matrix.

    Args:
        metrics: Dictionary with model evaluation metrics
        output_path: Path to save the confusion matrix plot
    """

    plt.figure(figsize=(8, 6))

    cm = metrics['confusion_matrix']

    # Calculate percentages for annotations
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)

    # Format annotations to show count and percentage
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f"{cm[i, j]} ({cm_perc[i, j]:.1f}%)"

    # Plot using seaborn for better aesthetics
    ax = sns.heatmap(
        cm, annot=annot, fmt='', cmap='Blues',
        xticklabels=['Away Win', 'Home Win'],
        yticklabels=['Away Win', 'Home Win']
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_roc_curve(metrics, output_path="roc_curve.png"):
    """
    Plot ROC curve for the model.

    Args:
        metrics: Dictionary with model evaluation metrics
        output_path: Path to save the ROC curve plot
    """

    plt.figure(figsize=(8, 6))

    # Get true labels and predicted probabilities
    y_true = metrics['true_labels']
    y_score = metrics['probabilities'][:, 1]  # Probability of class 1 (home win)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (area = {roc_auc:.3f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"ROC curve saved to {output_path}")
    plt.close()


def plot_probability_distribution(metrics, output_path="probability_distribution.png"):
    """
    Plot the distribution of predicted probabilities.

    Args:
        metrics: Dictionary with model evaluation metrics
        output_path: Path to save the plot
    """

    plt.figure(figsize=(10, 6))

    # Get true labels and predicted probabilities
    y_true = metrics['true_labels']
    probs = metrics['probabilities'][:, 1]  # Probability of class 1 (home win)

    # Plot distributions separately for each class
    sns.histplot(
        probs[y_true == 0], bins=20, alpha=0.5,
        label='Away Win (True)', color='red'
    )
    sns.histplot(
        probs[y_true == 1], bins=20, alpha=0.5,
        label='Home Win (True)', color='blue'
    )

    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Predicted Probability of Home Win')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Probability distribution plot saved to {output_path}")
    plt.close()


def analyze_feature_importance(model, data, feature_names=None):
    """
    Analyze feature importance using SHAP values for a GNN model.
    This is an approximation since GNNs are more complex than standard models.

    Args:
        model: Trained GNN model
        data: Dictionary containing model data
        feature_names: List of feature names

    Returns:
        feature_importance: Dictionary with feature importance scores
    """
    print("\n====== Analyzing Feature Importance ======")

    try:
        import shap
    except ImportError:
        print("SHAP package not installed. Please install using: pip install shap")
        return None

    model.eval()

    # Create a wrapper function for SHAP to analyze
    def predict_fn(features):
        """Prediction function for SHAP explainer"""
        import torch
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Replace features in the original data
        x_modified = data['x'].clone()
        x_modified[data['game_indices']] = features_tensor

        # Make prediction using the model
        outputs = model(x_modified, data['edge_index'], data['game_indices'])
        probabilities = torch.exp(outputs)

        # Return home win probability (class 1)
        return probabilities[:, 1].detach().numpy()

    # Get features and labels for games
    game_features = data['x'][data['game_indices']].numpy()
    game_labels = data['y'].numpy()

    # Find training and test indices
    train_game_indices = torch.tensor([i for i, idx in enumerate(data['game_indices'])
                                       if idx in data['train_games']], dtype=torch.long)

    test_game_indices = torch.tensor([i for i, idx in enumerate(data['game_indices'])
                                      if idx in data['test_games']], dtype=torch.long)

    # Create a SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.KernelExplainer(predict_fn,
                                     game_features[train_game_indices.numpy()[:10]])  # Use a subset for efficiency

    # Calculate SHAP values for test samples (use a subset if many test samples)
    print(f"Calculating SHAP values for test samples...")
    test_sample_indices = test_game_indices.numpy()[:20]  # Limit to 20 samples for efficiency
    test_features = game_features[test_sample_indices]
    shap_values = explainer.shap_values(test_features)

    # Process SHAP values
    mean_shap_values = np.abs(shap_values).mean(0)

    # Create feature importance dictionary
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(mean_shap_values))]

    feature_importance = {
        'feature_names': feature_names,
        'importance_scores': mean_shap_values,
        'shap_values': shap_values,
        'test_features': test_features
    }

    # Print feature importance
    print("\nFeature Importance Ranking:")
    for i, (feature, importance) in enumerate(
            sorted(zip(feature_names, mean_shap_values), key=lambda x: x[1], reverse=True)
    ):
        print(f"{i + 1}. {feature}: {importance:.4f}")

    return feature_importance


def plot_feature_importance(feature_importance, output_path="feature_importance.png"):
    """
    Plot feature importance based on SHAP values.

    Args:
        feature_importance: Dictionary with feature importance data
        output_path: Path to save the plot
    """

    # Basic bar plot of feature importance
    plt.figure(figsize=(10, 6))

    # Sort features by importance
    indices = np.argsort(feature_importance['importance_scores'])

    plt.barh(
        range(len(indices)),
        feature_importance['importance_scores'][indices],
        align='center'
    )

    # Set y-tick labels to feature names
    plt.yticks(
        range(len(indices)),
        [feature_importance['feature_names'][i] for i in indices]
    )

    plt.xlabel('Mean |SHAP Value|')
    plt.title('Feature Importance')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    plt.close()

    # Create a summary plot using SHAP's built-in function
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        feature_importance['shap_values'],
        feature_importance['test_features'],
        feature_names=feature_importance['feature_names'],
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_summary.png'))
    print(f"SHAP summary plot saved to {output_path.replace('.png', '_summary.png')}")
    plt.close()


def create_comprehensive_visualization_report(model, metrics, history, feature_importance=None, output_dir="."):
    """
    Create a comprehensive visualization report for model evaluation.

    Args:
        model: Trained model
        metrics: Dictionary with evaluation metrics
        history: Dictionary with training history
        feature_importance: Dictionary with feature importance analysis
        output_dir: Directory to save visualization report
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot training metrics
    plot_training_metrics(history, output_dir)

    # Plot confusion matrix
    plot_confusion_matrix(metrics, f"{output_dir}/confusion_matrix.png")

    # Plot ROC curve
    plot_roc_curve(metrics, f"{output_dir}/roc_curve.png")

    # Plot probability distribution
    plot_probability_distribution(metrics, f"{output_dir}/probability_distribution.png")

    # Plot feature importance if available
    if feature_importance is not None:
        plot_feature_importance(feature_importance, f"{output_dir}/feature_importance.png")

    print(f"Comprehensive visualization report saved to {output_dir}")


def prepare_train_test_data(features, edge_list, labels, test_size=0.2):
    """
    Prepare data for GNN training and testing.

    Args:
        features: List of feature vectors
        edge_list: List of edges as (source, target) tuples
        labels: Dictionary mapping game indices to outcome labels
        test_size: Proportion of data to use for testing

    Returns:
        model_data: Dictionary containing all data needed for training and evaluation
    """

    print(f"Preparing train/test split with test_size={test_size}...")

    # Convert features to tensor
    x = torch.tensor(np.array(features), dtype=torch.float)

    # Convert edge list to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Create a 2x0 empty tensor as a valid but empty edge_index
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create masks for game nodes and labels tensor
    game_indices = list(labels.keys())
    game_mask = torch.zeros(len(features), dtype=torch.bool)
    game_mask[game_indices] = True

    # Create labels tensor
    y = torch.tensor([labels[idx] for idx in game_indices], dtype=torch.long)

    # Split game indices into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(game_indices)), test_size=test_size, random_state=42,
        stratify=[labels[game_indices[i]] for i in range(len(game_indices))]
    )

    # Map local indices back to global indices
    train_games = [game_indices[i] for i in train_indices]
    test_games = [game_indices[i] for i in test_indices]

    # Create train and test masks
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[train_games] = True

    test_mask = torch.zeros(len(features), dtype=torch.bool)
    test_mask[test_games] = True

    print(f"Data prepared with {len(game_indices)} games. "
          f"Training on {len(train_indices)} games, "
          f"testing on {len(test_indices)} games.")

    # Create dictionary with all training data
    model_data = {
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'game_mask': game_mask,
        'game_indices': torch.tensor(game_indices, dtype=torch.long),
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_games': train_games,
        'test_games': test_games
    }

    return model_data


def train_one_epoch(model, data, optimizer):
    """
    Train the GNN model for one epoch.

    Args:
        model: GNN model
        data: Dictionary containing training data
        optimizer: Optimizer for training

    Returns:
        loss: Training loss
    """

    model.train()
    optimizer.zero_grad()

    # Forward pass - only compute predictions for game nodes
    out = model(data['x'], data['edge_index'], data['game_indices'])

    # Find indices of training games within the game_indices tensor
    train_game_indices = torch.tensor([i for i, idx in enumerate(data['game_indices'])
                                       if idx in data['train_games']], dtype=torch.long)

    # Get training predictions and labels
    if len(train_game_indices) > 0:
        train_pred = out[train_game_indices]
        train_labels = data['y'][train_game_indices]

        # Compute loss
        loss = F.nll_loss(train_pred, train_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()
    else:
        print("Warning: No training data in this batch.")
        return 0.0


class EnhancedHockeyGNN(nn.Module):
    """
    Enhanced GNN model for hockey game prediction with configurable dropout rates.
    """

    def __init__(self, in_channels, hidden_channels=64, dropout_rate1=0.5, dropout_rate2=0.3):
        super(EnhancedHockeyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.fc = nn.Linear(hidden_channels, 2)  # Binary classification

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN with enhanced regularization.
        """
        # First graph convolution with batch norm and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate1, training=self.training)

        # Second graph convolution with batch norm and dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate2, training=self.training)

        # Select only game nodes for prediction
        x_games = x[game_indices]

        # Apply final classification layer
        logits = self.fc(x_games)

        return F.log_softmax(logits, dim=1)


def evaluate_model_enhanced(model, data):
    """
    Enhanced evaluation function with ROC AUC score.

    Args:
        model: GNN model
        data: Dictionary containing evaluation data

    Returns:
        metrics: Dictionary with accuracy, F1 score, ROC AUC, etc.
    """
    model.eval()

    with torch.no_grad():
        # Forward pass - only compute predictions for game nodes
        out = model(data['x'], data['edge_index'], data['game_indices'])

        # Find indices of test games within the game_indices tensor
        test_game_indices = torch.tensor([i for i, idx in enumerate(data['game_indices'])
                                          if idx in data['test_games']], dtype=torch.long)

        # Get test predictions and labels
        if len(test_game_indices) > 0:
            test_pred = out[test_game_indices]
            test_labels = data['y'][test_game_indices]

            # Get predicted classes and probabilities
            probs = torch.exp(test_pred)
            _, predicted = test_pred.max(1)

            # Calculate metrics
            correct = predicted.eq(test_labels).sum().item()
            total = len(test_labels)
            accuracy = correct / total

            # Calculate F1 score
            f1 = f1_score(test_labels.cpu().numpy(), predicted.cpu().numpy())

            # Calculate ROC AUC (for home team win probability)
            try:
                roc_auc = roc_auc_score(test_labels.cpu().numpy(), probs[:, 1].cpu().numpy())
            except:
                roc_auc = 0.5  # Default if calculation fails

            # Create confusion matrix
            conf_matrix = confusion_matrix(test_labels.cpu().numpy(), predicted.cpu().numpy())

            return {
                'accuracy': accuracy,
                'f1': f1,
                'roc_auc': roc_auc,
                'correct': correct,
                'total': total,
                'confusion_matrix': conf_matrix,
                'predicted': predicted.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'true_labels': test_labels.cpu().numpy()
            }
        else:
            print("Warning: No test data available for evaluation.")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'roc_auc': 0.5,
                'correct': 0,
                'total': 0,
                'confusion_matrix': np.array([[0, 0], [0, 0]])
            }