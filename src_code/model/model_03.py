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
from src_code.utils.save_graph_utils import load_graph


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

    # Load the graph
    data_graph = load_graph(config.file_paths["graph"])
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

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