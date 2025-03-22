import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from src_code.utils.save_graph_utils import load_filtered_graph


def run_gnn_enhanced(config, config_model):
    """
    Enhanced version of run_gnn with improved model complexity, evaluation, and visualization.
    Now supports multiple window sizes for feature extraction and handles imbalanced classes better.
    Modified to run on CPU only.

    Args:
        config: Configuration object with file paths
        config_model: Configuration object with model parameters
    """
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from sklearn.model_selection import train_test_split
    from collections import Counter, defaultdict

    print("====== Starting Enhanced GNN Training ======")
    print(f"Loading graph from {config.file_paths['graph']}")
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    # Load the graph with date filtering
    data_graph = load_filtered_graph(config.file_paths["graph"], training_cutoff_date)
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Get basic stats about the graph
    stats = get_simple_node_stats(data_graph)

    # Force CPU usage regardless of CUDA availability
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU operation)")

    # Extract features for multi-task prediction with improved handling
    features, edge_list, labels_dict, node_mapping, diagnostics = extract_features_for_multi_task(
        data_graph, window_sizes=config.stat_window_sizes
    )

    # Prepare data for multi-task training - now handles imbalanced classes
    model_data = prepare_multi_task_data(features, edge_list, labels_dict, test_size=0.2)

    # Skip shootout prediction if no shootout data
    skip_shootout = not diagnostics.get('has_shootout_data', False)
    if skip_shootout:
        print("Warning: No shootout data found - shootout prediction will be skipped")

    # Extract parameters from config_model with defaults
    epochs = config_model.num_epochs if hasattr(config_model, 'num_epochs') else 150
    hidden_channels = config_model.hidden_channels if hasattr(config_model, 'hidden_channels') else 128
    window_sizes = config.stat_window_sizes if hasattr(config, 'stat_window_sizes') else [5]
    lr = config_model.learning_rate if hasattr(config_model, 'learning_rate') else 0.005
    dropout_rate1 = config_model.dropout_rate1 if hasattr(config_model, 'dropout_rate1') else 0.4
    dropout_rate2 = config_model.dropout_rate2 if hasattr(config_model, 'dropout_rate2') else 0.3
    patience = config_model.patience if hasattr(config_model, 'patience') else 15  # Increased patience
    weight_decay = config_model.weight_decay if hasattr(config_model, 'weight_decay') else 1e-4

    # Create the multi-task model and move to CPU
    print(
        f"Creating multi-task GNN model with {model_data['x'].shape[1]} input features and {hidden_channels} hidden channels")
    model = MultiTaskHockeyGNN(
        in_channels=model_data['x'].shape[1],
        hidden_channels=hidden_channels,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2
    ).to(device)  # Force to CPU

    # Create output directory for results
    output_dir = config.file_paths["gnn_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis outputs will be saved to {output_dir}")

    # Move data to CPU device before training
    model_data['x'] = model_data['x'].to(device)
    model_data['edge_index'] = model_data['edge_index'].to(device)
    model_data['game_indices'] = model_data['game_indices'].to(device)

    # Move task labels to CPU device
    for task, labels in model_data['task_labels'].items():
        model_data['task_labels'][task] = labels.to(device)

    # Also move task weights if they exist
    if 'task_weights' in model_data:
        for task, weights in model_data['task_weights'].items():
            model_data['task_weights'][task] = weights.to(device)

    # Train the multi-task model with improved training function
    print(f"\nTraining multi-task GNN with up to {epochs} epochs (patience={patience})")
    training_result = train_multi_task_model(
        model=model,
        data=model_data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device  # Pass CPU device to training function
    )

    # Print task-specific performance
    print("\n===== Final Model Performance =====")
    for task, metrics in training_result['test_metrics'].items():
        print(f"Task: {task}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")

    # Save model to file
    model_path = os.path.join(output_dir, "multitask_hockey_gnn.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': model_data['x'].shape[1],
            'hidden_channels': hidden_channels,
            'dropout_rate1': dropout_rate1,
            'dropout_rate2': dropout_rate2
        }
    }, model_path)
    print(f"Model saved to {model_path}")

    # Make predictions for sample games
    print("\n====== Making predictions with trained model ======")
    teams_to_predict = [
        ('TOR', 'MTL'),
        ('BOS', 'TBL'),
        ('EDM', 'CGY'),
        ('NYR', 'NYI'),
        ('PIT', 'WSH')
    ]

    for home_team, away_team in teams_to_predict:
        try:
            predictions = predict_game_multi_task(model, data_graph, home_team, away_team, window_sizes)
            print(f"\nMatch: {home_team} (home) vs {away_team} (away):")
            print(f"  Regulation win probability: {predictions.get('regulation_win', 'N/A'):.4f}")

            if 'overtime_win' in predictions:
                print(f"  Overtime win probability: {predictions.get('overtime_win'):.4f}")
            else:
                print(f"  Overtime win probability: N/A")

            if not skip_shootout and 'shootout_win' in predictions:
                print(f"  Shootout win probability: {predictions.get('shootout_win'):.4f}")
            else:
                print(f"  Shootout win probability: N/A (no shootout data available)")

            print(f"  Probability game goes to overtime: {predictions.get('goes_to_overtime', 'N/A'):.4f}")

            if not skip_shootout and 'goes_to_shootout' in predictions:
                print(f"  Probability game goes to shootout: {predictions.get('goes_to_shootout'):.4f}")
            else:
                print(f"  Probability game goes to shootout: N/A (no shootout data available)")

        except Exception as e:
            print(f"\nError predicting {home_team} vs {away_team}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Calculate and save additional evaluation metrics
    try:
        # Save confusion matrices and ROC curves if matplotlib is available
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        for task in training_result['test_metrics'].keys():
            # Get predictions for this task
            model.eval()
            with torch.no_grad():
                outputs = model(model_data['x'].to(device),
                                model_data['edge_index'].to(device),
                                model_data['game_indices'].to(device))

                task_y = model_data['task_labels'][task].to(device)
                test_mask = (task_y[model_data['test_indices']] >= 0)

                if test_mask.sum() > 0:
                    # Get predictions and labels
                    task_test_idx = torch.tensor([model_data['test_indices'][i] for i, m in enumerate(test_mask) if m],
                                                 dtype=torch.long, device=device)

                    task_pred = outputs[task][task_test_idx]
                    task_probs = torch.exp(task_pred)
                    _, task_pred_class = task_pred.max(1)
                    task_labels_filtered = task_y[model_data['test_indices']][test_mask]

                    # Create confusion matrix
                    cm = confusion_matrix(task_labels_filtered.cpu().numpy(),
                                          task_pred_class.cpu().numpy())

                    # Plot and save confusion matrix
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title(f'Confusion Matrix - {task}')
                    plt.colorbar()

                    if task.endswith('_win'):
                        class_labels = ['Away Win', 'Home Win']
                    else:
                        class_labels = ['No', 'Yes']

                    tick_marks = np.arange(len(class_labels))
                    plt.xticks(tick_marks, class_labels)
                    plt.yticks(tick_marks, class_labels)

                    # Add text annotations
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                     horizontalalignment="center",
                                     color="white" if cm[i, j] > thresh else "black")

                    plt.tight_layout()
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')

                    # Save the figure
                    cm_path = os.path.join(output_dir, f"{task}_confusion_matrix.png")
                    plt.savefig(cm_path)
                    plt.close()

                    # Create ROC curve
                    if len(np.unique(task_labels_filtered.cpu().numpy())) > 1:
                        fpr, tpr, _ = roc_curve(task_labels_filtered.cpu().numpy(),
                                                task_probs[:, 1].cpu().numpy())
                        roc_auc = auc(fpr, tpr)

                        plt.figure(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2,
                                 label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve - {task}')
                        plt.legend(loc="lower right")

                        # Save the figure
                        roc_path = os.path.join(output_dir, f"{task}_roc_curve.png")
                        plt.savefig(roc_path)
                        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate evaluation visualizations: {str(e)}")

    print("\n====== GNN Analysis Complete ======")
    print(f"All analysis outputs saved to {output_dir}")

    return model, training_result


def predict_game_multi_task(model, data_graph, home_team, away_team, window_sizes=[5]):
    """
    Make multi-task predictions for a game between two teams.
    Modified to force CPU operation and handle feature dimension mismatches.

    Args:
        model: Trained MultiTaskHockeyGNN model
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_sizes: List of window sizes for features

    Returns:
        Dictionary with predictions for each task
    """
    print(f"\nPredicting outcomes for {home_team} (home) vs {away_team} (away)...")

    # Set model to evaluation mode
    model.eval()

    # Force CPU usage
    device = torch.device('cpu')
    # Ensure model is on CPU
    model = model.to(device)
    print(f"Using device: {device} (forced CPU operation)")

    # Create features similar to extract_features_for_multi_task
    # Find most recent TGP nodes for each team
    home_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{home_team}')
                 and data.get('type') == 'team_game_performance']

    away_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{away_team}')
                 and data.get('type') == 'team_game_performance']

    # Check if we have data for both teams
    if not home_tgps:
        print(f"Warning: No TGP data found for home team {home_team}")
    if not away_tgps:
        print(f"Warning: No TGP data found for away team {away_team}")

    if not home_tgps or not away_tgps:
        print("Cannot make prediction without data for both teams")
        return {
            'regulation_win': 0.5,  # Default to 50% chance
            'overtime_win': 0.5,
            'shootout_win': 0.5,
            'goes_to_overtime': 0.2,  # Default based on league average
            'goes_to_shootout': 0.06  # Default based on league average
        }

    # Sort by game date (most recent first)
    home_tgps = sorted(home_tgps,
                       key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in data_graph.nodes[x] else 0,
                       reverse=True)

    away_tgps = sorted(away_tgps,
                       key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in data_graph.nodes[x] else 0,
                       reverse=True)

    # Get the most recent TGP data for each team
    home_tgp_data = data_graph.nodes[home_tgps[0]] if home_tgps else {}
    away_tgp_data = data_graph.nodes[away_tgps[0]] if away_tgps else {}

    # Initialize feature arrays with the correct dimensions
    feature_dim = 0
    for node, data in data_graph.nodes(data=True):
        if data.get('type') == 'team_game_performance':
            # Find a TGP node to determine feature dimension
            test_features = extract_team_features(data, window_sizes)
            feature_dim = len(test_features)
            break

    if feature_dim == 0:
        # Fallback calculation based on window sizes
        # Each window size has 10 features + 1 for days since last game + 1 for home/away indicator
        feature_dim = len(window_sizes) * 10 + 2
        print(f"Using estimated feature dimension: {feature_dim}")
    else:
        print(f"Detected feature dimension: {feature_dim}")

    # Extract features using the helper function
    home_features = extract_team_features(home_tgp_data, window_sizes)
    away_features = extract_team_features(away_tgp_data, window_sizes)

    # Ensure home features has home indicator (1.0)
    if home_features[-1] != 1.0:
        home_features[-1] = 1.0  # Home team

    # Ensure away features has away indicator (0.0)
    if away_features[-1] != 0.0:
        away_features[-1] = 0.0  # Away team

    # Combine features using NumPy
    import numpy as np

    # Convert lists to numpy arrays
    home_features_np = np.array(home_features, dtype=np.float32)
    away_features_np = np.array(away_features, dtype=np.float32)

    # Create dummy game node features
    game_features_np = np.zeros(feature_dim, dtype=np.float32)
    game_features_np[-1] = 0.5  # Mark as game node

    # Print shapes for debugging
    print(
        f"Feature shapes - Home: {home_features_np.shape}, Away: {away_features_np.shape}, Game: {game_features_np.shape}")

    # Before stacking, ensure all arrays have the same shape
    if home_features_np.shape != away_features_np.shape or home_features_np.shape != game_features_np.shape:
        print(f"Warning: Feature dimension mismatch! Fixing dimensions...")

        # Make sure all arrays have the same dimension
        max_dim = max(len(home_features_np), len(away_features_np), len(game_features_np))

        # Resize arrays if needed
        if len(home_features_np) < max_dim:
            home_features_np = np.pad(home_features_np, (0, max_dim - len(home_features_np)), 'constant',
                                      constant_values=0)
        if len(away_features_np) < max_dim:
            away_features_np = np.pad(away_features_np, (0, max_dim - len(away_features_np)), 'constant',
                                      constant_values=0)
        if len(game_features_np) < max_dim:
            game_features_np = np.pad(game_features_np, (0, max_dim - len(game_features_np)), 'constant',
                                      constant_values=0)
            game_features_np[-1] = 0.5  # Ensure game marker is still set

        print(
            f"Adjusted feature shapes - Home: {home_features_np.shape}, Away: {away_features_np.shape}, Game: {game_features_np.shape}")

    # Stack features into a single array
    features_np = np.stack([home_features_np, away_features_np, game_features_np])

    # Create tensor for network input and move to CPU
    x = torch.tensor(features_np, dtype=torch.float).to(device)

    # Create edge connections and move to CPU
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 0, 2, 1],  # Source nodes
        [2, 0, 2, 1, 0, 2, 1, 2]  # Target nodes
    ], dtype=torch.long).to(device)

    # Game index for prediction - created directly on CPU
    game_indices = torch.tensor([2], dtype=torch.long).to(device)  # Index of the game node

    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(x, edge_index, game_indices)

            # Get probabilities for each task
            predictions = {}
            for task, task_pred in outputs.items():
                # Apply exp to get actual probabilities
                probs = torch.exp(task_pred)
                # Get probability of class 1 (home win or "yes")
                predictions[task] = probs[0, 1].item()

        return predictions
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {
            'regulation_win': 0.5,  # Default to 50% chance
            'overtime_win': 0.5,
            'shootout_win': 0.5,
            'goes_to_overtime': 0.2,
            'goes_to_shootout': 0.06
        }


def extract_team_features(team_data, window_sizes):
    """
    Extract features for a team from team game performance data.

    Args:
        team_data: Dictionary of team game performance data
        window_sizes: List of window sizes to use for features

    Returns:
        List of features
    """
    features = []

    # Process each window size
    for window_size in window_sizes:
        # Win rates - overall
        win_rate = 0.5  # Default
        if f'hist_{window_size}_win_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_win_avg'], list):
                win_rate = sum(team_data[f'hist_{window_size}_win_avg']) / 3
            else:
                win_rate = team_data[f'hist_{window_size}_win_avg']
        features.append(win_rate)

        # Win rates - specific for regulation (index 0)
        regulation_win_rate = 0.5  # Default
        if f'hist_{window_size}_win_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_win_avg'], list) and len(
                    team_data[f'hist_{window_size}_win_avg']) > 0:
                regulation_win_rate = team_data[f'hist_{window_size}_win_avg'][0]
            elif not isinstance(team_data[f'hist_{window_size}_win_avg'], list):
                regulation_win_rate = team_data[f'hist_{window_size}_win_avg']
        features.append(regulation_win_rate)

        # Win rates - specific for overtime (index 1)
        overtime_win_rate = 0.5  # Default
        if f'hist_{window_size}_win_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_win_avg'], list) and len(
                    team_data[f'hist_{window_size}_win_avg']) > 1:
                overtime_win_rate = team_data[f'hist_{window_size}_win_avg'][1]
            # If it's not a list, we'll use the overall win rate
        features.append(overtime_win_rate)

        # Win rates - specific for shootout (index 2)
        shootout_win_rate = 0.5  # Default
        if f'hist_{window_size}_win_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_win_avg'], list) and len(
                    team_data[f'hist_{window_size}_win_avg']) > 2:
                shootout_win_rate = team_data[f'hist_{window_size}_win_avg'][2]
            # If it's not a list, we'll use the overall win rate
        features.append(shootout_win_rate)

        # Historical goal rates - overall
        goal_rate = 0.0  # Default
        if f'hist_{window_size}_goal_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_goal_avg'], list):
                goal_rate = sum(team_data[f'hist_{window_size}_goal_avg']) / 3
            else:
                goal_rate = team_data[f'hist_{window_size}_goal_avg']
        features.append(goal_rate)

        # Goal rates - specific for regulation (index 0)
        regulation_goal_rate = 0.0  # Default
        if f'hist_{window_size}_goal_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_goal_avg'], list) and len(
                    team_data[f'hist_{window_size}_goal_avg']) > 0:
                regulation_goal_rate = team_data[f'hist_{window_size}_goal_avg'][0]
            elif not isinstance(team_data[f'hist_{window_size}_goal_avg'], list):
                regulation_goal_rate = team_data[f'hist_{window_size}_goal_avg']
        features.append(regulation_goal_rate)

        # Historical goals against rates - overall
        goals_against_rate = 0.0  # Default
        if f'hist_{window_size}_goal_against_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_goal_against_avg'], list):
                goals_against_rate = sum(team_data[f'hist_{window_size}_goal_against_avg']) / 3
            else:
                goals_against_rate = team_data[f'hist_{window_size}_goal_against_avg']
        features.append(goals_against_rate)

        # Goals against rates - specific for regulation (index 0)
        regulation_goals_against_rate = 0.0  # Default
        if f'hist_{window_size}_goal_against_avg' in team_data:
            if isinstance(team_data[f'hist_{window_size}_goal_against_avg'], list) and len(
                    team_data[f'hist_{window_size}_goal_against_avg']) > 0:
                regulation_goals_against_rate = team_data[f'hist_{window_size}_goal_against_avg'][0]
            elif not isinstance(team_data[f'hist_{window_size}_goal_against_avg'], list):
                regulation_goals_against_rate = team_data[f'hist_{window_size}_goal_against_avg']
        features.append(regulation_goals_against_rate)

        # Add historical frequency of games going to overtime
        overtime_freq = 0.0  # Default
        if f'hist_{window_size}_games' in team_data:
            if isinstance(team_data[f'hist_{window_size}_games'], list) and len(
                    team_data[f'hist_{window_size}_games']) > 1:
                total_games = sum(team_data[f'hist_{window_size}_games'])
                overtime_games = team_data[f'hist_{window_size}_games'][1] + \
                                 team_data[f'hist_{window_size}_games'][2]
                overtime_freq = overtime_games / total_games if total_games > 0 else 0.0
        features.append(overtime_freq)

        # Add historical frequency of games going to shootout
        shootout_freq = 0.0  # Default
        if f'hist_{window_size}_games' in team_data:
            if isinstance(team_data[f'hist_{window_size}_games'], list) and len(
                    team_data[f'hist_{window_size}_games']) > 2:
                total_games = sum(team_data[f'hist_{window_size}_games'])
                shootout_games = team_data[f'hist_{window_size}_games'][2]
                shootout_freq = shootout_games / total_games if total_games > 0 else 0.0
        features.append(shootout_freq)

    # Days since last game
    if 'days_since_last_game' in team_data:
        days_value = min(team_data['days_since_last_game'], 30) / 30  # Normalize
        features.append(days_value)
    else:
        features.append(1.0)  # Default (max) days since last game

    # Team indicator (will be set in the calling function)
    features.append(0.5)  # Default placeholder

    return features


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

    # First, let's analyze the data to understand what we're working with
    regulation_games = 0
    overtime_games = 0
    shootout_games = 0

    for game_id in game_nodes:
        game_data = data_graph.nodes[game_id]
        if not (game_data.get('home_team') and game_data.get('away_team')):
            continue

        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        if home_tgp in data_graph.nodes and away_tgp in data_graph.nodes:
            home_tgp_data = data_graph.nodes[home_tgp]
            away_tgp_data = data_graph.nodes[away_tgp]

            # Check what period the game ended in
            home_games = [0, 0, 0]
            if isinstance(home_tgp_data.get('games'), list):
                home_games = home_tgp_data['games']

            if home_games[0] == 1 and home_games[1] == 0 and home_games[2] == 0:
                regulation_games += 1
            elif home_games[1] == 1 and home_games[2] == 0:
                overtime_games += 1
            elif home_games[2] == 1:
                shootout_games += 1

    print(
        f"Data analysis: {regulation_games} regulation games, {overtime_games} overtime games, {shootout_games} shootout games")
    has_shootout_data = shootout_games > 0

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
            game_features[-1] = 0.5  # Mark as game node

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

                    # Game did not go to shootout (implied)
                    if has_shootout_data:
                        labels_dict['goes_to_shootout'][game_idx] = 0

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
                    if has_shootout_data:
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
                    if has_shootout_data:
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
    diagnostics['has_shootout_data'] = has_shootout_data

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
    if reg_games > 0:
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
    print(f"Has shootout data: {diagnostics['has_shootout_data']}")

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

    # Normalize the features
    features_list = normalize_features(features_list)

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


class MultiTaskHockeyGNN(nn.Module):
    """
    Multi-task GNN model for hockey game prediction.
    Predicts 5 separate outcomes:
    1. Regulation win probability
    2. Overtime win probability
    3. Shootout win probability
    4. Probability of game going to overtime
    5. Probability of game going to shootout

    This improved version uses residual connections and more advanced layers
    to better handle complex and imbalanced tasks.
    """

    def __init__(self, in_channels, hidden_channels=128, dropout_rate1=0.4, dropout_rate2=0.3):
        super(MultiTaskHockeyGNN, self).__init__()

        # Shared layer structure
        # Input embedding layer
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # GCN layers with residual connections
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Dropout rates
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

        # Task-specific output layers with specialized architecture
        # Regulation win task
        self.regulation_win_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 2)
        )

        # Overtime win task
        self.overtime_win_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 2)
        )

        # Shootout win task
        self.shootout_win_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 2)
        )

        # Game goes to overtime task (potentially imbalanced)
        self.goes_to_overtime_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),  # Higher dropout to prevent overfitting to majority class
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, 2)
        )

        # Game goes to shootout task (potentially imbalanced)
        self.goes_to_shootout_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),  # Higher dropout to prevent overfitting to majority class
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN with residual connections and task-specific outputs.

        Args:
            x: Node features
            edge_index: Edge indices
            game_indices: Indices of game nodes

        Returns:
            Dictionary of task-specific predictions
        """
        # Initial embedding
        h = self.input_linear(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate1, training=self.training)

        # First GCN layer with residual connection
        h1 = self.conv1(h, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate1, training=self.training)
        h = h + h1  # Residual connection

        # Second GCN layer with residual connection
        h2 = self.conv2(h, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout_rate2, training=self.training)
        h = h + h2  # Residual connection

        # Third GCN layer with residual connection
        h3 = self.conv3(h, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout_rate2, training=self.training)
        h = h + h3  # Residual connection

        # Select only game nodes for prediction
        x_games = h[game_indices]

        # Task-specific predictions using deeper networks
        regulation_win_logits = self.regulation_win_fc(x_games)

        # Only compute overtime and shootout tasks if needed
        overtime_win_logits = self.overtime_win_fc(x_games)
        shootout_win_logits = self.shootout_win_fc(x_games)

        # For potentially imbalanced classes, use deeper networks
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
    Now supports handling of class imbalance through task weights.

    Args:
        features: List of feature vectors
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to multi-task labels
        test_size: Proportion of data to use for testing

    Returns:
        model_data: Dictionary containing all data needed for training and evaluation
    """
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
    task_weights = {}

    for task, task_dict in labels_dict.items():
        # Create a tensor with -1 for games without a label for this task
        y = torch.full((len(game_indices),), -1, dtype=torch.long)

        # Fill in the labels for games that have them
        for i, idx in enumerate(game_indices):
            if idx in task_dict:
                y[i] = task_dict[idx]

        task_labels[task] = y

        # Calculate class weights for the task (to handle imbalance)
        valid_labels = y[y >= 0]
        if len(valid_labels) > 0:
            # Count the occurrences of each class
            class_counts = torch.bincount(valid_labels)
            if len(class_counts) > 1:  # Binary classification
                # Calculate weights as 1/frequency
                weights = 1.0 / class_counts.float()
                # Normalize weights so they sum to 1
                weights = weights / weights.sum()

                # Adjust weights to be less extreme (avoid exploding gradients)
                min_weight = weights.min()
                max_weight = weights.max()
                if max_weight > 5 * min_weight:
                    # Scale weights to be within a reasonable range
                    weights = (weights - min_weight) / (max_weight - min_weight) * 4 + 1

                task_weights[task] = weights
            else:
                # Only one class present - could be a problem
                print(f"Warning: Task '{task}' has only one class present in the data")
                task_weights[task] = torch.tensor([1.0])

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

    # Print task-specific statistics including weights
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

            # Print weights if available
            if task in task_weights:
                weights = task_weights[task]
                print(f"  Class weights: {weights.tolist()}")
                # Calculate effective sample counts after weighting
                effective_counts = class_counts.float() * weights
                print(f"  Effective samples after weighting: {effective_counts.tolist()}")

    print(f"Data prepared with {len(game_indices)} games. "
          f"Training on {len(train_indices)} games, "
          f"testing on {len(test_indices)} games.")

    # Create dictionary with all training data
    model_data = {
        'x': x,
        'edge_index': edge_index,
        'task_labels': task_labels,
        'task_weights': task_weights,
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


def train_multi_task_model(model, data, epochs=200, lr=0.01, weight_decay=1e-4, patience=10, device=None):
    """
    Train a multi-task GNN model with early stopping.
    Now supports class weights to handle imbalanced data.
    Modified to ensure CPU-only operation.

    Args:
        model: MultiTaskHockeyGNN model
        data: Dictionary containing training data
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization factor
        patience: Number of epochs to wait for improvement before stopping
        device: Device to use for training (should be CPU)

    Returns:
        Dictionary with training history and best model state
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

    print(f"Training multi-task model for up to {epochs} epochs (patience={patience})...")

    # Force CPU usage regardless of what's passed in
    device = torch.device('cpu')
    print(f"Training on device: {device} (forced CPU operation)")

    # Ensure model is on CPU
    model = model.to(device)

    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience // 2, verbose=True
    )

    # Ensure data is on CPU
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)

    # Convert task labels to CPU device
    task_labels = {}
    for task, labels in data['task_labels'].items():
        task_labels[task] = labels.to(device)

    # Move task weights to CPU device
    task_weights = {}
    for task, weights in data.get('task_weights', {}).items():
        task_weights[task] = weights.to(device)

    # Get train and test indices
    train_indices = data['train_indices']
    test_indices = data['test_indices']

    # Training history
    history = {
        'train_loss': [],
        'train_metrics': {},
        'val_metrics': {}
    }

    # Initialize task metrics in history
    for task in task_labels.keys():
        history['train_metrics'][task] = {'accuracy': [], 'f1': [], 'roc_auc': [], 'precision': [], 'recall': []}
        history['val_metrics'][task] = {'accuracy': [], 'f1': [], 'roc_auc': [], 'precision': [], 'recall': []}

    # Early stopping variables
    best_avg_f1 = 0
    best_model_state = None
    epochs_no_improve = 0

    # Train model
    for epoch in range(epochs):
        # Training mode
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x, edge_index, game_indices)

        # Calculate loss for each task
        total_loss = 0
        task_losses = {}
        num_tasks_with_data = 0

        for task, task_y in task_labels.items():
            # Find training examples for this task
            mask = (task_y[train_indices] >= 0)
            if mask.sum() > 0:
                # Get predictions and labels for this task
                task_train_idx = torch.tensor([train_indices[i] for i, m in enumerate(mask) if m],
                                              dtype=torch.long, device=device)

                task_pred = outputs[task][task_train_idx]
                task_labels_filtered = task_y[train_indices][mask]

                # Use class weights if available
                if task in task_weights:
                    # Use weighted NLL loss
                    task_loss = F.nll_loss(task_pred, task_labels_filtered, weight=task_weights[task])
                else:
                    # Use standard NLL loss
                    task_loss = F.nll_loss(task_pred, task_labels_filtered)

                total_loss += task_loss
                task_losses[task] = task_loss.item()
                num_tasks_with_data += 1

        # Normalize the loss by the number of tasks
        if num_tasks_with_data > 0:
            total_loss /= num_tasks_with_data

            # Backward pass
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Record training loss
        history['train_loss'].append(total_loss.item())

        # Evaluate on training and validation sets
        model.eval()
        with torch.no_grad():
            all_metrics = evaluate_multi_task_model(model, data, device)

            # Store metrics in history
            for task, metrics in all_metrics['train_metrics'].items():
                for metric_name, value in metrics.items():
                    history['train_metrics'][task][metric_name].append(value)

            for task, metrics in all_metrics['test_metrics'].items():
                for metric_name, value in metrics.items():
                    history['val_metrics'][task][metric_name].append(value)

        # Calculate average F1 across all tasks for early stopping
        active_tasks = [task for task in task_labels.keys()
                        if task in all_metrics['test_metrics'] and 'f1' in all_metrics['test_metrics'][task]]

        if active_tasks:
            # Prioritize regulation_win and goes_to_overtime tasks
            priority_tasks = [task for task in active_tasks
                              if task in ['regulation_win', 'goes_to_overtime']]

            # If we have the priority tasks, use those for early stopping
            if priority_tasks:
                avg_f1 = sum(all_metrics['test_metrics'][task]['f1'] for task in priority_tasks) / len(priority_tasks)
            else:
                avg_f1 = sum(all_metrics['test_metrics'][task]['f1'] for task in active_tasks) / len(active_tasks)

            # Update learning rate based on average F1
            scheduler.step(avg_f1)

            # Early stopping check
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            epochs_no_improve += 1

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1 or epochs_no_improve == patience:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss.item():.4f}")

            # Print task-specific metrics
            for task in active_tasks:
                val_acc = all_metrics['test_metrics'][task]['accuracy']
                val_f1 = all_metrics['test_metrics'][task]['f1']
                val_prec = all_metrics['test_metrics'][task].get('precision', 0.0)
                val_rec = all_metrics['test_metrics'][task].get('recall', 0.0)
                print(f"  {task}: Acc={val_acc:.4f}, F1={val_f1:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with average F1: {best_avg_f1:.4f}")
    else:
        print("Warning: No best model found - using final model")

    # Final evaluation
    final_metrics = evaluate_multi_task_model(model, data, device)

    # Print final metrics
    print("\nFinal model performance:")
    for task, metrics in final_metrics['test_metrics'].items():
        print(f"Task: {task}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Return the trained model, history, and metrics
    return {
        'model': model,
        'history': history,
        'best_avg_f1': best_avg_f1,
        'epochs_trained': len(history['train_loss']),
        'best_model_state': best_model_state,
        'test_metrics': final_metrics['test_metrics']
    }


def evaluate_multi_task_model(model, data, device=None):
    """
    Evaluate a multi-task GNN model with expanded metrics.
    Modified to force CPU operation.

    Args:
        model: MultiTaskHockeyGNN model
        data: Dictionary containing model data
        device: Device to use for evaluation (should be CPU)

    Returns:
        Dictionary with metrics for each task
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

    # Force CPU usage regardless of what's passed in
    device = torch.device('cpu')

    # Ensure data is on CPU
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    train_indices = data['train_indices']
    test_indices = data['test_indices']

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(x, edge_index, game_indices)

        # Calculate metrics for each task
        train_metrics = {}
        test_metrics = {}

        for task, task_y in data['task_labels'].items():
            task_y = task_y.to(device)

            # Training metrics
            train_mask = (task_y[train_indices] >= 0)
            if train_mask.sum() > 0:
                # Get predictions and labels for this task
                task_train_idx = torch.tensor([train_indices[i] for i, m in enumerate(train_mask) if m],
                                              dtype=torch.long, device=device)

                task_pred = outputs[task][task_train_idx]
                task_probs = torch.exp(task_pred)
                _, task_pred_class = task_pred.max(1)
                task_labels_filtered = task_y[train_indices][train_mask]

                # Calculate expanded metrics
                task_train_metrics = calculate_binary_metrics(
                    task_pred_class.cpu().numpy(),
                    task_probs[:, 1].cpu().numpy(),
                    task_labels_filtered.cpu().numpy()
                )

                train_metrics[task] = task_train_metrics

            # Testing metrics
            test_mask = (task_y[test_indices] >= 0)
            if test_mask.sum() > 0:
                # Get predictions and labels for this task
                task_test_idx = torch.tensor([test_indices[i] for i, m in enumerate(test_mask) if m],
                                             dtype=torch.long, device=device)

                task_pred = outputs[task][task_test_idx]
                task_probs = torch.exp(task_pred)
                _, task_pred_class = task_pred.max(1)
                task_labels_filtered = task_y[test_indices][test_mask]

                # Calculate expanded metrics
                task_test_metrics = calculate_binary_metrics(
                    task_pred_class.cpu().numpy(),
                    task_probs[:, 1].cpu().numpy(),
                    task_labels_filtered.cpu().numpy()
                )

                test_metrics[task] = task_test_metrics

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }


def calculate_binary_metrics(y_pred, y_proba, y_true):
    """
    Calculate expanded metrics for binary classification.

    Args:
        y_pred: Predicted class labels
        y_proba: Predicted probability of class 1
        y_true: True class labels

    Returns:
        Dictionary with accuracy, F1, ROC AUC, precision, and recall
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

    # Check if we have enough samples of each class for valid metrics
    unique_classes = np.unique(y_true)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Initialize other metrics with defaults
    f1 = 0.0
    roc_auc = 0.5
    precision = 0.0
    recall = 0.0

    if len(unique_classes) > 1:
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='binary')

        # Calculate precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)

        # Calculate ROC AUC if we have probabilities for both classes
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            # Use default if calculation fails
            roc_auc = 0.5
    else:
        # Only one class in the data - metrics won't be meaningful
        class_id = unique_classes[0]
        if class_id == 1:
            precision = 1.0 if np.any(y_pred == 1) else 0.0
            recall = 1.0
        else:  # class_id == 0
            precision = 0.0
            recall = 0.0 if np.any(y_pred == 1) else 1.0

    # Check for NaN values and replace with 0
    if np.isnan(precision):
        precision = 0.0
    if np.isnan(recall):
        recall = 0.0
    if np.isnan(f1):
        f1 = 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall
    }


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

    # Basic statistics about game nodes
    game_nodes = [node for node, data in graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"Found {len(game_nodes)} game nodes")

    return {
        'node_type_counts': dict(node_types),
        'tgp_count': len(tgp_nodes),
        'game_count': len(game_nodes)
    }
