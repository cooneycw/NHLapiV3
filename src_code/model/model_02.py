import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src_code.utils.save_graph_utils import load_filtered_graph
import copy


def run_goal_prediction_gnn(config, config_model):
    """
    Wrapper function that maintains backward compatibility with existing code.
    This simply calls the new implementation with StandardScaler.

    Args:
        config: Configuration object with file paths and settings
        config_model: Configuration object with model parameters

    Returns:
        model: Trained GNN model
        results: Dictionary with evaluation metrics and results
    """
    print("Using enhanced version with StandardScaler...")
    return run_goal_prediction_gnn_with_scalers(config, config_model)


class MultiHeadedHockeyGNN(nn.Module):
    """
    Multi-headed GNN for predicting various hockey game metrics.

    Architecture:
    - Shared feature extraction layers
    - Separate prediction heads for home goals, away goals, and shots on goal
    - Configurable dedicated outcome classification head
    """

    def __init__(self, in_channels, hidden_channels=128, dropout_rate1=0.4, dropout_rate2=0.3,
                 with_outcome_head=True, outcome_head_layers=None):
        super(MultiHeadedHockeyGNN, self).__init__()

        # Input embedding layer
        self.input_feature_bn = nn.BatchNorm1d(in_channels)  # First: normalize raw input features
        self.input_linear = nn.Linear(in_channels, hidden_channels)  # Second: transform to hidden dimension
        self.input_bn = nn.BatchNorm1d(hidden_channels)  # Third: normalize the transformed features

        # GNN layers with residual connections
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Dropout rates
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

        # Whether to use dedicated outcome head
        self.with_outcome_head = with_outcome_head

        # Goal prediction heads
        self.home_goal_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 1)  # Regression output for home goals
        )

        self.away_goal_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 1)  # Regression output for away goals
        )

        # Shots on goal prediction heads
        self.home_shots_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 1)  # Regression output for home shots
        )

        self.away_shots_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, 1)  # Regression output for away shots
        )

        # Dedicated outcome classification head with configurable architecture
        if with_outcome_head:
            if outcome_head_layers is None:
                # Default architecture if no layers specified
                outcome_head_layers = [hidden_channels // 2, hidden_channels // 4]

            # Build the outcome prediction network dynamically
            outcome_layers = []

            # Input layer
            prev_size = hidden_channels

            # Add all hidden layers with BatchNorm, ReLU, and Dropout
            for layer_size in outcome_head_layers:
                outcome_layers.extend([
                    nn.Linear(prev_size, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate2)
                ])
                prev_size = layer_size

            # Add the final output layer (3 classes: home win, draw, away win)
            outcome_layers.append(nn.Linear(prev_size, 3))

            # Create the sequential model
            self.outcome_fc = nn.Sequential(*outcome_layers)

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN with residual connections.

        Args:
            x: Node features
            edge_index: Edge indices
            game_indices: Indices of game nodes

        Returns:
            Dictionary with home and away goal and shots on goal predictions,
            and optionally outcome classification logits
        """
        # Initial embedding
        # Normalize raw input features first
        x = self.input_feature_bn(x)
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

        # Generate predictions from each head
        home_goals = self.home_goal_fc(x_games)
        away_goals = self.away_goal_fc(x_games)
        home_shots = self.home_shots_fc(x_games)
        away_shots = self.away_shots_fc(x_games)

        result = {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_shot_attempts': home_shots,
            'away_shot_attempts': away_shots
        }

        # Add outcome logits if using outcome head
        if self.with_outcome_head:
            outcome_logits = self.outcome_fc(x_games)
            result['outcome_logits'] = outcome_logits

        return result

    def get_embedding(self, x, edge_index):
        """
        Get node embeddings for analysis.

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Node embeddings
        """
        # Initial embedding
        x = self.input_feature_bn(x)
        h = self.input_linear(x)
        h = self.input_bn(h)
        h = F.relu(h)

        # First GCN layer with residual connection
        h1 = self.conv1(h, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h = h + h1  # Residual connection

        # Second GCN layer with residual connection
        h2 = self.conv2(h, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h = h + h2  # Residual connection

        # Third GCN layer with residual connection
        h3 = self.conv3(h, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h = h + h3  # Residual connection

        return h


def prepare_goal_prediction_data_with_scalers(features, edge_list, labels_dict, test_size=0.2, val_size=0.1):
    """
    Prepare data for multi-task prediction model training with standardized targets.

    Args:
        features: List of feature vectors
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary with goal and shot labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation

    Returns:
        model_data: Dictionary containing all data needed for training and evaluation
    """
    print(
        f"Preparing multi-task prediction data with standardized targets (test_size={test_size}, val_size={val_size})...")

    # Convert features to tensor
    x = torch.tensor(np.array(features), dtype=torch.float)

    # Convert edge list to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Create a 2x0 empty tensor as a valid but empty edge_index
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Get all game indices that have complete labels
    goal_indices = set(labels_dict['home_goals'].keys()) & set(labels_dict['away_goals'].keys())
    shot_indices = set(labels_dict['home_shot_attempts'].keys()) & set(labels_dict['away_shot_attempts'].keys())

    # We want games that have both goal and shot data
    game_indices = sorted(list(goal_indices & shot_indices))

    print(f"Found {len(goal_indices)} games with goal data")
    print(f"Found {len(shot_indices)} games with shot data")
    print(f"Using {len(game_indices)} games with complete data for training")

    game_mask = torch.zeros(len(features), dtype=torch.bool)
    game_mask[game_indices] = True

    # Create label tensors for goals and shots
    home_goals_raw = torch.tensor([labels_dict['home_goals'][idx] for idx in game_indices], dtype=torch.float)
    away_goals_raw = torch.tensor([labels_dict['away_goals'][idx] for idx in game_indices], dtype=torch.float)
    home_shots_raw = torch.tensor([labels_dict['home_shot_attempts'][idx] for idx in game_indices], dtype=torch.float)
    away_shots_raw = torch.tensor([labels_dict['away_shot_attempts'][idx] for idx in game_indices], dtype=torch.float)

    # Create train/val/test splits - FIRST, to avoid data leakage when using StandardScaler
    # First, split into train+val and test
    train_val_indices, test_indices = train_test_split(
        range(len(game_indices)), test_size=test_size, random_state=42
    )

    # Then split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced size of train+val
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_adjusted, random_state=42
    )

    # Map local indices back to global indices
    train_games = [game_indices[i] for i in train_indices]
    val_games = [game_indices[i] for i in val_indices]
    test_games = [game_indices[i] for i in test_indices]

    # Create train, val, and test masks
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[train_games] = True

    val_mask = torch.zeros(len(features), dtype=torch.bool)
    val_mask[val_games] = True

    test_mask = torch.zeros(len(features), dtype=torch.bool)
    test_mask[test_games] = True

    # Calculate original dataset statistics (store for later reference)
    home_goals_mean = home_goals_raw[train_indices].mean().item()
    home_goals_std = home_goals_raw[train_indices].std().item()
    away_goals_mean = away_goals_raw[train_indices].mean().item()
    away_goals_std = away_goals_raw[train_indices].std().item()
    home_shots_mean = home_shots_raw[train_indices].mean().item()
    home_shots_std = home_shots_raw[train_indices].std().item()
    away_shots_mean = away_shots_raw[train_indices].mean().item()
    away_shots_std = away_shots_raw[train_indices].std().item()

    # Print original statistics
    print(f"Original goal statistics (training set):")
    print(f"  Home goals: mean={home_goals_mean:.2f}, std={home_goals_std:.2f}")
    print(f"  Away goals: mean={away_goals_mean:.2f}, std={away_goals_std:.2f}")

    print(f"Original shot attempt statistics (training set):")
    print(f"  Home shot attempts: mean={home_shots_mean:.2f}, std={home_shots_std:.2f}")
    print(f"  Away shot attempts: mean={away_shots_mean:.2f}, std={away_shots_std:.2f}")

    # Now standardize goals and shots using sklearn's StandardScaler
    # Create scalers
    goals_scaler = StandardScaler()
    shots_scaler = StandardScaler()

    # Reshape data for fitting (scikit-learn expects 2D arrays)
    # Fit scalers only on training data to prevent data leakage
    all_goals_train = np.vstack([
        home_goals_raw[train_indices].numpy().reshape(-1, 1),
        away_goals_raw[train_indices].numpy().reshape(-1, 1)
    ])
    all_shots_train = np.vstack([
        home_shots_raw[train_indices].numpy().reshape(-1, 1),
        away_shots_raw[train_indices].numpy().reshape(-1, 1)
    ])

    # Fit scalers on training data only
    goals_scaler.fit(all_goals_train)
    shots_scaler.fit(all_shots_train)

    # Transform all data (including validation and test sets)
    # For home goals
    home_goals_np = home_goals_raw.numpy().reshape(-1, 1)
    home_goals_scaled_np = goals_scaler.transform(home_goals_np)
    home_goals = torch.tensor(home_goals_scaled_np.flatten(), dtype=torch.float)

    # For away goals
    away_goals_np = away_goals_raw.numpy().reshape(-1, 1)
    away_goals_scaled_np = goals_scaler.transform(away_goals_np)
    away_goals = torch.tensor(away_goals_scaled_np.flatten(), dtype=torch.float)

    # For home shots
    home_shots_np = home_shots_raw.numpy().reshape(-1, 1)
    home_shots_scaled_np = shots_scaler.transform(home_shots_np)
    home_shots = torch.tensor(home_shots_scaled_np.flatten(), dtype=torch.float)

    # For away shots
    away_shots_np = away_shots_raw.numpy().reshape(-1, 1)
    away_shots_scaled_np = shots_scaler.transform(away_shots_np)
    away_shots = torch.tensor(away_shots_scaled_np.flatten(), dtype=torch.float)

    # Print scaled statistics
    print(f"Scaled goal statistics (training set):")
    print(
        f"  Home goals: mean={home_goals[train_indices].mean().item():.2f}, std={home_goals[train_indices].std().item():.2f}")
    print(
        f"  Away goals: mean={away_goals[train_indices].mean().item():.2f}, std={away_goals[train_indices].std().item():.2f}")

    print(f"Scaled shot attempt statistics (training set):")
    print(
        f"  Home shot attempts: mean={home_shots[train_indices].mean().item():.2f}, std={home_shots[train_indices].std().item():.2f}")
    print(
        f"  Away shot attempts: mean={away_shots[train_indices].mean().item():.2f}, std={away_shots[train_indices].std().item():.2f}")

    # Calculate correlation between goals and shots for analysis
    home_corr = np.corrcoef(home_goals_raw[train_indices].numpy(), home_shots_raw[train_indices].numpy())[0, 1]
    away_corr = np.corrcoef(away_goals_raw[train_indices].numpy(), away_shots_raw[train_indices].numpy())[0, 1]
    print(f"Correlation between goals and shot attempts - Home: {home_corr:.2f}, Away: {away_corr:.2f}")

    # Store raw data for visualization and evaluation
    raw_data = {
        'home_goals_raw': home_goals_raw,
        'away_goals_raw': away_goals_raw,
        'home_shot_attempts_raw': home_shots_raw,
        'away_shot_attempts_raw': away_shots_raw
    }

    # Create dictionary with all training data
    model_data = {
        'x': x,
        'edge_index': edge_index,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_shot_attempts': home_shots,
        'away_shot_attempts': away_shots,
        'game_mask': game_mask,
        'game_indices': torch.tensor(game_indices, dtype=torch.long),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'train_games': train_games,
        'val_games': val_games,
        'test_games': test_games,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'raw_data': raw_data,
        'stats': {
            'home_goals_mean': home_goals_mean,
            'home_goals_std': home_goals_std,
            'away_goals_mean': away_goals_mean,
            'away_goals_std': away_goals_std,
            'home_shots_mean': home_shots_mean,
            'home_shots_std': home_shots_std,
            'away_shots_mean': away_shots_mean,
            'away_shots_std': away_shots_std,
            'home_corr': home_corr,
            'away_corr': away_corr
        },
        'scalers': {
            'goals_scaler': goals_scaler,
            'shots_scaler': shots_scaler
        }
    }

    print(f"Data prepared with {len(game_indices)} games:")
    print(f"  Training: {len(train_indices)} games")
    print(f"  Validation: {len(val_indices)} games")
    print(f"  Testing: {len(test_indices)} games")

    return model_data


def train_goal_prediction_model(config_model, model, data, epochs=150, lr=0.001, weight_decay=1e-4, patience=15,
                                alpha=0.6, beta=0.3, gamma=0.1, device=None,
                                use_outcome_head=True):
    """
    Train the multi-task prediction model with early stopping.

    Args:
        model: MultiHeadedHockeyGNN model
        data: Dictionary containing training data
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization factor
        patience: Number of epochs to wait for improvement before stopping
        alpha: Weight for goal prediction loss
        beta: Weight for shots on goal prediction loss
        gamma: Weight for game outcome prediction loss
        device: Device to use for training (CPU or CUDA)
        use_outcome_head: Whether to use the dedicated outcome classification head

    Returns:
        Dictionary with training history and best model state
    """
    print(f"Training multi-task prediction model for up to {epochs} epochs (patience={patience})...")
    print(f"Loss weights - Goals: {alpha:.2f}, Shots: {beta:.2f}, Outcome: {gamma:.2f}")

    # Force CPU usage if no device is specified
    if device is None:
        device = torch.device('cpu')
    print(f"Training on device: {device}")

    # Ensure model is on the device
    model = model.to(device)

    # Create optimizer with weight decay
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,  # Higher initial LR for SGD
                                momentum=0.9,
                                weight_decay=weight_decay,
                                nesterov=True)
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 3, verbose=True
    )

    # Ensure data is on the device
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    home_goals = data['home_goals'].to(device)  # These are already scaled by StandardScaler
    away_goals = data['away_goals'].to(device)  # These are already scaled by StandardScaler
    home_shots = data['home_shot_attempts'].to(device)  # These are already scaled by StandardScaler
    away_shots = data['away_shot_attempts'].to(device)  # These are already scaled by StandardScaler

    # Get raw data for outcome computation (since outcome is binary and shouldn't be scaled)
    raw_data = data['raw_data']
    home_goals_raw = raw_data['home_goals_raw'].to(device)
    away_goals_raw = raw_data['away_goals_raw'].to(device)

    # Get train and validation indices
    train_indices = data['train_indices']
    val_indices = data['val_indices']

    # Define multi-task loss function with goal, shots, and outcome components
    def multi_task_loss(config_model, outputs, true_home_goals, true_away_goals, true_home_shots, true_away_shots,
                        raw_home_goals, raw_away_goals, indices):
        # Extract predictions
        pred_home_goals = outputs['home_goals'].squeeze()[indices]
        pred_away_goals = outputs['away_goals'].squeeze()[indices]
        pred_home_shots = outputs['home_shot_attempts'].squeeze()[indices]
        pred_away_shots = outputs['away_shot_attempts'].squeeze()[indices]

        # MSE loss for goal prediction (using scaled values)
        home_goals_mse = F.mse_loss(pred_home_goals, true_home_goals[indices])
        away_goals_mse = F.mse_loss(pred_away_goals, true_away_goals[indices])
        goals_mse = (home_goals_mse + away_goals_mse) / 2.0

        # MSE loss for shots prediction (using scaled values)
        home_shots_mse = F.mse_loss(pred_home_shots, true_home_shots[indices])
        away_shots_mse = F.mse_loss(pred_away_shots, true_away_shots[indices])
        shots_mse = (home_shots_mse + away_shots_mse) / 2.0

        # Game outcome loss - use raw (unscaled) values for outcome determination
        if use_outcome_head and 'outcome_logits' in outputs:
            # Using dedicated outcome head with cross-entropy loss
            # Create target class labels: 0=home win, 1=draw, 2=away win
            home_win = (raw_home_goals[indices] > raw_away_goals[indices])
            away_win = (raw_home_goals[indices] < raw_away_goals[indices])
            draw = ~(home_win | away_win)

            true_outcome = torch.zeros_like(home_win, dtype=torch.long)
            true_outcome[home_win] = 0
            true_outcome[draw] = 1
            true_outcome[away_win] = 2

            # Apply class weights for imbalanced outcomes
            if hasattr(config_model, 'class_weights') and config_model.class_weights is not None:
                class_weights = torch.tensor(config_model.class_weights, device=device)
                outcome_loss = F.cross_entropy(outputs['outcome_logits'][indices], true_outcome, weight=class_weights)
            else:
                # Fall back to unweighted cross-entropy
                outcome_loss = F.cross_entropy(outputs['outcome_logits'][indices], true_outcome)
        else:
            # Legacy approach: binary cross-entropy based on goal difference
            outcome_loss = F.binary_cross_entropy_with_logits(
                (pred_home_goals - pred_away_goals),
                (raw_home_goals[indices] > raw_away_goals[indices]).float()
            )

        # Combine losses with weights
        total_loss = alpha * goals_mse + beta * shots_mse + gamma * outcome_loss

        return total_loss, goals_mse, shots_mse, outcome_loss

    # Training history
    history = {
        'train_loss': [],
        'train_goal_loss': [],
        'train_shots_loss': [],
        'train_outcome_loss': [],
        'train_rmse_home_goals': [],
        'train_rmse_away_goals': [],
        'train_rmse_home_shots': [],
        'train_rmse_away_shots': [],
        'train_outcome_acc': [],
        'val_loss': [],
        'val_goal_loss': [],
        'val_shots_loss': [],
        'val_outcome_loss': [],
        'val_rmse_home_goals': [],
        'val_rmse_away_goals': [],
        'val_rmse_home_shots': [],
        'val_rmse_away_shots': [],
        'val_outcome_acc': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # Access scalers for inverse transformation
    goals_scaler = data['scalers']['goals_scaler']
    shots_scaler = data['scalers']['shots_scaler']

    # Train model
    for epoch in range(epochs):
        # Training mode
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x, edge_index, game_indices)

        # Calculate multi-task loss for training set
        train_loss, train_goal_loss, train_shots_loss, train_outcome_loss = multi_task_loss(config_model,
            outputs, home_goals, away_goals, home_shots, away_shots,
            home_goals_raw, away_goals_raw, train_indices
        )

        # Backward pass
        train_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate training metrics
        pred_home_goals = outputs['home_goals'].squeeze().detach()
        pred_away_goals = outputs['away_goals'].squeeze().detach()
        pred_home_shots = outputs['home_shot_attempts'].squeeze().detach()
        pred_away_shots = outputs['away_shot_attempts'].squeeze().detach()

        # Inverse transform predictions back to original scale for RMSE calculation
        train_pred_home_goals_scaled = pred_home_goals[train_indices].cpu().numpy().reshape(-1, 1)
        train_pred_away_goals_scaled = pred_away_goals[train_indices].cpu().numpy().reshape(-1, 1)
        train_pred_home_shots_scaled = pred_home_shots[train_indices].cpu().numpy().reshape(-1, 1)
        train_pred_away_shots_scaled = pred_away_shots[train_indices].cpu().numpy().reshape(-1, 1)

        # Inverse transform to original scale
        train_pred_home_goals = goals_scaler.inverse_transform(train_pred_home_goals_scaled).flatten()
        train_pred_away_goals = goals_scaler.inverse_transform(train_pred_away_goals_scaled).flatten()
        train_pred_home_shots = shots_scaler.inverse_transform(train_pred_home_shots_scaled).flatten()
        train_pred_away_shots = shots_scaler.inverse_transform(train_pred_away_shots_scaled).flatten()

        train_true_home_goals = home_goals_raw[train_indices].cpu().numpy()
        train_true_away_goals = away_goals_raw[train_indices].cpu().numpy()
        train_true_home_shots = raw_data['home_shot_attempts_raw'][train_indices].cpu().numpy()
        train_true_away_shots = raw_data['away_shot_attempts_raw'][train_indices].cpu().numpy()

        # Calculate RMSE on original scale
        train_rmse_home_goals = np.sqrt(mean_squared_error(train_true_home_goals, train_pred_home_goals))
        train_rmse_away_goals = np.sqrt(mean_squared_error(train_true_away_goals, train_pred_away_goals))
        train_rmse_home_shots = np.sqrt(mean_squared_error(train_true_home_shots, train_pred_home_shots))
        train_rmse_away_shots = np.sqrt(mean_squared_error(train_true_away_shots, train_pred_away_shots))

        # For outcome accuracy, we need to handle three-class prediction properly
        if use_outcome_head and 'outcome_logits' in outputs:
            # Get predicted classes from outcome head
            train_pred_classes = outputs['outcome_logits'][train_indices].argmax(dim=1).cpu().numpy()
            # Create true outcome classes
            train_true_classes = np.zeros_like(train_true_home_goals, dtype=int)
            train_true_classes[(train_true_home_goals > train_true_away_goals)] = 0  # Home win
            train_true_classes[(train_true_home_goals == train_true_away_goals)] = 1  # Draw
            train_true_classes[(train_true_home_goals < train_true_away_goals)] = 2  # Away win
            train_outcome_acc = np.mean(train_pred_classes == train_true_classes)
        else:
            # Legacy binary accuracy
            train_outcome_correct = np.sum((train_pred_home_goals > train_pred_away_goals) ==
                                           (train_true_home_goals > train_true_away_goals))
            train_outcome_acc = train_outcome_correct / len(train_indices)

        # Evaluation mode
        model.eval()
        with torch.no_grad():
            # Forward pass on validation set
            outputs = model(x, edge_index, game_indices)

            # Calculate validation loss
            val_loss, val_goal_loss, val_shots_loss, val_outcome_loss = multi_task_loss(config_model,
                outputs, home_goals, away_goals, home_shots, away_shots,
                home_goals_raw, away_goals_raw, val_indices
            )

            # Calculate validation metrics
            pred_home_goals = outputs['home_goals'].squeeze()
            pred_away_goals = outputs['away_goals'].squeeze()
            pred_home_shots = outputs['home_shot_attempts'].squeeze()
            pred_away_shots = outputs['away_shot_attempts'].squeeze()

            # Inverse transform predictions for validation set
            val_pred_home_goals_scaled = pred_home_goals[val_indices].cpu().numpy().reshape(-1, 1)
            val_pred_away_goals_scaled = pred_away_goals[val_indices].cpu().numpy().reshape(-1, 1)
            val_pred_home_shots_scaled = pred_home_shots[val_indices].cpu().numpy().reshape(-1, 1)
            val_pred_away_shots_scaled = pred_away_shots[val_indices].cpu().numpy().reshape(-1, 1)

            # Inverse transform to original scale
            val_pred_home_goals = goals_scaler.inverse_transform(val_pred_home_goals_scaled).flatten()
            val_pred_away_goals = goals_scaler.inverse_transform(val_pred_away_goals_scaled).flatten()
            val_pred_home_shots = shots_scaler.inverse_transform(val_pred_home_shots_scaled).flatten()
            val_pred_away_shots = shots_scaler.inverse_transform(val_pred_away_shots_scaled).flatten()

            val_true_home_goals = home_goals_raw[val_indices].cpu().numpy()
            val_true_away_goals = away_goals_raw[val_indices].cpu().numpy()
            val_true_home_shots = raw_data['home_shot_attempts_raw'][val_indices].cpu().numpy()
            val_true_away_shots = raw_data['away_shot_attempts_raw'][val_indices].cpu().numpy()

            # Calculate RMSE on original scale for validation set
            val_rmse_home_goals = np.sqrt(mean_squared_error(val_true_home_goals, val_pred_home_goals))
            val_rmse_away_goals = np.sqrt(mean_squared_error(val_true_away_goals, val_pred_away_goals))
            val_rmse_home_shots = np.sqrt(mean_squared_error(val_true_home_shots, val_pred_home_shots))
            val_rmse_away_shots = np.sqrt(mean_squared_error(val_true_away_shots, val_pred_away_shots))

            # Calculate validation outcome accuracy
            if use_outcome_head and 'outcome_logits' in outputs:
                # Get predicted classes from outcome head
                val_pred_classes = outputs['outcome_logits'][val_indices].argmax(dim=1).cpu().numpy()
                # Create true outcome classes
                val_true_classes = np.zeros_like(val_true_home_goals, dtype=int)
                val_true_classes[(val_true_home_goals > val_true_away_goals)] = 0  # Home win
                val_true_classes[(val_true_home_goals == val_true_away_goals)] = 1  # Draw
                val_true_classes[(val_true_home_goals < val_true_away_goals)] = 2  # Away win
                val_outcome_acc = np.mean(val_pred_classes == val_true_classes)
            else:
                # Legacy binary accuracy
                val_outcome_correct = np.sum((val_pred_home_goals > val_pred_away_goals) ==
                                             (val_true_home_goals > val_true_away_goals))
                val_outcome_acc = val_outcome_correct / len(val_indices)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss.item())
        history['train_goal_loss'].append(train_goal_loss.item())
        history['train_shots_loss'].append(train_shots_loss.item())
        history['train_outcome_loss'].append(train_outcome_loss.item())
        history['train_rmse_home_goals'].append(train_rmse_home_goals)
        history['train_rmse_away_goals'].append(train_rmse_away_goals)
        history['train_rmse_home_shots'].append(train_rmse_home_shots)
        history['train_rmse_away_shots'].append(train_rmse_away_shots)
        history['train_outcome_acc'].append(train_outcome_acc)

        history['val_loss'].append(val_loss.item())
        history['val_goal_loss'].append(val_goal_loss.item())
        history['val_shots_loss'].append(val_shots_loss.item())
        history['val_outcome_loss'].append(val_outcome_loss.item())
        history['val_rmse_home_goals'].append(val_rmse_home_goals)
        history['val_rmse_away_goals'].append(val_rmse_away_goals)
        history['val_rmse_home_shots'].append(val_rmse_home_shots)
        history['val_rmse_away_shots'].append(val_rmse_away_shots)
        history['val_outcome_acc'].append(val_outcome_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1 or epochs_no_improve == patience:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train - Loss: {train_loss.item():.4f} (Goals: {train_goal_loss.item():.4f}, "
                  f"Shots: {train_shots_loss.item():.4f}, Outcome: {train_outcome_loss.item():.4f})")
            print(f"  Train - Goals RMSE: Home={train_rmse_home_goals:.2f}, Away={train_rmse_away_goals:.2f}")
            print(f"  Train - Shot Attempts RMSE: Home={train_rmse_home_shots:.2f}, Away={train_rmse_away_shots:.2f}")
            print(f"  Train - Outcome Acc: {train_outcome_acc:.2f}")

            print(f"  Val   - Loss: {val_loss.item():.4f} (Goals: {val_goal_loss.item():.4f}, "
                  f"Shots: {val_shots_loss.item():.4f}, Outcome: {val_outcome_loss.item():.4f})")
            print(f"  Val   - Goals RMSE: Home={val_rmse_home_goals:.2f}, Away={val_rmse_away_goals:.2f}")
            print(f"  Val   - Shot Attempts RMSE: Home={val_rmse_home_shots:.2f}, Away={val_rmse_away_shots:.2f}")
            print(f"  Val   - Outcome Acc: {val_outcome_acc:.2f}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")
    else:
        print("Warning: No best model found - using final model")

    # Return the trained model and history
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'best_model_state': best_model_state
    }


def evaluate_multi_task_model(model, data, device=None):
    """
    Evaluate the multi-task hockey prediction model on test data.
    Includes confusion matrix for three outcome classes.

    Args:
        model: Trained MultiHeadedHockeyGNN model
        data: Dictionary containing test data
        device: Device to use for evaluation

    Returns:
        Dictionary with evaluation metrics for all prediction tasks
    """
    # Force CPU usage if no device is specified
    if device is None:
        device = torch.device('cpu')

    # Ensure model is on the device
    model = model.to(device)

    # Ensure data is on the device
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    home_goals = data['home_goals'].to(device)  # Scaled
    away_goals = data['away_goals'].to(device)  # Scaled
    home_shots = data['home_shot_attempts'].to(device)  # Scaled
    away_shots = data['away_shot_attempts'].to(device)  # Scaled

    # Get raw (unscaled) data for RMSE calculation
    raw_data = data['raw_data']
    home_goals_raw = raw_data['home_goals_raw']
    away_goals_raw = raw_data['away_goals_raw']
    home_shots_raw = raw_data['home_shot_attempts_raw']
    away_shots_raw = raw_data['away_shot_attempts_raw']

    # Get test indices
    test_indices = data['test_indices']

    # Access scalers for inverse transformation
    goals_scaler = data['scalers']['goals_scaler']
    shots_scaler = data['scalers']['shots_scaler']

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(x, edge_index, game_indices)

        # Get scaled predictions
        pred_home_goals_scaled = outputs['home_goals'].squeeze()
        pred_away_goals_scaled = outputs['away_goals'].squeeze()
        pred_home_shots_scaled = outputs['home_shot_attempts'].squeeze()
        pred_away_shots_scaled = outputs['away_shot_attempts'].squeeze()

        # Get test predictions (still scaled)
        test_pred_home_goals_scaled = pred_home_goals_scaled[test_indices].cpu().numpy().reshape(-1, 1)
        test_pred_away_goals_scaled = pred_away_goals_scaled[test_indices].cpu().numpy().reshape(-1, 1)
        test_pred_home_shots_scaled = pred_home_shots_scaled[test_indices].cpu().numpy().reshape(-1, 1)
        test_pred_away_shots_scaled = pred_away_shots_scaled[test_indices].cpu().numpy().reshape(-1, 1)

        # Inverse transform to original scale
        test_pred_home_goals = goals_scaler.inverse_transform(test_pred_home_goals_scaled).flatten()
        test_pred_away_goals = goals_scaler.inverse_transform(test_pred_away_goals_scaled).flatten()
        test_pred_home_shots = shots_scaler.inverse_transform(test_pred_home_shots_scaled).flatten()
        test_pred_away_shots = shots_scaler.inverse_transform(test_pred_away_shots_scaled).flatten()

        # Get true values (already in original scale)
        test_true_home_goals = home_goals_raw[test_indices].cpu().numpy()
        test_true_away_goals = away_goals_raw[test_indices].cpu().numpy()
        test_true_home_shots = home_shots_raw[test_indices].cpu().numpy()
        test_true_away_shots = away_shots_raw[test_indices].cpu().numpy()

        # Calculate regression metrics for goals
        home_goals_rmse = np.sqrt(mean_squared_error(test_true_home_goals, test_pred_home_goals))
        away_goals_rmse = np.sqrt(mean_squared_error(test_true_away_goals, test_pred_away_goals))
        home_goals_mae = mean_absolute_error(test_true_home_goals, test_pred_home_goals)
        away_goals_mae = mean_absolute_error(test_true_away_goals, test_pred_away_goals)

        # Calculate regression metrics for shots
        home_shots_rmse = np.sqrt(mean_squared_error(test_true_home_shots, test_pred_home_shots))
        away_shots_rmse = np.sqrt(mean_squared_error(test_true_away_shots, test_pred_away_shots))
        home_shots_mae = mean_absolute_error(test_true_home_shots, test_pred_home_shots)
        away_shots_mae = mean_absolute_error(test_true_away_shots, test_pred_away_shots)

        # Calculate exact prediction accuracy (rounded to nearest integer)
        home_goals_exact = np.mean(np.round(test_pred_home_goals) == test_true_home_goals)
        away_goals_exact = np.mean(np.round(test_pred_away_goals) == test_true_away_goals)

        # We don't expect exact shot predictions to be as accurate, but we'll calculate them anyway
        home_shots_exact = np.mean(np.round(test_pred_home_shots) == test_true_home_shots)
        away_shots_exact = np.mean(np.round(test_pred_away_shots) == test_true_away_shots)

        # Function to convert goals to outcome classes (0=home win, 1=draw, 2=away win)
        def get_outcome_class(home_goals, away_goals):
            if home_goals > away_goals:
                return 0  # Home win
            elif home_goals == away_goals:
                return 1  # Draw
            else:
                return 2  # Away win

        # If model has dedicated outcome head, use it for predictions
        if hasattr(model, 'with_outcome_head') and model.with_outcome_head and 'outcome_logits' in outputs:
            pred_classes = outputs['outcome_logits'][test_indices].argmax(dim=1).cpu().numpy()
        else:
            # Convert rounded goal predictions to outcome classes
            pred_classes = np.array([get_outcome_class(h, a) for h, a in
                                     zip(np.round(test_pred_home_goals), np.round(test_pred_away_goals))])

        # Convert true goals to outcome classes
        true_classes = np.array([get_outcome_class(h, a) for h, a in
                                 zip(test_true_home_goals, test_true_away_goals)])

        # Create confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)

        # Calculate overall accuracy
        outcome_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        # Get detailed classification report
        class_names = ["Home Win", "Draw", "Away Win"]
        classification_metrics = classification_report(
            true_classes, pred_classes, target_names=class_names, output_dict=True
        )

        # Calculate legacy binary metrics for backward compatibility
        binary_pred_winner = (test_pred_home_goals > test_pred_away_goals).astype(int)
        binary_true_winner = (test_true_home_goals > test_true_away_goals).astype(int)
        binary_outcome_accuracy = np.mean(binary_pred_winner == binary_true_winner)

        # Calculate draw prediction accuracy
        pred_draw = (np.abs(np.round(test_pred_home_goals) - np.round(test_pred_away_goals)) == 0).astype(int)
        true_draw = (test_true_home_goals == test_true_away_goals).astype(int)
        draw_accuracy = np.mean(pred_draw == true_draw)

        # Calculate goal difference accuracy
        pred_goal_diff = np.round(test_pred_home_goals - test_pred_away_goals)
        true_goal_diff = test_true_home_goals - test_true_away_goals
        goal_diff_exact = np.mean(pred_goal_diff == true_goal_diff)
        goal_diff_rmse = np.sqrt(mean_squared_error(true_goal_diff, pred_goal_diff))

        # Calculate over/under prediction accuracy
        thresholds = [4.5, 5.5, 6.5]
        over_under_acc = {}

        for threshold in thresholds:
            # Use sigmoid function centered on the threshold
            pred_over = ((test_pred_home_goals + test_pred_away_goals) > threshold).astype(int)
            true_over = ((test_true_home_goals + test_true_away_goals) > threshold).astype(int)
            over_under_acc[f'over_{threshold}'] = np.mean(pred_over == true_over)

        # Calculate correlation between predicted and actual values
        goals_home_corr = np.corrcoef(test_true_home_goals, test_pred_home_goals)[0, 1]
        goals_away_corr = np.corrcoef(test_true_away_goals, test_pred_away_goals)[0, 1]
        shots_home_corr = np.corrcoef(test_true_home_shots, test_pred_home_shots)[0, 1]
        shots_away_corr = np.corrcoef(test_true_away_shots, test_pred_away_shots)[0, 1]

    # Print evaluation results
    print("\n===== Multi-Task Hockey Prediction Model Evaluation =====")
    print(f"Test set size: {len(test_indices)} games")

    print("\nGoal Prediction Metrics:")
    print(f"  RMSE: Home={home_goals_rmse:.2f}, Away={away_goals_rmse:.2f}")
    print(f"  MAE: Home={home_goals_mae:.2f}, Away={away_goals_mae:.2f}")
    print(f"  Exact accuracy: Home={home_goals_exact:.2f}, Away={away_goals_exact:.2f}")
    print(f"  Correlation: Home={goals_home_corr:.2f}, Away={goals_away_corr:.2f}")

    print("\nShot Attempts Prediction Metrics:")
    print(f"  RMSE: Home={home_shots_rmse:.2f}, Away={away_shots_rmse:.2f}")
    print(f"  MAE: Home={home_shots_mae:.2f}, Away={away_shots_mae:.2f}")
    print(f"  Exact accuracy: Home={home_shots_exact:.2f}, Away={away_shots_exact:.2f}")
    print(f"  Correlation: Home={shots_home_corr:.2f}, Away={shots_away_corr:.2f}")

    print("\nGame Outcome Metrics:")
    print(f"  Three-class outcome accuracy: {outcome_accuracy:.2f}")
    print(f"  Binary winner prediction accuracy: {binary_outcome_accuracy:.2f}")
    print(f"  Draw prediction accuracy: {draw_accuracy:.2f}")
    print(f"  Goal difference exact: {goal_diff_exact:.2f}, RMSE: {goal_diff_rmse:.2f}")

    # Print confusion matrix
    print("\nConfusion Matrix (rows=actual, columns=predicted):")
    print("               Home Win   Draw   Away Win")
    for i, name in enumerate(class_names):
        print(f"{name: <14} {cm[i, 0]: >8}  {cm[i, 1]: >5}  {cm[i, 2]: >8}")

    # Print per-class metrics
    print("\nClass-specific metrics:")
    for name in class_names:
        metrics = classification_metrics[name]
        print(
            f"  {name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")

    print("\nOver/Under prediction accuracy:")
    for threshold, acc in over_under_acc.items():
        print(f"  {threshold}: {acc:.2f}")

    # Return all evaluation metrics
    return {
        'goals': {
            'home_rmse': home_goals_rmse,
            'away_rmse': away_goals_rmse,
            'home_mae': home_goals_mae,
            'away_mae': away_goals_mae,
            'home_exact': home_goals_exact,
            'away_exact': away_goals_exact,
            'home_corr': goals_home_corr,
            'away_corr': goals_away_corr
        },
        'shots': {
            'home_rmse': home_shots_rmse,
            'away_rmse': away_shots_rmse,
            'home_mae': home_shots_mae,
            'away_mae': away_shots_mae,
            'home_exact': home_shots_exact,
            'away_exact': away_shots_exact,
            'home_corr': shots_home_corr,
            'away_corr': shots_away_corr
        },
        'outcomes': {
            'three_class_accuracy': outcome_accuracy,
            'binary_winner_accuracy': binary_outcome_accuracy,
            'draw_accuracy': draw_accuracy,
            'goal_diff_exact': goal_diff_exact,
            'goal_diff_rmse': goal_diff_rmse,
            'over_under_acc': over_under_acc,
            'confusion_matrix': cm,
            'classification_metrics': classification_metrics
        },
        'predictions': {
            'home_goals': test_pred_home_goals,
            'away_goals': test_pred_away_goals,
            'home_shots': test_pred_home_shots,
            'away_shots': test_pred_away_shots,
            'true_home_goals': test_true_home_goals,
            'true_away_goals': test_true_away_goals,
            'true_home_shots': test_true_home_shots,
            'true_away_shots': test_true_away_shots,
            'pred_classes': pred_classes,
            'true_classes': true_classes
        }
    }


def predict_game_shots_and_goals(model, data_graph, home_team, away_team, window_sizes=[5, 10, 20, 40, 82],
                                 scalers=None, device=None):
    """
    Predict the outcome of a game between two teams, including shot attempts.
    Uses scalers to properly scale/inverse scale predictions.

    Args:
        model: Trained MultiHeadedHockeyGNN model
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_sizes: List of historical window sizes to use for features (must match training)
        scalers: Dictionary with goals_scaler and shots_scaler from training
        device: Device to use for prediction

    Returns:
        Dictionary with prediction results
    """
    print(f"\nPredicting outcome for {home_team} (home) vs {away_team} (away)...")

    # Force CPU usage if no device is specified
    if device is None:
        device = torch.device('cpu')

    # Ensure model is on the device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

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
        return {'error': f"No data for home team {home_team}"}

    if not away_tgps:
        print(f"Warning: No TGP data found for away team {away_team}")
        return {'error': f"No data for away team {away_team}"}

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

    # Define base team stats
    team_stats = [
        # Original stats
        'goal_avg', 'goal_against_avg', 'shot_on_goal_avg',
        'shot_saved_avg', 'shot_blocked_avg', 'shot_missed_avg',
        'faceoff_won_avg', 'hit_another_player_avg', 'penalties_duration_avg',

        # Added stats
        'faceoff_taken_avg',  # Total faceoffs
        'shot_attempt_avg',  # Total shot attempts
        'giveaways_avg', 'takeaways_avg',  # Puck possession metrics
        'hit_by_player_avg',  # Physical play received
        'shot_missed_shootout_avg',  # Shootout performance
        'penalties_avg',  # Penalty frequency
        'penalties_drawn_avg',  # Drawing penalties
        'penalty_shot_avg',  # Penalty shots awarded
        'penalty_shot_goal_avg',  # Penalty shot conversion
    ]

    # Extract home team features
    home_features = []

    # Add historical statistics for each window size
    for window_size in window_sizes:
        for stat in team_stats:
            # For home team
            hist_key = f'hist_{window_size}_{stat}'
            if hist_key in home_tgp_data:
                if isinstance(home_tgp_data[hist_key], list):
                    # Use regulation stats (index 0)
                    value = home_tgp_data[hist_key][0]
                else:
                    value = home_tgp_data[hist_key]
            else:
                value = 0.0
            home_features.append(value)

            # For away team (same stats but from away perspective)
            if hist_key in away_tgp_data:
                if isinstance(away_tgp_data[hist_key], list):
                    # Use regulation stats (index 0)
                    value = away_tgp_data[hist_key][0]
                else:
                    value = away_tgp_data[hist_key]
            else:
                value = 0.0
            home_features.append(value)

        # Add historical win rates for each window size
        win_key = f'hist_{window_size}_win_avg'
        if win_key in home_tgp_data:
            if isinstance(home_tgp_data[win_key], list):
                home_win_rate = home_tgp_data[win_key][0]  # Regulation win rate
            else:
                home_win_rate = home_tgp_data[win_key]
        else:
            home_win_rate = 0.5  # Default
        home_features.append(home_win_rate)

        if win_key in away_tgp_data:
            if isinstance(away_tgp_data[win_key], list):
                away_win_rate = away_tgp_data[win_key][0]  # Regulation win rate
            else:
                away_win_rate = away_tgp_data[win_key]
        else:
            away_win_rate = 0.5  # Default
        home_features.append(away_win_rate)

    # Add days since last game (non-window dependent)
    if 'days_since_last_game' in home_tgp_data:
        days_value = min(home_tgp_data['days_since_last_game'], 30) / 30  # Normalize
        home_features.append(days_value)
    else:
        home_features.append(1.0)  # Default (max) days since last game

    if 'days_since_last_game' in away_tgp_data:
        days_value = min(away_tgp_data['days_since_last_game'], 30) / 30  # Normalize
        home_features.append(days_value)
    else:
        home_features.append(1.0)  # Default (max) days since last game

    # Node type indicator
    home_features.append(1.0)  # Home team indicator

    # Extract features for away team - similar to home but with indicators flipped
    away_features = copy.deepcopy(home_features)
    away_features[-1] = 0.0  # Set to away team indicator

    # Game node features are the average of home and away
    game_features = np.mean([np.array(home_features), np.array(away_features)], axis=0)
    game_features[-1] = 0.5  # Game node indicator

    # Verify feature count
    print(f"Feature count: {len(home_features)}")

    # Create tensors
    features = torch.tensor(
        np.array([home_features, away_features, game_features], dtype=np.float32),
        dtype=torch.float
    ).to(device)

    # Create edge indices - connect game node to home and away team nodes
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 0, 2, 1],  # Source nodes
        [2, 0, 2, 1, 0, 2, 1, 2]  # Target nodes
    ], dtype=torch.long).to(device)

    # Game index
    game_indices = torch.tensor([2], dtype=torch.long).to(device)  # Index of the game node

    # Make prediction
    with torch.no_grad():
        outputs = model(features, edge_index, game_indices)

        # Get raw predictions (scaled)
        pred_home_goals_scaled = outputs['home_goals'].item()
        pred_away_goals_scaled = outputs['away_goals'].item()
        pred_home_shots_scaled = outputs['home_shot_attempts'].item()
        pred_away_shots_scaled = outputs['away_shot_attempts'].item()

        # Inverse transform with scalers if available
        if scalers is not None:
            goals_scaler = scalers['goals_scaler']
            shots_scaler = scalers['shots_scaler']

            # Apply inverse transform to get predictions in original scale
            pred_home_goals = goals_scaler.inverse_transform([[pred_home_goals_scaled]])[0][0]
            pred_away_goals = goals_scaler.inverse_transform([[pred_away_goals_scaled]])[0][0]
            pred_home_shots = shots_scaler.inverse_transform([[pred_home_shots_scaled]])[0][0]
            pred_away_shots = shots_scaler.inverse_transform([[pred_away_shots_scaled]])[0][0]
        else:
            # If no scalers are available, use default values based on training data
            # These values should be based on your training data statistics
            home_goals_mean, away_goals_mean = 3.12, 2.83
            home_shots_mean, away_shots_mean = 46.61, 44.91

            # Rescale based on averages (assuming model outputs are small values)
            scale_factor_goals = (home_goals_mean + away_goals_mean) / 2 / 0.15  # Approximate scaling
            scale_factor_shots = (home_shots_mean + away_shots_mean) / 2 / 0.15  # Approximate scaling

            pred_home_goals = pred_home_goals_scaled * scale_factor_goals
            pred_away_goals = pred_away_goals_scaled * scale_factor_goals
            pred_home_shots = pred_home_shots_scaled * scale_factor_shots
            pred_away_shots = pred_away_shots_scaled * scale_factor_shots

        # Calculate outcome probabilities
        if 'outcome_logits' in outputs:
            # If we have dedicated outcome predictions, use those
            outcome_probs = F.softmax(outputs['outcome_logits'], dim=1)[0].cpu().numpy()
            home_win_prob = outcome_probs[0]
            draw_prob = outcome_probs[1]
            away_win_prob = outcome_probs[2]
        else:
            # Otherwise, infer from goal difference
            goal_diff = pred_home_goals - pred_away_goals

            if goal_diff > 0.5:  # Home win
                home_win_prob = 0.5 + (min(goal_diff, 3) / 6)  # Cap at 3 goal difference
                away_win_prob = 1 - home_win_prob
                draw_prob = 0.0
            elif goal_diff < -0.5:  # Away win
                away_win_prob = 0.5 + (min(-goal_diff, 3) / 6)  # Cap at 3 goal difference
                home_win_prob = 1 - away_win_prob
                draw_prob = 0.0
            else:  # Predicted draw
                draw_prob = 0.6
                home_win_prob = 0.2
                away_win_prob = 0.2

        # Calculate total goals over/under probabilities
        total_goals = pred_home_goals + pred_away_goals
        over_under_probs = {}

        for threshold in [4.5, 5.5, 6.5]:
            # Use sigmoid function centered on the threshold
            if total_goals > threshold:
                over_prob = 0.5 + (min(total_goals - threshold, 3) / 6)  # Cap at 3 goals over
            else:
                over_prob = 0.5 - (min(threshold - total_goals, 3) / 6)  # Cap at 3 goals under

            over_under_probs[f'over_{threshold}'] = min(max(over_prob, 0.05), 0.95)  # Keep between 0.05 and 0.95
            over_under_probs[f'under_{threshold}'] = 1.0 - over_under_probs[f'over_{threshold}']

    # Print prediction results
    print(f"\nPrediction Results:")
    print(f"Predicted score: {home_team} {pred_home_goals:.2f} - {pred_away_goals:.2f} {away_team}")
    print(f"Predicted shot attempts: {home_team} {pred_home_shots:.2f} - {pred_away_shots:.2f} {away_team}")
    print(f"Outcome probabilities:")
    print(f"  {home_team} win: {home_win_prob:.2f}")
    print(f"  Draw: {draw_prob:.2f}")
    print(f"  {away_team} win: {away_win_prob:.2f}")

    print(f"Over/Under probabilities:")
    for threshold, prob in over_under_probs.items():
        print(f"  {threshold}: {prob:.2f}")

    # Return prediction results
    return {
        'home_team': home_team,
        'away_team': away_team,
        'pred_home_goals': pred_home_goals,
        'pred_away_goals': pred_away_goals,
        'pred_home_shot_attempts': pred_home_shots,
        'pred_away_shot_attempts': pred_away_shots,
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'over_under_probs': over_under_probs,
        'shots_to_goals_ratio_home': pred_home_shots / pred_home_goals if pred_home_goals > 0 else float('inf'),
        'shots_to_goals_ratio_away': pred_away_shots / pred_away_goals if pred_away_goals > 0 else float('inf')
    }


def run_goal_prediction_gnn_with_scalers(config, config_model):
    """
    Main function to run goal prediction GNN model with StandardScaler.

    Args:
        config: Configuration object with file paths and settings
        config_model: Configuration object with model parameters

    Returns:
        model: Trained GNN model
        results: Dictionary with evaluation metrics and results
    """
    print("====== Starting Goal Prediction GNN Training with StandardScaler ======")

    # Load graph with filtering based on split date
    print(f"Loading graph from {config.file_paths['graph']}")
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    data_graph = load_filtered_graph(config.file_paths["graph"], training_cutoff_date)
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Extract features for goal prediction
    window_size = config.stat_window_sizes  # Use window sizes from config
    features, edge_list, labels_dict, node_mapping, feature_names, diagnostics = extract_goal_prediction_features(
        data_graph, window_sizes=window_size
    )

    # Prepare training/validation/test data with StandardScaler
    model_data = prepare_goal_prediction_data_with_scalers(
        features, edge_list, labels_dict, test_size=0.2, val_size=0.1
    )

    # Create output directory for results
    output_dir = config.file_paths["gnn_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis outputs will be saved to {output_dir}")

    # Force CPU usage
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU operation)")

    # Extract model parameters from config_model
    in_channels = model_data['x'].shape[1]
    hidden_channels = config_model.hidden_channels if hasattr(config_model, 'hidden_channels') else 128
    dropout_rate1 = config_model.dropout_rate1 if hasattr(config_model, 'dropout_rate1') else 0.4
    dropout_rate2 = config_model.dropout_rate2 if hasattr(config_model, 'dropout_rate2') else 0.3
    epochs = config_model.num_epochs if hasattr(config_model, 'num_epochs') else 150
    lr = config_model.learning_rate if hasattr(config_model, 'learning_rate') else 0.001
    weight_decay = config_model.weight_decay if hasattr(config_model, 'weight_decay') else 1e-4
    patience = config_model.patience if hasattr(config_model, 'patience') else 15
    alpha = config_model.alpha if hasattr(config_model, 'alpha') else 0.6
    beta = config_model.beta if hasattr(config_model, 'beta') else 0.1
    gamma = config_model.gamma if hasattr(config_model, 'gamma') else 0.3

    # Create the multi-headed model
    print(f"Creating model with {in_channels} input features and {hidden_channels} hidden channels")
    model = MultiHeadedHockeyGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2,
        with_outcome_head=config_model.use_outcome_head if hasattr(config_model, 'use_outcome_head') else True,
        outcome_head_layers=config_model.outcome_head_layers if hasattr(config_model, 'outcome_head_layers') else None
    ).to(device)

    # Train the model
    print(f"\nTraining model for up to {epochs} epochs (patience={patience})")
    training_result = train_goal_prediction_model(
        config_model=config_model,
        model=model,
        data=model_data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # Evaluate the model
    print("\nEvaluating model on test data")
    evaluation_results = evaluate_multi_task_model(
        model=model,
        data=model_data,
        device=device
    )

    # Create visualizations
    print("\nGenerating visualizations")
    visualize_multi_task_predictions(
        evaluation_results=evaluation_results,
        output_dir=os.path.join(output_dir, 'multi_task_viz')
    )

    # Save model to file along with scalers
    model_path = os.path.join(output_dir, "multi_headed_hockey_gnn_scaled.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'dropout_rate1': dropout_rate1,
            'dropout_rate2': dropout_rate2
        },
        'feature_names': feature_names,
        'stats': model_data['stats'],
        'scalers': model_data['scalers']  # Save the scalers with the model
    }, model_path)
    print(f"Model and scalers saved to {model_path}")

    # Make predictions for sample games
    print("\n====== Making predictions with trained model ======")
    teams_to_predict = [
        ('TOR', 'MTL'),
        ('BOS', 'TBL'),
        ('EDM', 'CGY'),
        ('NYR', 'NYI'),
        ('PIT', 'WSH')
    ]

    predictions = {}
    for home_team, away_team in teams_to_predict:
        try:
            game_prediction = predict_game_shots_and_goals(
                model=model,
                data_graph=data_graph,
                home_team=home_team,
                away_team=away_team,
                window_sizes=window_size,
                scalers=model_data['scalers'],  # Pass the scalers for proper prediction
                device=device
            )
            predictions[f"{home_team}_vs_{away_team}"] = game_prediction
        except Exception as e:
            print(f"\nError predicting {home_team} vs {away_team}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Return the trained model and results
    results = {
        'training_result': training_result,
        'evaluation_results': evaluation_results,
        'predictions': predictions,
        'diagnostics': diagnostics,
        'feature_names': feature_names,
        'scalers': model_data['scalers']  # Include scalers in results
    }

    print("\n====== Multi-Headed Hockey GNN Analysis with StandardScaler Complete ======")

    return model, results


def extract_goal_prediction_features(data_graph, window_sizes):
    """
    Extract features for goal and shots prediction from the graph.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary with goal and shot labels
        node_mapping: Dictionary mapping node IDs to indices in features_list
        feature_names: List of feature names for interpretation
    """
    # Ensure window_sizes is a list
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes]

    print(f"Extracting features for multi-task prediction with window sizes {window_sizes}...")

    # Initialize containers
    features_list = []
    node_mapping = {}
    feature_names = []

    # Initialize labels dictionary with shot attempts instead of shots on goal
    labels_dict = {
        'home_goals': {},
        'away_goals': {},
        'home_shot_attempts': {},
        'away_shot_attempts': {}
    }

    # Track statistics
    diagnostics = {
        'total_games': 0,
        'games_with_goals': 0,
        'games_with_shots': 0,
        'games_with_complete_data': 0,
        'node_types': {'game': 0, 'home_team': 0, 'away_team': 0}
    }

    # Get all game nodes
    game_nodes = [node for node, data in data_graph.nodes(data=True)
                  if data.get('type') == 'game']

    print(f"Found {len(game_nodes)} game nodes in the graph")
    diagnostics['total_games'] = len(game_nodes)

    # Define base team stats
    team_stats = [
        # Original stats
        'goal_avg', 'goal_against_avg', 'shot_on_goal_avg',
        'shot_saved_avg', 'shot_blocked_avg', 'shot_missed_avg',
        'faceoff_won_avg', 'hit_another_player_avg', 'penalties_duration_avg',

        # Added stats
        'faceoff_taken_avg',  # Total faceoffs
        'shot_attempt_avg',  # Total shot attempts
        'giveaways_avg', 'takeaways_avg',  # Puck possession metrics
        'hit_by_player_avg',  # Physical play received
        'shot_missed_shootout_avg',  # Shootout performance
        'penalties_avg',  # Penalty frequency
        'penalties_drawn_avg',  # Drawing penalties
        'penalty_shot_avg',  # Penalty shots awarded
        'penalty_shot_goal_avg',  # Penalty shot conversion
    ]

    # Generate feature names for all window sizes
    for window_size in window_sizes:
        for stat in team_stats:
            feature_names.append(f'home_{window_size}_{stat}')
            feature_names.append(f'away_{window_size}_{stat}')

        # Add win rate for each window size
        feature_names.append(f'home_{window_size}_win_avg')
        feature_names.append(f'away_{window_size}_win_avg')

    # Add other extra features that don't depend on window size
    feature_names.extend([
        'home_days_since_last_game',
        'away_days_since_last_game',
        'node_type_indicator'  # 1.0 for home, 0.0 for away, 0.5 for game
    ])

    print(f'feature count: {len(feature_names)}  feature names: \n{feature_names}')

    # Process each game
    for game_id in game_nodes:
        game_data = data_graph.nodes[game_id]
        if not (game_data.get('home_team') and game_data.get('away_team')):
            continue

        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')

        # Get TGP nodes for this game
        home_tgp = f"{game_id}_{home_team}"
        away_tgp = f"{game_id}_{away_team}"

        if home_tgp in data_graph.nodes and away_tgp in data_graph.nodes:
            home_tgp_data = data_graph.nodes[home_tgp]
            away_tgp_data = data_graph.nodes[away_tgp]

            # Extract features for home team
            home_features = []

            # Add historical statistics for each window size
            for window_size in window_sizes:
                for stat in team_stats:
                    hist_key = f'hist_{window_size}_{stat}'
                    # For home team
                    if hist_key in home_tgp_data:
                        if isinstance(home_tgp_data[hist_key], list):
                            # Use regulation stats (index 0)
                            value = home_tgp_data[hist_key][0]
                        else:
                            value = home_tgp_data[hist_key]
                    else:
                        value = 0.0
                    home_features.append(value)

                    # For away team (same stats but from away perspective)
                    if hist_key in away_tgp_data:
                        if isinstance(away_tgp_data[hist_key], list):
                            # Use regulation stats (index 0)
                            value = away_tgp_data[hist_key][0]
                        else:
                            value = away_tgp_data[hist_key]
                    else:
                        value = 0.0
                    home_features.append(value)

                # Add historical win rates for each window size
                win_key = f'hist_{window_size}_win_avg'
                if win_key in home_tgp_data:
                    if isinstance(home_tgp_data[win_key], list):
                        home_win_rate = home_tgp_data[win_key][0]  # Regulation win rate
                    else:
                        home_win_rate = home_tgp_data[win_key]
                else:
                    home_win_rate = 0.5  # Default
                home_features.append(home_win_rate)

                if win_key in away_tgp_data:
                    if isinstance(away_tgp_data[win_key], list):
                        away_win_rate = away_tgp_data[win_key][0]  # Regulation win rate
                    else:
                        away_win_rate = away_tgp_data[win_key]
                else:
                    away_win_rate = 0.5  # Default
                home_features.append(away_win_rate)

            # Add days since last game (non-window dependent)
            if 'days_since_last_game' in home_tgp_data:
                days_value = min(home_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                home_features.append(days_value)
            else:
                home_features.append(1.0)  # Default (max) days since last game

            if 'days_since_last_game' in away_tgp_data:
                days_value = min(away_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                home_features.append(days_value)
            else:
                home_features.append(1.0)  # Default (max) days since last game

            # Home team indicator
            home_features.append(1.0)  # Home team indicator

            # Extract features for away team - similar to home but with indicators flipped
            away_features = copy.deepcopy(home_features)
            away_features[-1] = 0.0  # Set to away team indicator

            # Game node features are the average of home and away
            game_features = np.mean([np.array(home_features), np.array(away_features)], axis=0)
            game_features[-1] = 0.5  # Game node indicator

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
            features_list.append(game_features.astype(np.float32))
            diagnostics['node_types']['game'] += 1

            # Create goal and shot labels
            has_goal_data = False
            has_shot_data = False

            if 'goal' in home_tgp_data and 'goal' in away_tgp_data:
                # Get the total goals (sum across all periods)
                home_goals = home_tgp_data['goal'][0]
                away_goals = away_tgp_data['goal'][0]

                # Store goal counts as regression targets
                labels_dict['home_goals'][game_idx] = home_goals
                labels_dict['away_goals'][game_idx] = away_goals
                has_goal_data = True
                diagnostics['games_with_goals'] += 1

            # Change from 'shot_on_goal' to 'shot_attempt'
            if 'shot_attempt' in home_tgp_data and 'shot_attempt' in away_tgp_data:
                # Get the total shot attempts (sum across all periods)
                home_shots = home_tgp_data['shot_attempt'][0]
                away_shots = away_tgp_data['shot_attempt'][0]

                # Store shot attempt counts as regression targets
                labels_dict['home_shot_attempts'][game_idx] = home_shots
                labels_dict['away_shot_attempts'][game_idx] = away_shots
                has_shot_data = True
                diagnostics['games_with_shots'] += 1

            if has_goal_data and has_shot_data:
                diagnostics['games_with_complete_data'] += 1

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

    print(f"Extracted features for {len(features_list)} nodes")
    print(f"Created {len(edge_list)} edges")
    print(f"Found {diagnostics['games_with_goals']} games with goal data")
    print(f"Found {diagnostics['games_with_shots']} games with shots data")
    print(f"Found {diagnostics['games_with_complete_data']} games with complete data")

    # Calculate feature count for validation
    expected_feature_count = len(team_stats) * 2 * len(window_sizes) + 2 * len(window_sizes) + 3
    actual_feature_count = len(feature_names)

    print(f"Feature count: {actual_feature_count} (Expected: {expected_feature_count})")
    if actual_feature_count != expected_feature_count:
        print("WARNING: Feature count mismatch. Check feature generation logic.")

    # Normalize features to improve model training
    # features_list = normalize_features(features_list) [removed]

    return features_list, edge_list, labels_dict, node_mapping, feature_names, diagnostics


def visualize_multi_task_predictions(evaluation_results, output_dir):
    """
    Create visualizations for multi-task prediction evaluation.

    Args:
        evaluation_results: Results from evaluate_multi_task_model
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get prediction data
    pred_home_goals = evaluation_results['predictions']['home_goals']
    pred_away_goals = evaluation_results['predictions']['away_goals']
    true_home_goals = evaluation_results['predictions']['true_home_goals']
    true_away_goals = evaluation_results['predictions']['true_away_goals']
    pred_home_shots = evaluation_results['predictions']['home_shots']
    pred_away_shots = evaluation_results['predictions']['away_shots']
    true_home_shots = evaluation_results['predictions']['true_home_shots']
    true_away_shots = evaluation_results['predictions']['true_away_shots']

    # Create DataFrame for easier plotting
    df_goals = pd.DataFrame({
        'Predicted Home Goals': pred_home_goals,
        'Actual Home Goals': true_home_goals,
        'Predicted Away Goals': pred_away_goals,
        'Actual Away Goals': true_away_goals,
        'Predicted Goal Diff': pred_home_goals - pred_away_goals,
        'Actual Goal Diff': true_home_goals - true_away_goals,
        'Predicted Total Goals': pred_home_goals + pred_away_goals,
        'Actual Total Goals': true_home_goals + true_away_goals
    })

    df_shots = pd.DataFrame({
        'Predicted Home Shots': pred_home_shots,
        'Actual Home Shots': true_home_shots,
        'Predicted Away Shots': pred_away_shots,
        'Actual Away Shots': true_away_shots,
        'Predicted Shot Diff': pred_home_shots - pred_away_shots,
        'Actual Shot Diff': true_home_shots - true_away_shots,
        'Predicted Total Shots': pred_home_shots + pred_away_shots,
        'Actual Total Shots': true_home_shots + true_away_shots
    })

    # Get outcome prediction data
    pred_classes = evaluation_results['predictions']['pred_classes']
    true_classes = evaluation_results['predictions']['true_classes']

    # Add outcome information to dataframe
    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    df_goals['True Outcome'] = [outcome_map[c] for c in true_classes]
    df_goals['Predicted Outcome'] = [outcome_map[c] for c in pred_classes]
    df_goals['Correct Outcome'] = pred_classes == true_classes

    # Create visualizations for goals
    # 1. Scatter plot of predicted vs actual goals
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Actual Home Goals', y='Predicted Home Goals', data=df_goals, alpha=0.6, label='Home')
    sns.scatterplot(x='Actual Away Goals', y='Predicted Away Goals', data=df_goals, alpha=0.6, label='Away')

    # Add reference line
    max_val = max(df_goals['Actual Home Goals'].max(), df_goals['Actual Away Goals'].max(),
                  df_goals['Predicted Home Goals'].max(), df_goals['Predicted Away Goals'].max()) + 0.5
    plt.plot([0, max_val], [0, max_val], 'k--')

    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    plt.grid(True, alpha=0.3)
    plt.title('Predicted vs Actual Goals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual_goals.png'), dpi=300)
    plt.close()

    # 2. Create similar visualization for shots
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Actual Home Shots', y='Predicted Home Shots', data=df_shots, alpha=0.6, label='Home')
    sns.scatterplot(x='Actual Away Shots', y='Predicted Away Shots', data=df_shots, alpha=0.6, label='Away')

    # Add reference line
    max_val = max(df_shots['Actual Home Shots'].max(), df_shots['Actual Away Shots'].max(),
                  df_shots['Predicted Home Shots'].max(), df_shots['Predicted Away Shots'].max()) + 0.5
    plt.plot([0, max_val], [0, max_val], 'k--')

    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    plt.grid(True, alpha=0.3)
    plt.title('Predicted vs Actual Shots on Goal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual_shots.png'), dpi=300)
    plt.close()

    # 3. Create histogram of prediction errors for goals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    home_goal_errors = pred_home_goals - true_home_goals
    sns.histplot(home_goal_errors, kde=True, color='blue')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title(f'Home Goals Error (RMSE: {evaluation_results["goals"]["home_rmse"]:.2f})')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    away_goal_errors = pred_away_goals - true_away_goals
    sns.histplot(away_goal_errors, kde=True, color='red')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title(f'Away Goals Error (RMSE: {evaluation_results["goals"]["away_rmse"]:.2f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goal_prediction_errors.png'), dpi=300)
    plt.close()

    # 4. Create histogram of prediction errors for shots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    home_shot_errors = pred_home_shots - true_home_shots
    sns.histplot(home_shot_errors, kde=True, color='blue')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title(f'Home Shots Error (RMSE: {evaluation_results["shots"]["home_rmse"]:.2f})')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    away_shot_errors = pred_away_shots - true_away_shots
    sns.histplot(away_shot_errors, kde=True, color='red')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title(f'Away Shots Error (RMSE: {evaluation_results["shots"]["away_rmse"]:.2f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shot_prediction_errors.png'), dpi=300)
    plt.close()

    # 5. Create confusion matrix visualization
    if 'confusion_matrix' in evaluation_results['outcomes']:
        cm = evaluation_results['outcomes']['confusion_matrix']
        class_names = ["Home Win", "Draw", "Away Win"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Outcome')
        plt.ylabel('Actual Outcome')
        plt.title('Confusion Matrix for Game Outcome Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outcome_confusion_matrix.png'), dpi=300)
        plt.close()

        # Also create percentage-based confusion matrix
        plt.figure(figsize=(10, 8))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Outcome')
        plt.ylabel('Actual Outcome')
        plt.title('Confusion Matrix (% of Actual)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outcome_confusion_matrix_percent.png'), dpi=300)
        plt.close()

    # 6. Correlation between goals and shots
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='Actual Home Goals', y='Actual Home Shots', data=pd.DataFrame({
        'Actual Home Goals': true_home_goals,
        'Actual Home Shots': true_home_shots
    }), alpha=0.6)
    plt.title('Actual Home Goals vs Shots')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    sns.scatterplot(x='Actual Away Goals', y='Actual Away Shots', data=pd.DataFrame({
        'Actual Away Goals': true_away_goals,
        'Actual Away Shots': true_away_shots
    }), alpha=0.6)
    plt.title('Actual Away Goals vs Shots')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    sns.scatterplot(x='Predicted Home Goals', y='Predicted Home Shots', data=pd.DataFrame({
        'Predicted Home Goals': pred_home_goals,
        'Predicted Home Shots': pred_home_shots
    }), alpha=0.6)
    plt.title('Predicted Home Goals vs Shots')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    sns.scatterplot(x='Predicted Away Goals', y='Predicted Away Shots', data=pd.DataFrame({
        'Predicted Away Goals': pred_away_goals,
        'Predicted Away Shots': pred_away_shots
    }), alpha=0.6)
    plt.title('Predicted Away Goals vs Shots')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goals_shots_correlation.png'), dpi=300)
    plt.close()

    # 7. Plot prediction metrics by outcome class
    plt.figure(figsize=(12, 6))
    metrics = evaluation_results['outcomes']['classification_metrics']

    class_metrics = {
        'Precision': [metrics[name]['precision'] for name in class_names],
        'Recall': [metrics[name]['recall'] for name in class_names],
        'F1': [metrics[name]['f1-score'] for name in class_names]
    }

    df_metrics = pd.DataFrame(class_metrics, index=class_names)
    df_metrics.plot(kind='bar', figsize=(10, 6))
    plt.title('Outcome Prediction Metrics by Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outcome_metrics_by_class.png'), dpi=300)
    plt.close()

    print(f"Multi-task visualizations saved to {output_dir}")