import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from src_code.utils.save_graph_utils import load_graph


def run_gnn(config, config_model):
    print("====== Starting GNN Training ======")
    print(f"Loading graph from {config.file_paths['graph']}")
    data_graph = load_graph(config.file_paths["graph"])
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Train and evaluate the GNN
    print(f"Training GNN with {config_model.num_epochs} epochs and window size 5")
    model, accuracy, f1, losses = train_and_evaluate_gnn(data_graph, epochs=config_model.num_epochs, window_size=5)

    # Make a prediction for a new game
    print("\n====== Making predictions ======")
    home_win_probability = predict_game_outcome(model, data_graph, 'TOR', 'MTL')
    print(f'Probability of Toronto winning at home against Montreal: {home_win_probability:.4f}')

    home_win_probability = predict_game_outcome(model, data_graph, 'BOS', 'TBL')
    print(f'Probability of Boston winning at home against Tampa Bay: {home_win_probability:.4f}')


def extract_features_from_graph(data_graph, window_size=5):
    """
    Extract features from the graph for GNN input.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_size: Historical window size to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to outcome labels (1 for home win, 0 for away win)
        node_mapping: Dictionary mapping node IDs to their indices in features_list
    """
    print(f"Extracting features with window size {window_size}...")

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

            # Extract home team features
            home_features = []
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

            # Extract away team features (same as home but with away indicator)
            away_features = []
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
            game_features = np.zeros(5, dtype=np.float32)
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


class SimpleHockeyGNN(nn.Module):
    """
    Simple GNN model for hockey game prediction.
    Uses CPU instead of GPU for better stability.
    """

    def __init__(self, in_channels, hidden_channels=64):
        super(SimpleHockeyGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 2)  # Binary classification (home win or away win)

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN.

        Args:
            x: Node features
            edge_index: Edge list
            game_indices: Indices of game nodes

        Returns:
            logits: Classification logits for game nodes
        """
        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Select only game nodes for prediction
        x_games = x[game_indices]

        # Apply final classification layer
        logits = self.fc(x_games)

        return F.log_softmax(logits, dim=1)


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


def evaluate_model(model, data):
    """
    Evaluate the GNN model.

    Args:
        model: GNN model
        data: Dictionary containing evaluation data

    Returns:
        metrics: Dictionary with accuracy and F1 score
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

            # Get predicted classes
            _, predicted = test_pred.max(1)

            # Calculate metrics
            correct = predicted.eq(test_labels).sum().item()
            total = len(test_labels)
            accuracy = correct / total

            # Calculate F1 score
            f1 = f1_score(test_labels.cpu().numpy(), predicted.cpu().numpy())

            return {
                'accuracy': accuracy,
                'f1': f1,
                'correct': correct,
                'total': total
            }
        else:
            print("Warning: No test data available for evaluation.")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'correct': 0,
                'total': 0
            }


def train_and_evaluate_gnn(data_graph, epochs=100, hidden_channels=64, window_size=5, lr=0.01):
    """
    Train and evaluate the GNN model.

    Args:
        data_graph: NetworkX graph containing hockey data
        epochs: Number of training epochs
        hidden_channels: Number of hidden channels in the GNN
        window_size: Historical window size to use for features
        lr: Learning rate for optimizer

    Returns:
        model: Trained GNN model
        accuracy: Test accuracy
        f1: F1 score
        train_losses: List of training losses
    """
    print("\n====== Starting GNN Training Process ======")

    # Extract features and prepare data
    features, edge_list, labels, node_mapping = extract_features_from_graph(data_graph, window_size)
    model_data = prepare_train_test_data(features, edge_list, labels)

    # Get input dimension from features
    in_channels = model_data['x'].shape[1]

    # Create model (use CPU for better diagnostics)
    print(f"Creating model with {in_channels} input features and {hidden_channels} hidden channels...")
    model = SimpleHockeyGNN(in_channels, hidden_channels)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    train_losses = []

    for epoch in range(epochs):
        loss = train_one_epoch(model, model_data, optimizer)
        train_losses.append(loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, model_data)
            print(f'Epoch: {epoch}/{epochs}, Loss: {loss:.4f}, '
                  f'Accuracy: {metrics["accuracy"]:.4f}, F1: {metrics["f1"]:.4f} '
                  f'({metrics["correct"]}/{metrics["total"]} correct)')

    # Final evaluation
    metrics = evaluate_model(model, model_data)
    print(f'\nFinal results - Accuracy: {metrics["accuracy"]:.4f}, '
          f'F1: {metrics["f1"]:.4f} ({metrics["correct"]}/{metrics["total"]} correct)')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    print(f"Training loss curve saved to training_loss.png")

    return model, metrics["accuracy"], metrics["f1"], train_losses


def predict_game_outcome(model, data_graph, home_team, away_team, window_size=5):
    """
    Predict the outcome of a game between two teams.

    Args:
        model: Trained GNN model
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_size: Historical window size to use for features

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

    # Extract home team features
    if home_tgps:
        home_tgp_data = data_graph.nodes[home_tgps[0]]

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
        home_features = [0.5, 0, 0]

    # Add days since last game (default to 3)
    home_features.append(3 / 30)

    # Home advantage indicator
    home_features.append(1.0)  # Home team

    # Extract away team features
    if away_tgps:
        away_tgp_data = data_graph.nodes[away_tgps[0]]

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
        away_features = [0.5, 0, 0]

    # Add days since last game (default to 3)
    away_features.append(3 / 30)

    # Away team indicator
    away_features.append(0.0)  # Away team

    # Create zero features for game node
    game_features = np.zeros(5, dtype=np.float32)
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