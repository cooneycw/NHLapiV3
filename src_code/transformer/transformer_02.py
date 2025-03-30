import os
import math
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src_code.utils.save_embeddings_utils import load_hockey_embeddings
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def run_transformer_model(config, config_transformer):
    """
    Run the transformer-based modeling pipeline.

    This function:
    1. Loads the embeddings
    2. Prepares the dataset
    3. Creates and trains the transformer model
    4. Evaluates model performance

    Args:
        config: Config object with paths and settings
        config_transformer: ConfigModel object with model settings
    """
    print("Starting transformer model training...")

    # Load embeddings
    embeddings = load_hockey_embeddings(config, format='both')

    # Create dataset
    dataset = HockeyDataset(
        embeddings,
        target='outcome',
        sequence_length=5,
        feature_set='teams'
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Get feature dimension from the dataset
    sample_features, _ = dataset[0]
    # Get feature dimension from the dataset
    sample_features, _ = dataset[0]
    print(f"DEBUG - Sample features shape: {sample_features.shape}")

    # Fix the input dimension extraction
    if isinstance(sample_features.shape[1], tuple):
        # If shape[1] is a tuple, take its first element
        input_dim = sample_features.shape[1][0]
    else:
        # Normal case - shape[1] is a single integer
        input_dim = int(sample_features.shape[1])

    print(f"Using input dimension: {input_dim}")
    # Create model
    model = HockeyTransformer(
        input_dim=input_dim,
        hidden_dim=config_transformer.hidden_dim,
        num_layers=config_transformer.num_layers,
        num_heads=config_transformer.num_heads,
        dropout=config_transformer.dropout
    )

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config_transformer.lr)

    # Training loop
    num_epochs = config_transformer.num_epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, targets in train_loader:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            train_loss += loss.item() * features.size(0)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == targets.unsqueeze(1)).sum().item()
            train_total += targets.size(0)

        # Calculate epoch statistics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, targets in val_loader:
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets.unsqueeze(1))

                # Track statistics
                val_loss += loss.item() * features.size(0)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == targets.unsqueeze(1)).sum().item()
                val_total += targets.size(0)

        # Calculate epoch statistics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # Save the model
    model_path = config.file_paths['transformer_model'] + "hockey_transformer.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final evaluation
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for features, targets in val_loader:
            outputs = model(features)

            # Fix tensor handling to account for different batch sizes
            if outputs.dim() == 0:  # It's a scalar tensor
                all_outputs.append(outputs.item())  # Use append for single values
            else:
                # For normal batches, ensure we only squeeze the second dimension
                all_outputs.extend(outputs.squeeze(1).tolist())

            all_targets.extend(targets.tolist())

    # Calculate metrics
    threshold = 0.5
    predictions = [1 if x > threshold else 0 for x in all_outputs]

    auc = roc_auc_score(all_targets, all_outputs)
    precision = precision_score(all_targets, predictions)
    recall = recall_score(all_targets, predictions)
    f1 = f1_score(all_targets, predictions)

    print("\nFinal Evaluation Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'model': model,
        'metrics': {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }


class HockeyTransformer(nn.Module):
    """
    Transformer model for hockey game prediction.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        """
        Initialize the transformer model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of transformer hidden layer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(HockeyTransformer, self).__init__()

        # Ensure dimensions are integers
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        num_heads = int(num_heads)

        print(f"Initializing HockeyTransformer with input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
              f"num_heads={num_heads}, dropout={dropout}")

        # Add positional encoding for sequence information
        self.pos_encoder = PositionalEncoding(input_dim, dropout)

        # Transformer encoder layer
        encoder_layers = []
        for _ in range(num_layers):
            try:
                layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                )
                encoder_layers.append(layer)
            except Exception as e:
                print(f"Error creating transformer layer: {e}")
                print(f"Types: input_dim={type(input_dim)}, num_heads={type(num_heads)}")
                raise

        # Transformer encoder
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Apply positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder layers sequentially
        for layer in self.transformer_encoder:
            x = layer(x)

        # Use the last sequence element for prediction
        x = x[:, -1, :]

        # Pass through output layer
        x = self.output_layer(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (persistent state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class HockeyDataset(Dataset):
    """
    Dataset class for hockey game predictions.

    This creates a PyTorch dataset from the hockey embeddings to use with DataLoader.
    """

    def __init__(self, embeddings, target='outcome', sequence_length=5, feature_set='teams'):
        """
        Initialize the dataset.

        Args:
            embeddings: Dictionary containing the hockey embeddings
            target: Target variable to predict ('outcome', 'goals', etc.)
            sequence_length: Number of previous games to include in each sequence
            feature_set: Which feature set to use ('teams', 'players', 'pairs', 'all')
        """
        self.embeddings = embeddings
        self.target = target
        self.sequence_length = sequence_length
        self.feature_set = feature_set

        # Process the data based on whether we have JSON or CSV format
        if 'json' in embeddings:
            self._prepare_from_json()
        elif 'csv' in embeddings:
            self._prepare_from_csv()
        else:
            raise ValueError("No valid embeddings found in the provided data")

    def _prepare_from_json(self):
        """Prepare dataset from JSON format embeddings"""
        json_data = self.embeddings['json']

        # Get sequences for each team
        self.sequences = []
        for team, team_sequences in json_data['team_sequences'].items():
            for sequence in team_sequences:
                if sequence['sequence_length'] >= self.sequence_length:
                    # Create a new entry for this sequence
                    seq_entry = {
                        'team_id': team,
                        'sequence_id': f"{team}_{sequence['end_game_id']}",
                        'games': sequence['games'][-self.sequence_length:],
                        'target_game': sequence['games'][-1]
                    }

                    # Set the target value
                    if self.target == 'outcome':
                        outcome = seq_entry['target_game']['outcome']
                        if seq_entry['target_game']['is_home']:
                            seq_entry['target'] = 1 if outcome == 'home_win' else 0
                        else:
                            seq_entry['target'] = 1 if outcome == 'away_win' else 0

                    self.sequences.append(seq_entry)

        print(f"Prepared {len(self.sequences)} sequences from JSON data")

    def _prepare_from_csv(self):
        """Prepare dataset from CSV format embeddings"""
        csv_data = self.embeddings['csv']

        # Get sequence data
        if 'sequences' not in csv_data:
            raise ValueError("Sequence data not found in CSV embeddings")

        seq_df = csv_data['sequences']
        team_df = csv_data['teams'] if 'teams' in csv_data else None

        # Group by sequence_id
        self.sequences = []
        for seq_id, group in seq_df.groupby('sequence_id'):
            # Sort by sequence position
            group = group.sort_values('sequence_position')

            # Only use sequences that have at least sequence_length games
            if len(group) >= self.sequence_length:
                # Get the last sequence_length games
                seq_games = group.tail(self.sequence_length).reset_index(drop=True)

                # Create a sequence entry
                seq_entry = {
                    'team_id': seq_games.iloc[0]['team_id'],
                    'sequence_id': seq_id,
                    'game_ids': seq_games['game_id'].tolist(),
                    'target_game_id': seq_games.iloc[-1]['game_id']
                }

                # Set the target value
                if self.target == 'outcome':
                    last_game = seq_games.iloc[-1]
                    is_home = last_game['is_home']
                    outcome = last_game['game_outcome']

                    if is_home:
                        seq_entry['target'] = 1 if outcome == 'home_win' else 0
                    else:
                        seq_entry['target'] = 1 if outcome == 'away_win' else 0

                # Add features if requested
                if self.feature_set in ['teams', 'all'] and team_df is not None:
                    # Get team features for each game in the sequence
                    game_team_features = []
                    for i, game_id in enumerate(seq_entry['game_ids']):
                        team_id = seq_entry['team_id']
                        game_key = f"{game_id}_{team_id}"

                        # Find this game in the team dataframe
                        team_row = team_df[
                            (team_df['game_id'] == game_id) &
                            (team_df['team_id'] == team_id)
                            ]

                        if len(team_row) > 0:
                            # Extract numeric features
                            features = team_row.select_dtypes(include=[np.number]).iloc[0].to_dict()
                            game_team_features.append(features)
                        else:
                            # Use zeros if game not found
                            print(f"Warning: Game {game_key} not found in team data")
                            game_team_features.append({})

                    seq_entry['team_features'] = game_team_features

                self.sequences.append(seq_entry)

        print(f"Prepared {len(self.sequences)} sequences from CSV data")

    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sequence and its target"""
        sequence = self.sequences[idx]

        # For JSON format
        if 'json' in self.embeddings:
            # Extract features based on feature_set
            features = []
            for game in sequence['games']:
                game_features = {}

                # Get team features
                if self.feature_set in ['teams', 'all']:
                    team_key = game['team_key']
                    team_data = self.embeddings['json']['teams'][team_key]

                    # Add standard team features
                    for key, value in team_data.items():
                        if key not in ['team_id', 'game_id', 'game_date']:
                            game_features[f'team_{key}'] = value

                # Convert to tensor
                feature_tensor = torch.tensor(list(game_features.values()), dtype=torch.float32)
                features.append(feature_tensor)

            # Stack features across time dimension
            features_tensor = torch.stack(features)

            # Return features and target
            return features_tensor, torch.tensor(sequence['target'], dtype=torch.float32)

        # For CSV format
        elif 'csv' in self.embeddings:
            if 'team_features' in sequence:
                # Convert dict of features to tensor for each game
                features = []
                for game_features in sequence['team_features']:
                    # Get feature values, ensuring consistent order
                    feature_values = []
                    for key in sorted(game_features.keys()):
                        if key not in ['game_id', 'team_id']:
                            feature_values.append(float(game_features[key]))

                    # Add zeros if no features found
                    if not feature_values:
                        # Use a default size for features
                        feature_values = [0.0] * 10

                    feature_tensor = torch.tensor(feature_values, dtype=torch.float32)
                    features.append(feature_tensor)

                # Stack features across time dimension
                features_tensor = torch.stack(features)

                # Return features and target
                return features_tensor, torch.tensor(sequence['target'], dtype=torch.float32)
            else:
                # If no features available, return dummy tensor
                return torch.zeros((self.sequence_length, 10)), torch.tensor(sequence['target'], dtype=torch.float32)
