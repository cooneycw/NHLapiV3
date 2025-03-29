import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap


class RegulationHockeyGNN(nn.Module):
    """
    Optimized GNN model for regulation win prediction with improved architecture.
    """

    def __init__(self, in_channels, hidden_channels=128, dropout_rate1=0.3, dropout_rate2=0.4, dropout_rate3=0.5):
        super(RegulationHockeyGNN, self).__init__()

        # Shared layer structure
        # Input embedding layer
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.layer_norm3 = nn.LayerNorm(hidden_channels)

        # GCN layers with residual connections
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Dropout rates - varied across layers
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3

        # Single task output layer - regulation win prediction with deeper architecture
        self.regulation_win_fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate3),
            nn.Linear(hidden_channels // 4, 2)
        )

    def forward(self, x, edge_index, game_indices):
        """
        Forward pass of the GNN with residual connections and layer normalization.
        """
        # Initial embedding
        h = self.input_linear(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate1, training=self.training)

        # First GCN layer with residual connection and layer normalization
        h1 = self.conv1(h, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate1, training=self.training)
        h = h + h1  # Residual connection
        h = self.layer_norm1(h)  # Layer normalization for stability

        # Second GCN layer with residual connection and layer normalization
        h2 = self.conv2(h, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout_rate2, training=self.training)
        h = h + h2  # Residual connection
        h = self.layer_norm2(h)  # Layer normalization for stability

        # Third GCN layer with residual connection and layer normalization
        h3 = self.conv3(h, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout_rate3, training=self.training)
        h = h + h3  # Residual connection
        h = self.layer_norm3(h)  # Layer normalization for stability

        # Select only game nodes for prediction
        x_games = h[game_indices]

        # Regulation win prediction only
        regulation_win_logits = self.regulation_win_fc(x_games)
        regulation_win_probs = F.log_softmax(regulation_win_logits, dim=1)

        return regulation_win_probs


def focal_loss(pred, target, gamma=1.5, alpha=None, reduction='mean'):
    """
    Modified focal loss implementation with lower gamma for more stable training.
    """
    # Apply softmax to get probabilities
    ce_loss = F.cross_entropy(pred, target, weight=alpha, reduction='none')

    # Calculate focal term with reduced gamma for more stability
    p_t = torch.exp(-ce_loss)
    loss = (1 - p_t) ** gamma * ce_loss

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def train_regulation_model(model, data, epochs=200, lr=0.003, weight_decay=1e-4,
                           patience=25, min_delta=0.005, batch_size=64, device=None):
    """
    Enhanced training function with improved stability measures.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

    print(f"Training regulation win model for up to {epochs} epochs (patience={patience})...")

    # Force CPU usage regardless of what's passed in
    device = torch.device('cpu')
    print(f"Training on device: {device} (forced CPU operation)")

    # Ensure model is on CPU
    model = model.to(device)

    # Create optimizer with AdamW instead of Adam for better weight decay handling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use ReduceLROnPlateau instead of CosineAnnealing for more stable learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=7, verbose=True, min_lr=1e-6
    )

    # Ensure data is on CPU
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    y = data['y'].to(device)

    # Get train and test indices
    train_indices = data['train_indices']
    test_indices = data['test_indices']

    # More balanced class weights for focal loss
    class_weights = torch.tensor([1.05, 1.0], device=device)

    # Calculate number of batches
    num_batches = (len(train_indices) + batch_size - 1) // batch_size

    # Training history
    history = {
        'train_loss': [],
        'train_metrics': {'accuracy': [], 'f1': [], 'roc_auc': [], 'precision': [], 'recall': []},
        'val_metrics': {'accuracy': [], 'f1': [], 'roc_auc': [], 'precision': [], 'recall': []}
    }

    # Early stopping variables
    best_f1 = 0
    best_model_state = None
    epochs_no_improve = 0

    # For tracking improvement trends
    validation_f1_history = []

    # Train model
    for epoch in range(epochs):
        # Training mode
        model.train()
        epoch_loss = 0.0

        # Shuffle training indices
        train_shuffle = torch.randperm(len(train_indices))

        # Process in mini-batches
        for i in range(num_batches):
            optimizer.zero_grad()

            # Get batch indices
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(train_indices))
            batch_idx = [train_indices[train_shuffle[j].item()] for j in range(batch_start, batch_end)]
            batch_tensor = torch.tensor(batch_idx, dtype=torch.long, device=device)

            # Forward pass
            outputs = model(x, edge_index, game_indices)
            batch_y = y[batch_tensor]

            # Use modified focal loss with lower gamma
            loss = focal_loss(outputs[batch_tensor], batch_y, gamma=1.5, alpha=class_weights)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            epoch_loss += loss.item()

        # Average loss for the epoch
        epoch_loss /= num_batches
        history['train_loss'].append(epoch_loss)

        # Evaluate on training and validation sets
        model.eval()
        with torch.no_grad():
            # Training metrics
            train_outputs = model(x, edge_index, game_indices)
            train_probs = torch.exp(train_outputs)
            _, train_preds = train_outputs.max(1)

            train_metrics = calculate_binary_metrics(
                train_preds[train_indices].cpu().numpy(),
                train_probs[train_indices][:, 1].cpu().numpy(),
                y[train_indices].cpu().numpy()
            )

            for metric_name, value in train_metrics.items():
                history['train_metrics'][metric_name].append(value)

            # Test metrics
            test_metrics = calculate_binary_metrics(
                train_preds[test_indices].cpu().numpy(),
                train_probs[test_indices][:, 1].cpu().numpy(),
                y[test_indices].cpu().numpy()
            )

            for metric_name, value in test_metrics.items():
                history['val_metrics'][metric_name].append(value)

        # Track F1 history for trend analysis
        val_f1 = test_metrics['f1']
        validation_f1_history.append(val_f1)

        # Update learning rate based on validation F1
        scheduler.step(val_f1)

        # Early stopping check with minimum delta
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check for consistent improvement over last 5 epochs
        if len(validation_f1_history) >= 5:
            last_5_trend = np.array(validation_f1_history[-5:])
            if np.all(np.diff(last_5_trend) < 0.001) and epochs_no_improve >= 10:
                print(f"No significant improvement in last 5 epochs, stopping early")
                break

        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1 or epochs_no_improve == patience:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
            print(
                f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, ROC AUC: {train_metrics['roc_auc']:.4f}")
            print(
                f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, ROC AUC: {test_metrics['roc_auc']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with F1: {best_f1:.4f}")
    else:
        print("Warning: No best model found - using final model")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(x, edge_index, game_indices)
        final_probs = torch.exp(final_outputs)
        _, final_preds = final_outputs.max(1)

        final_metrics = calculate_binary_metrics(
            final_preds[test_indices].cpu().numpy(),
            final_probs[test_indices][:, 1].cpu().numpy(),
            y[test_indices].cpu().numpy()
        )

    # Print final metrics
    print("\n===== Final Model Performance =====")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1: {final_metrics['f1']:.4f}")
    print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")

    # Return the trained model, history, and metrics
    return {
        'model': model,
        'history': history,
        'best_f1': best_f1,
        'epochs_trained': epoch + 1,
        'best_model_state': best_model_state,
        'test_metrics': final_metrics
    }


def generate_feature_importance(model, data, feature_names, output_dir):
    """
    Fixed gradient-based feature importance analysis.
    """
    print("\n===== Generating Feature Importance Analysis =====")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get device
    device = torch.device('cpu')

    # Get feature data
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    y = data['y'].to(device)
    test_indices = data['test_indices']

    # Set model to evaluation mode
    model.eval()

    try:
        # Enable gradients for analysis
        x.requires_grad_(True)

        # Feature importance dictionary
        feature_importance = defaultdict(list)

        # Select a subset of test games for analysis
        if len(test_indices) > 30:
            sample_indices = np.random.choice(len(test_indices), 30, replace=False)
            analysis_indices = [test_indices[i] for i in sample_indices]
        else:
            analysis_indices = test_indices

        # Calculate gradients for each selected game
        for idx in analysis_indices:
            # Clear previous gradients
            if x.grad is not None:
                x.grad.zero_()

            # Forward pass
            outputs = model(x, edge_index, game_indices)
            prob_home_win = torch.exp(outputs[idx, 1])

            # Backward pass
            prob_home_win.backward(retain_graph=True)

            # Get gradients - use detach() to avoid gradient tracking issues
            feature_grads = (x.grad[idx] * x[idx]).abs().detach().cpu().numpy()

            # Store in dictionary
            for i, feature_name in enumerate(feature_names[:len(feature_grads)]):
                feature_importance[feature_name].append(feature_grads[i])

        # Calculate average importance across all analyzed games
        avg_importance = {}
        for feature_name, values in feature_importance.items():
            avg_importance[feature_name] = np.mean(values)

        # Convert to DataFrame
        df_importance = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        })
        df_importance = df_importance.sort_values('importance', ascending=False)

        # Create bar plot of feature importance
        plt.figure(figsize=(14, 10))
        top_features = df_importance.head(20)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 20 Features for Regulation Win Prediction', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regulation_win_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Feature importance analysis completed. Visualizations saved to:", output_dir)
        return df_importance

    except Exception as e:
        print(f"Error generating feature importance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

        # Try a simpler approach based on model weights if gradient approach fails
        try:
            # Extract weights from the first layer
            with torch.no_grad():
                weights = model.input_linear.weight.cpu().numpy()
                importance = np.abs(weights).mean(0)

                df_importance = pd.DataFrame({
                    'feature': feature_names[:len(importance)],
                    'importance': importance
                })
                df_importance = df_importance.sort_values('importance', ascending=False)

                # Plot top 20 features
                plt.figure(figsize=(14, 10))
                top_features = df_importance.head(20)
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title('Top 20 Features for Regulation Win Prediction (Weight-based)', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'regulation_win_weight_importance.png'), dpi=300,
                            bbox_inches='tight')
                plt.close()

                print("Created weight-based feature importance visualization as fallback.")
                return df_importance

        except Exception as e2:
            print(f"Error creating weight-based feature importance: {str(e2)}")
            return None


def train_and_evaluate_model_ensemble(data, feature_names, output_dir, n_models=5):
    """
    Enhanced ensemble training with varied learning rates and patience values.
    """
    print(f"\n===== Training Ensemble of {n_models} Models =====")

    # Create directory for ensemble results
    ensemble_dir = os.path.join(output_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    # Get input dimensions
    in_channels = data['x'].shape[1]

    # List to store models
    models = []
    metrics = []

    # Use different seeds, learning rates and patience values
    seeds = [42, 123, 256, 789, 1024][:n_models]
    learning_rates = [0.003, 0.002, 0.004, 0.0025, 0.0035][:n_models]
    patience_values = [15, 20, 25, 30, 35][:n_models]

    # Train each model with different hyperparameters
    for i, (seed, lr, patience) in enumerate(zip(seeds, learning_rates, patience_values)):
        print(f"\nTraining model {i + 1}/{n_models} with seed {seed}, lr={lr}, patience={patience}")

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        model = RegulationHockeyGNN(
            in_channels=in_channels,
            hidden_channels=128,
            dropout_rate1=0.3,
            dropout_rate2=0.4,
            dropout_rate3=0.5
        )

        # Train model with varied parameters to promote diversity
        result = train_regulation_model(
            model=model,
            data=data,
            epochs=150,
            lr=lr,
            weight_decay=1e-4,
            patience=patience,
            min_delta=0.005,
            batch_size=64,
        )

        # Save model metrics
        metrics.append(result['test_metrics'])

        # Save model
        model_path = os.path.join(ensemble_dir, f'regulation_model_{i + 1}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': seed,
            'lr': lr,
            'patience': patience,
            'metrics': result['test_metrics'],
        }, model_path)

        # Add model to list
        models.append(model)

    # Evaluate ensemble with weighted averaging based on individual model performance
    ensemble_metrics = evaluate_ensemble_weighted(models, data, metrics)

    # Print ensemble results
    print("\n===== Ensemble Training Complete =====")
    print(f"Individual model metrics:")
    for i, m in enumerate(metrics):
        print(f"  Model {i + 1}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, ROC AUC={m['roc_auc']:.4f}")

    print(f"Ensemble metrics:")
    for k, v in ensemble_metrics.items():
        print(f"  {k}: {v:.4f}")

    return models, ensemble_metrics


def evaluate_ensemble_weighted(models, data, model_metrics):
    """
    Evaluate ensemble with weighted averaging based on individual model performance.
    """
    device = torch.device('cpu')

    # Get feature data
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    y = data['y'].to(device)
    test_indices = data['test_indices']

    # Set all models to evaluation mode
    for model in models:
        model.eval()

    # Calculate weights based on F1 scores
    f1_scores = [metrics['f1'] for metrics in model_metrics]
    # If a model has F1=0, give it a small weight to avoid division by zero
    f1_scores = [max(score, 0.01) for score in f1_scores]
    weights = np.array(f1_scores) / sum(f1_scores)

    # Get weighted ensemble predictions
    all_probs = []
    with torch.no_grad():
        for i, model in enumerate(models):
            outputs = model(x, edge_index, game_indices)
            probs = torch.exp(outputs)
            # Weight this model's prediction by its performance
            all_probs.append(probs[:, 1] * weights[i])

    # Sum weighted probabilities
    ensemble_probs = sum(all_probs)
    ensemble_preds = (ensemble_probs > 0.5).float()

    # Calculate metrics
    test_metrics = calculate_binary_metrics(
        ensemble_preds[test_indices].cpu().numpy(),
        ensemble_probs[test_indices].cpu().numpy(),
        y[test_indices].cpu().numpy()
    )

    return test_metrics



def normalize_and_standardize_features(features_list):
    """
    Normalize and standardize feature vectors to improve GNN training.
    Combines min-max normalization with z-score standardization.

    Args:
        features_list: List of feature vectors

    Returns:
        normalized_features: List of normalized and standardized feature vectors
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

    # First apply min-max normalization for each feature
    for i in range(num_features - 1):
        col = features_array[:, i]
        col_min = np.min(col)
        col_max = np.max(col)

        # Only normalize if there's a range of values
        if col_max > col_min:
            normalized_array[:, i] = (col - col_min) / (col_max - col_min)

    # Then apply z-score standardization
    for i in range(num_features - 1):
        mean = np.mean(normalized_array[:, i])
        std = np.std(normalized_array[:, i])
        if std > 0:
            normalized_array[:, i] = (normalized_array[:, i] - mean) / std

    # Convert back to list of numpy arrays
    normalized_features = [normalized_array[i] for i in range(num_samples)]

    print(f"Features normalized and standardized. Shape: {normalized_array.shape}")
    return normalized_features




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




def evaluate_ensemble(models, data):
    """
    Evaluate the performance of an ensemble of models.

    Args:
        models: List of trained models
        data: Dictionary containing model data

    Returns:
        Dictionary of ensemble metrics
    """
    device = torch.device('cpu')

    # Get feature data
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    game_indices = data['game_indices'].to(device)
    y = data['y'].to(device)
    test_indices = data['test_indices']

    # Set all models to evaluation mode
    for model in models:
        model.eval()

    # Get ensemble predictions
    all_probs = []
    with torch.no_grad():
        for model in models:
            outputs = model(x, edge_index, game_indices)
            probs = torch.exp(outputs)
            all_probs.append(probs[:, 1])

    # Average probabilities across models
    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    ensemble_preds = (ensemble_probs > 0.5).float()

    # Calculate metrics
    test_metrics = calculate_binary_metrics(
        ensemble_preds[test_indices].cpu().numpy(),
        ensemble_probs[test_indices].cpu().numpy(),
        y[test_indices].cpu().numpy()
    )

    return test_metrics


def run_regulation_gnn_optimized(config, config_model):
    """
    Optimized version of run_regulation_gnn with improved training, feature importance,
    and ensemble methods.

    Args:
        config: Configuration object with file paths
        config_model: Configuration object with model parameters
    """
    import os
    import torch
    from collections import Counter

    print("====== Starting Optimized Regulation Win GNN Training ======")
    print(f"Loading graph from {config.file_paths['graph']}")
    training_cutoff_date = config.split_data if hasattr(config, 'split_data') else None

    # Load the graph with date filtering
    from src_code.utils.save_graph_utils import load_filtered_graph
    data_graph = load_filtered_graph(config.file_paths["graph"], training_cutoff_date)
    print(f"Graph loaded with {len(data_graph.nodes)} nodes and {len(data_graph.edges)} edges")

    # Get basic stats about the graph
    stats = get_simple_node_stats(data_graph)

    # Force CPU usage
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU operation)")

    # Extract features for regulation win prediction
    features, edge_list, labels_dict, node_mapping, feature_names, diagnostics = extract_features_for_regulation_only(
        data_graph, window_sizes=config.stat_window_sizes
    )

    # Apply improved feature normalization
    features = normalize_and_standardize_features(features)

    # Prepare data for regulation win training
    model_data = prepare_regulation_data(features, edge_list, labels_dict, test_size=0.2)

    # Extract parameters from config_model with defaults
    epochs = config_model.num_epochs if hasattr(config_model, 'num_epochs') else 150
    hidden_channels = config_model.hidden_channels if hasattr(config_model, 'hidden_channels') else 128
    window_sizes = config.stat_window_sizes if hasattr(config, 'stat_window_sizes') else [5]
    patience = config_model.patience if hasattr(config_model, 'patience') else 20

    # Create output directory for results
    output_dir = config.file_paths["gnn_analysis"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis outputs will be saved to {output_dir}")

    # Determine whether to use ensemble or single model
    use_ensemble = config_model.use_ensemble if hasattr(config_model, 'use_ensemble') else True

    if use_ensemble:
        # Train ensemble of models
        models, ensemble_metrics = train_and_evaluate_model_ensemble(
            data=model_data,
            feature_names=feature_names,
            output_dir=output_dir,
            n_models=5
        )

        # Use the first model for feature importance analysis
        model = models[0]
    else:
        # Create the regulation win model
        print(
            f"Creating regulation win GNN model with {model_data['x'].shape[1]} input features and {hidden_channels} hidden channels")
        model = RegulationHockeyGNN(
            in_channels=model_data['x'].shape[1],
            hidden_channels=hidden_channels,
            dropout_rate1=0.3,
            dropout_rate2=0.4,
            dropout_rate3=0.5
        ).to(device)

        # Train the regulation win model with optimized parameters
        print(f"\nTraining regulation win GNN with up to {epochs} epochs (patience={patience})")
        training_result = train_regulation_model(
            model=model,
            data=model_data,
            epochs=epochs,
            lr=0.005,
            weight_decay=5e-4,
            patience=patience,
            min_delta=0.005,
            batch_size=64,
            device=device
        )

    # Generate feature importance analysis
    feature_importance = generate_feature_importance(
        model=model,
        data=model_data,
        feature_names=feature_names,
        output_dir=output_dir
    )

    # Save model to file
    model_path = os.path.join(output_dir, "regulation_hockey_gnn.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': model_data['x'].shape[1],
            'hidden_channels': hidden_channels,
            'dropout_rate1': 0.3,
            'dropout_rate2': 0.4,
            'dropout_rate3': 0.5
        },
        'feature_names': feature_names
    }, model_path)
    print(f"Model saved to {model_path}")

    # Make predictions for sample games using the model or ensemble
    print("\n====== Making predictions with trained model ======")
    teams_to_predict = [
        ('TOR', 'MTL'),
        ('BOS', 'TBL'),
        ('EDM', 'CGY'),
        ('NYR', 'NYI'),
        ('PIT', 'WSH')
    ]

    if use_ensemble:
        for home_team, away_team in teams_to_predict:
            try:
                predictions = predict_with_ensemble(models, data_graph, home_team, away_team, window_sizes)
                print(f"\nMatch: {home_team} (home) vs {away_team} (away):")
                print(f"  Home team regulation win probability: {predictions['regulation_win_home']:.4f}")
                print(f"  Away team regulation win probability: {predictions['regulation_win_away']:.4f}")
                print(f"  Ensemble confidence: {predictions['confidence']:.4f}")
            except Exception as e:
                print(f"\nError predicting {home_team} vs {away_team}: {str(e)}")
                import traceback
                traceback.print_exc()
    else:
        for home_team, away_team in teams_to_predict:
            try:
                predictions = predict_regulation_game(model, data_graph, home_team, away_team, window_sizes)
                print(f"\nMatch: {home_team} (home) vs {away_team} (away):")
                print(f"  Home team regulation win probability: {predictions['regulation_win_home']:.4f}")
                print(f"  Away team regulation win probability: {predictions['regulation_win_away']:.4f}")
            except Exception as e:
                print(f"\nError predicting {home_team} vs {away_team}: {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n====== Regulation Win GNN Analysis Complete ======")
    print(f"All analysis outputs saved to {output_dir}")

    if use_ensemble:
        return models, ensemble_metrics, feature_importance
    else:
        return model, training_result, feature_importance


def predict_with_ensemble(models, data_graph, home_team, away_team, window_sizes=[5]):
    """
    Make regulation win predictions using an ensemble of models.

    Args:
        models: List of trained RegulationHockeyGNN models
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_sizes: List of window sizes for features

    Returns:
        Dictionary with prediction information and ensemble confidence
    """
    print(f"\nPredicting regulation win outcome for {home_team} (home) vs {away_team} (away) using ensemble...")

    # Set models to evaluation mode
    for model in models:
        model.eval()

    # Force CPU usage
    device = torch.device('cpu')

    # Prepare input features (identical to single model prediction)
    home_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{home_team}')
                 and data.get('type') == 'team_game_performance']

    away_tgps = [node for node, data in data_graph.nodes(data=True)
                 if isinstance(node, str) and node.endswith(f'_{away_team}')
                 and data.get('type') == 'team_game_performance']

    # Check if we have data for both teams
    if not home_tgps or not away_tgps:
        print("Cannot make prediction without data for both teams")
        return {'regulation_win_home': 0.5, 'regulation_win_away': 0.5, 'confidence': 0.0}

    # Extract features and prepare inputs (same as single model)
    # Get the most recent TGP data for each team
    home_tgp_data = data_graph.nodes[sorted(home_tgps,
                                            key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in
                                                                                              data_graph.nodes[
                                                                                                  x] else 0,
                                            reverse=True)[0]] if home_tgps else {}

    away_tgp_data = data_graph.nodes[sorted(away_tgps,
                                            key=lambda x: data_graph.nodes[x]['game_date'] if 'game_date' in
                                                                                              data_graph.nodes[
                                                                                                  x] else 0,
                                            reverse=True)[0]] if away_tgps else {}

    # Get feature dimension from the first node
    feature_dim = 0
    for node, data in data_graph.nodes(data=True):
        if data.get('type') == 'team_game_performance':
            test_features = extract_team_features(data, window_sizes)
            feature_dim = len(test_features)
            break

    if feature_dim == 0:
        feature_dim = len(window_sizes) * 12 + 2
        print(f"Using estimated feature dimension: {feature_dim}")
    else:
        print(f"Detected feature dimension: {feature_dim}")

    # Extract features
    home_features = extract_team_features(home_tgp_data, window_sizes)
    away_features = extract_team_features(away_tgp_data, window_sizes)

    # Set team indicators
    home_features[-1] = 1.0  # Home team
    away_features[-1] = 0.0  # Away team

    # Convert to numpy arrays
    home_features_np = np.array(home_features, dtype=np.float32)
    away_features_np = np.array(away_features, dtype=np.float32)
    game_features_np = np.zeros(feature_dim, dtype=np.float32)
    game_features_np[-1] = 0.5  # Mark as game node

    # Ensure consistent dimensions
    max_dim = max(len(home_features_np), len(away_features_np), len(game_features_np))
    if len(home_features_np) < max_dim:
        home_features_np = np.pad(home_features_np, (0, max_dim - len(home_features_np)), 'constant')
    if len(away_features_np) < max_dim:
        away_features_np = np.pad(away_features_np, (0, max_dim - len(away_features_np)), 'constant')
    if len(game_features_np) < max_dim:
        game_features_np = np.pad(game_features_np, (0, max_dim - len(game_features_np)), 'constant')
        game_features_np[-1] = 0.5

    # Stack features
    features_np = np.stack([home_features_np, away_features_np, game_features_np])
    x = torch.tensor(features_np, dtype=torch.float).to(device)

    # Create edge connections
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 0, 2, 1],  # Source nodes
        [2, 0, 2, 1, 0, 2, 1, 2]  # Target nodes
    ], dtype=torch.long).to(device)

    # Game index for prediction
    game_indices = torch.tensor([2], dtype=torch.long).to(device)

    # Get predictions from all models
    all_probs = []
    try:
        with torch.no_grad():
            for model in models:
                outputs = model(x, edge_index, game_indices)
                probs = torch.exp(outputs)
                all_probs.append(probs[0, 1].item())  # Probability of home win

            # Calculate ensemble statistics
            ensemble_probs = np.array(all_probs)
            mean_prob = ensemble_probs.mean()
            std_prob = ensemble_probs.std()

            # Confidence measure: 1 - (2 * std / range)
            # Higher std = lower confidence, normalized to [0, 1]
            confidence = 1.0 - min(1.0, 2.0 * std_prob)

            return {
                'regulation_win_home': mean_prob,
                'regulation_win_away': 1.0 - mean_prob,
                'confidence': confidence,
                'individual_probs': all_probs
            }
    except Exception as e:
        print(f"Error during ensemble prediction: {str(e)}")
        return {
            'regulation_win_home': 0.5,
            'regulation_win_away': 0.5,
            'confidence': 0.0
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
    from collections import Counter
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


def extract_team_features(team_data, window_sizes):
    """
    Extract features for a team from team game performance data,
    focusing on regulation-relevant statistics.

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

        # Add shot-based metrics if available
        shots_per_game = 0.0
        if f'hist_{window_size}_shots_per_game' in team_data:
            shots_per_game = team_data[f'hist_{window_size}_shots_per_game']
        features.append(shots_per_game)

        shots_against = 0.0
        if f'hist_{window_size}_shots_against' in team_data:
            shots_against = team_data[f'hist_{window_size}_shots_against']
        features.append(shots_against)

        # Special teams metrics
        pp_pct = 0.0
        if f'hist_{window_size}_pp_pct' in team_data:
            pp_pct = team_data[f'hist_{window_size}_pp_pct']
        features.append(pp_pct)

        pk_pct = 0.0
        if f'hist_{window_size}_pk_pct' in team_data:
            pk_pct = team_data[f'hist_{window_size}_pk_pct']
        features.append(pk_pct)

    # Days since last game
    if 'days_since_last_game' in team_data:
        days_value = min(team_data['days_since_last_game'], 30) / 30  # Normalize
        features.append(days_value)
    else:
        features.append(1.0)  # Default (max) days since last game

    # Team indicator (will be set in the calling function)
    features.append(0.5)  # Default placeholder

    return features


def extract_features_for_regulation_only(data_graph, window_sizes=[5, 10, 20]):
    """
    Extract features focusing only on regulation win prediction.

    Args:
        data_graph: NetworkX graph containing hockey data
        window_sizes: List of historical window sizes to use for features

    Returns:
        features_list: List of feature vectors for each node
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to regulation win labels
        node_mapping: Dictionary mapping node IDs to their indices in features_list
        feature_names: List of feature names for feature importance analysis
        diagnostics: Dictionary with diagnostic information
    """
    print(f"Extracting features for regulation win prediction with window sizes {window_sizes}...")

    # Initialize containers
    features_list = []
    node_mapping = {}
    feature_names = []

    # For regulation win prediction only
    labels_dict = {}  # 1 for home win, 0 for away win in regulation

    # Diagnostic information
    diagnostics = {
        'feature_stats': {},
        'missing_values': {},
        'total_games': 0,
        'game_outcomes': {
            'regulation_home_win': 0,
            'regulation_away_win': 0,
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
    regulation_games = 0

    # First, analyze the data to understand what we're working with
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

    print(f"Data analysis: {regulation_games} regulation games")

    # Generate feature names
    for window_size in window_sizes:
        # Team performance metrics
        feature_names.extend([
            f'home_win_rate_{window_size}',
            f'home_regulation_win_rate_{window_size}',
            f'home_goal_rate_{window_size}',
            f'home_regulation_goal_rate_{window_size}',
            f'home_goals_against_rate_{window_size}',
            f'home_regulation_goals_against_{window_size}',
            f'away_win_rate_{window_size}',
            f'away_regulation_win_rate_{window_size}',
            f'away_goal_rate_{window_size}',
            f'away_regulation_goal_rate_{window_size}',
            f'away_goals_against_rate_{window_size}',
            f'away_regulation_goals_against_{window_size}'
        ])

        # Shot metrics (if available)
        feature_names.extend([
            f'home_shots_per_game_{window_size}',
            f'home_shots_against_{window_size}',
            f'away_shots_per_game_{window_size}',
            f'away_shots_against_{window_size}'
        ])

        # Special teams metrics (if available)
        feature_names.extend([
            f'home_pp_pct_{window_size}',
            f'home_pk_pct_{window_size}',
            f'away_pp_pct_{window_size}',
            f'away_pk_pct_{window_size}'
        ])

    # Other features
    feature_names.extend([
        'home_days_since_last_game',
        'away_days_since_last_game',
        'node_type_indicator'  # 1.0 for home, 0.0 for away, 0.5 for game
    ])

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

            # Extract features for home team
            home_features = []
            for window_size in window_sizes:
                # Regulation-focused features

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

                # Add shot-based metrics if available
                shots_per_game = 0.0
                if f'hist_{window_size}_shots_per_game' in home_tgp_data:
                    shots_per_game = home_tgp_data[f'hist_{window_size}_shots_per_game']
                home_features.append(shots_per_game)

                shots_against = 0.0
                if f'hist_{window_size}_shots_against' in home_tgp_data:
                    shots_against = home_tgp_data[f'hist_{window_size}_shots_against']
                home_features.append(shots_against)

                # Special teams metrics
                pp_pct = 0.0
                if f'hist_{window_size}_pp_pct' in home_tgp_data:
                    pp_pct = home_tgp_data[f'hist_{window_size}_pp_pct']
                home_features.append(pp_pct)

                pk_pct = 0.0
                if f'hist_{window_size}_pk_pct' in home_tgp_data:
                    pk_pct = home_tgp_data[f'hist_{window_size}_pk_pct']
                home_features.append(pk_pct)

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

                # Add shot-based metrics if available
                shots_per_game = 0.0
                if f'hist_{window_size}_shots_per_game' in away_tgp_data:
                    shots_per_game = away_tgp_data[f'hist_{window_size}_shots_per_game']
                away_features.append(shots_per_game)

                shots_against = 0.0
                if f'hist_{window_size}_shots_against' in away_tgp_data:
                    shots_against = away_tgp_data[f'hist_{window_size}_shots_against']
                away_features.append(shots_against)

                # Special teams metrics
                pp_pct = 0.0
                if f'hist_{window_size}_pp_pct' in away_tgp_data:
                    pp_pct = away_tgp_data[f'hist_{window_size}_pp_pct']
                away_features.append(pp_pct)

                pk_pct = 0.0
                if f'hist_{window_size}_pk_pct' in away_tgp_data:
                    pk_pct = away_tgp_data[f'hist_{window_size}_pk_pct']
                away_features.append(pk_pct)

            # Days since last game
            if 'days_since_last_game' in away_tgp_data:
                days_value = min(away_tgp_data['days_since_last_game'], 30) / 30  # Normalize
                away_features.append(days_value)
            else:
                away_features.append(1.0)  # Default (max) days since last game
                missing_values['days_since_last_game'] += 1

            # Away team indicator
            away_features.append(0.0)  # Away team

            # Game node features (placeholder with the same dimension as team features)
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

            # Create labels for regulation win prediction only
            labeled = False

            # Extract game outcome information - focus on regulation only
            if 'games' in home_tgp_data and 'win' in home_tgp_data and 'games' in away_tgp_data and 'win' in away_tgp_data:
                # Check if the game ended in regulation
                home_games = [0, 0, 0]
                if isinstance(home_tgp_data['games'], list):
                    home_games = home_tgp_data['games']

                home_wins = [0, 0, 0]
                if isinstance(home_tgp_data['win'], list):
                    home_wins = home_tgp_data['win']

                away_wins = [0, 0, 0]
                if isinstance(away_tgp_data['win'], list):
                    away_wins = away_tgp_data['win']

                # Only label regulation games
                if home_games[0] == 1 and home_games[1] == 0 and home_games[2] == 0:
                    # Game ended in regulation
                    if home_wins[0] == 1:
                        # Home team won in regulation
                        labels_dict[game_idx] = 1
                        diagnostics['game_outcomes']['regulation_home_win'] += 1
                        labeled = True
                    elif away_wins[0] == 1:
                        # Away team won in regulation
                        labels_dict[game_idx] = 0
                        diagnostics['game_outcomes']['regulation_away_win'] += 1
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
    print("\n=== Regulation Win Feature Extraction Summary ===")
    print(f"Total games: {diagnostics['total_games']}")
    print(
        f"Labeled regulation games: {diagnostics['labeled_games']} ({diagnostics['labeled_games'] / diagnostics['total_games'] * 100:.1f}%)")

    # Print game outcome distribution
    print("\nRegulation Game Outcome Distribution:")
    reg_games = diagnostics['game_outcomes']['regulation_home_win'] + diagnostics['game_outcomes'][
        'regulation_away_win']

    print(f"Regulation: {reg_games} games")
    if reg_games > 0:
        print(
            f"  Home wins: {diagnostics['game_outcomes']['regulation_home_win']} ({diagnostics['game_outcomes']['regulation_home_win'] / reg_games * 100:.1f}% of regulation games)")
        print(
            f"  Away wins: {diagnostics['game_outcomes']['regulation_away_win']} ({diagnostics['game_outcomes']['regulation_away_win'] / reg_games * 100:.1f}% of regulation games)")

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

    # Normalize the features
    features_list = normalize_features(features_list)

    return features_list, edge_list, labels_dict, node_mapping, feature_names, diagnostics

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


def prepare_regulation_data(features, edge_list, labels_dict, test_size=0.2):
    """
    Prepare data for regulation win GNN training and testing.
    Simplified to only handle regulation win prediction without class weights.

    Args:
        features: List of feature vectors
        edge_list: List of edges as (source, target) tuples
        labels_dict: Dictionary mapping game IDs to regulation win labels
        test_size: Proportion of data to use for testing

    Returns:
        model_data: Dictionary containing all data needed for training and evaluation
    """
    print(f"Preparing regulation win train/test split with test_size={test_size}...")

    # Convert features to tensor
    x = torch.tensor(np.array(features), dtype=torch.float)

    # Convert edge list to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Create a 2x0 empty tensor as a valid but empty edge_index
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Get all game indices that have a label
    game_indices = sorted(list(labels_dict.keys()))
    game_mask = torch.zeros(len(features), dtype=torch.bool)
    game_mask[game_indices] = True

    # Create label tensor
    y = torch.full((len(game_indices),), -1, dtype=torch.long)

    # Fill in the labels for games that have them
    for i, idx in enumerate(game_indices):
        if idx in labels_dict:
            y[i] = labels_dict[idx]

    # Create stratified splits based on regulation win outcomes
    target_for_split = [labels_dict[idx] for idx in game_indices if idx in labels_dict]
    train_indices, test_indices = train_test_split(
        list(range(len(game_indices))), test_size=test_size, random_state=42,
        stratify=target_for_split
    )

    # Map local indices back to global indices
    train_games = [game_indices[i] for i in train_indices]
    test_games = [game_indices[i] for i in test_indices]

    # Create train and test masks
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[train_games] = True

    test_mask = torch.zeros(len(features), dtype=torch.bool)
    test_mask[test_games] = True

    # Print class distribution
    train_y = y[train_indices]
    test_y = y[test_indices]

    print(f"Task 'regulation_win': {len(game_indices)} labeled games")

    train_class_counts = torch.bincount(train_y, minlength=2)
    print(f"Training set:")
    print(f"  Class 0 (Away win): {train_class_counts[0]} ({train_class_counts[0] / len(train_y) * 100:.1f}%)")
    print(f"  Class 1 (Home win): {train_class_counts[1]} ({train_class_counts[1] / len(train_y) * 100:.1f}%)")

    test_class_counts = torch.bincount(test_y, minlength=2)
    print(f"Test set:")
    print(f"  Class 0 (Away win): {test_class_counts[0]} ({test_class_counts[0] / len(test_y) * 100:.1f}%)")
    print(f"  Class 1 (Home win): {test_class_counts[1]} ({test_class_counts[1] / len(test_y) * 100:.1f}%)")

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
        'test_games': test_games,
        'train_indices': train_indices,
        'test_indices': test_indices
    }

    return model_data


def predict_regulation_game(model, data_graph, home_team, away_team, window_sizes=[5]):
    """
    Make regulation win predictions for a game between two teams.
    Modified to focus only on regulation win prediction.

    Args:
        model: Trained RegulationHockeyGNN model
        data_graph: NetworkX graph containing hockey data
        home_team: Name of the home team
        away_team: Name of the away team
        window_sizes: List of window sizes for features

    Returns:
        Dictionary with prediction information
    """
    print(f"\nPredicting regulation win outcome for {home_team} (home) vs {away_team} (away)...")

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
        return {'regulation_win_home': 0.5, 'regulation_win_away': 0.5}  # Default to 50% chance

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
        feature_dim = len(window_sizes) * 12 + 2  # 12 features per window + 2 additional features
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
    # Convert lists to numpy arrays
    home_features_np = np.array(home_features, dtype=np.float32)
    away_features_np = np.array(away_features, dtype=np.float32)

    # Create dummy game node features
    game_features_np = np.zeros(feature_dim, dtype=np.float32)
    game_features_np[-1] = 0.5  # Mark as game node

    # Print shapes for debugging
    print(
        f"Feature shapes - Home: {home_features_np.shape}, Away: {away_features_np.shape}, Game: {game_features_np.shape}")

    # Ensure all arrays have the same dimension
    max_dim = max(len(home_features_np), len(away_features_np), len(game_features_np))
    if len(home_features_np) < max_dim:
        home_features_np = np.pad(home_features_np, (0, max_dim - len(home_features_np)), 'constant', constant_values=0)
    if len(away_features_np) < max_dim:
        away_features_np = np.pad(away_features_np, (0, max_dim - len(away_features_np)), 'constant', constant_values=0)
    if len(game_features_np) < max_dim:
        game_features_np = np.pad(game_features_np, (0, max_dim - len(game_features_np)), 'constant', constant_values=0)
        game_features_np[-1] = 0.5  # Ensure game marker is still set

    # Stack features into a single array
    features_np = np.stack([home_features_np, away_features_np, game_features_np])

    # Create tensor for network input and move to CPU
    x = torch.tensor(features_np, dtype=torch.float).to(device)

    # Create edge connections and move to CPU
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 0, 2, 1],  # Source nodes
        [2, 0, 2, 1, 0, 2, 1, 2]  # Target nodes
    ], dtype=torch.long).to(device)

    # Game index for prediction
    game_indices = torch.tensor([2], dtype=torch.long).to(device)  # Index of the game node

    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(x, edge_index, game_indices)

            # Apply exp to get actual probabilities
            probs = torch.exp(outputs)
            # Get probability of class 1 (home win)
            regulation_win_prob = probs[0, 1].item()

            return {
                'regulation_win_home': regulation_win_prob,
                'regulation_win_away': 1.0 - regulation_win_prob
            }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {'regulation_win_home': 0.5, 'regulation_win_away': 0.5}