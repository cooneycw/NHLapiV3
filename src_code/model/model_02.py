from src_code.utils.save_graph_utils import load_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime


class HockeyGNN(nn.Module):
    def __init__(self, node_features, hidden_channels, num_layers=3):
        super(HockeyGNN, self).__init__()
        self.num_layers = num_layers

        # Separate convolutions for different node types
        self.team_convs = nn.ModuleList([
            GATConv(node_features, hidden_channels) if i == 0
            else GATConv(hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

        self.player_convs = nn.ModuleList([
            GATConv(node_features, hidden_channels) if i == 0
            else GATConv(hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

        # Prediction heads
        self.team_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2)  # win/loss prediction
        )

        self.player_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 4)  # toi, goals, assists, points
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 4)  # toi, goals, assists, points for pair
        )

    def forward(self, x, edge_index, node_type):
        # Initial features
        team_mask = (node_type == 'team')
        player_mask = (node_type == 'player')

        # Process through layers
        team_h = x[team_mask]
        player_h = x[player_mask]

        for i in range(self.num_layers):
            team_h = self.team_convs[i](team_h, edge_index)
            player_h = self.player_convs[i](player_h, edge_index)

            if i < self.num_layers - 1:
                team_h = F.relu(team_h)
                player_h = F.relu(player_h)
                team_h = F.dropout(team_h, p=0.2, training=self.training)
                player_h = F.dropout(player_h, p=0.2, training=self.training)

        # Predictions
        team_pred = self.team_predictor(team_h)
        player_pred = self.player_predictor(player_h)

        # Edge predictions
        edge_features = torch.cat([player_h[edge_index[0]], player_h[edge_index[1]]], dim=1)
        edge_pred = self.edge_predictor(edge_features)

        return team_pred, player_pred, edge_pred


def prepare_graph_data(graph, window_size=5):
    """Convert NetworkX graph to PyTorch Geometric Data objects"""

    def extract_node_features(node_data, node_type):
        if node_type == 'team':
            # Extract historical team stats
            hist_prefix = f'hist_{window_size}_'
            features = []
            for stat in ['win', 'loss', 'goal', 'goal_against']:
                stat_values = node_data.get(f'{hist_prefix}{stat}', [0, 0, 0])
                features.extend(stat_values)
            return np.array(features)

        elif node_type == 'player':
            # Extract historical player stats
            hist_prefix = f'hist_{window_size}_'
            features = []
            for stat in ['toi', 'goal', 'assist', 'point']:
                stat_values = node_data.get(f'{hist_prefix}{stat}', [0, 0, 0])
                features.extend(stat_values)
            return np.array(features)

        return np.zeros(12)  # Default feature size

    # Collect nodes and edges
    nodes = []
    node_types = []
    edges = []

    # Create node mapping
    node_to_idx = {}

    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        node_to_idx[node] = idx
        nodes.append(extract_node_features(data, data.get('type')))
        node_types.append(data.get('type'))

    # Collect edges
    for u, v in graph.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])
        edges.append([node_to_idx[v], node_to_idx[u]])  # Add reverse edge

    # Convert to tensors
    x = torch.FloatTensor(np.array(nodes))
    edge_index = torch.LongTensor(np.array(edges).T)

    # Create mask tensors
    team_mask = torch.BoolTensor([t == 'team' for t in node_types])
    player_mask = torch.BoolTensor([t == 'player' for t in node_types])

    return Data(
        x=x,
        edge_index=edge_index,
        team_mask=team_mask,
        player_mask=player_mask
    )


def create_temporal_split(graph, split_date):
    """Split data temporally based on game dates"""
    train_nodes = set()
    test_nodes = set()

    for node, data in graph.nodes(data=True):
        if 'game_date' in data:
            game_date = datetime.strptime(data['game_date'], '%Y-%m-%d')
            if game_date < split_date:
                train_nodes.add(node)
            else:
                test_nodes.add(node)

    return train_nodes, test_nodes


def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        team_pred, player_pred, edge_pred = model(
            batch.x,
            batch.edge_index,
            batch.node_type
        )

        # Calculate losses
        team_loss = F.cross_entropy(team_pred[batch.team_mask], batch.team_y)
        player_loss = F.mse_loss(player_pred[batch.player_mask], batch.player_y)
        edge_loss = F.mse_loss(edge_pred, batch.edge_y)

        # Combined loss
        loss = team_loss + player_loss + edge_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, data_loader, device):
    model.eval()
    team_preds = []
    player_preds = []
    edge_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            team_pred, player_pred, edge_pred = model(
                batch.x,
                batch.edge_index,
                batch.node_type
            )

            team_preds.append(team_pred[batch.team_mask].cpu())
            player_preds.append(player_pred[batch.player_mask].cpu())
            edge_preds.append(edge_pred.cpu())

    return {
        'team_preds': torch.cat(team_preds),
        'player_preds': torch.cat(player_preds),
        'edge_preds': torch.cat(edge_preds)
    }


def gnn(config, config_model):
    graph = load_graph(config)
    data = prepare_graph_data(graph)

    train_nodes, test_nodes = create_temporal_split(graph, config.split_date)

    # Further split test into validation and test
    val_nodes = set(list(test_nodes)[:len(test_nodes) // 2])
    test_nodes = set(list(test_nodes)[len(test_nodes) // 2:])

    # Create data loaders
    train_loader = DataLoader([data[list(train_nodes)]], batch_size=config_model.batch_size)
    val_loader = DataLoader([data[list(val_nodes)]], batch_size=config_model.batch_size)
    test_loader = DataLoader([data[list(test_nodes)]], batch_size=config_model.batch_size)

    # Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HockeyGNN(
        node_features=12,  # Based on historical features
        hidden_channels=config_model.hidden_channels,
        num_layers=config_model.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config_model.learning_rate)

    # Training loop
    for epoch in range(config_model.epochs):
        loss = train_model(model, train_loader, optimizer, device)

        if epoch % 5 == 0:
            val_metrics = evaluate_model(model, val_loader, device)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

    # Final evaluation
    test_metrics = evaluate_model(model, test_loader, device)

    return model, test_metrics