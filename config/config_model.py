class ConfigModel:
    def __init__(self):
        # Architecture parameters
        self.hidden_channels = 128  # Increased from 64 for better expressivity
        self.num_layers = 2  # Keep at 2 to avoid overfitting

        # Training parameters
        self.learning_rate = 0.001  # Higher initial LR to find good solutions faster
        self.num_epochs = 150  # More epochs, but early stopping will likely trigger much earlier
        self.batch_size = 32  # Larger batch for more stable gradients

        # Regularization parameters
        self.dropout_rate1 = 0.4  # Moderate dropout after first layer
        self.dropout_rate2 = 0.3  # Lighter dropout after second layer
        self.weight_decay = 1e-4  # L2 regularization to prevent overfitting

        # Early stopping parameters
        self.patience = 8  # Much shorter patience - stop training if no improvement for 8 epochs
        self.lr_reduce_factor = 0.3  # More aggressive LR reduction (0.3 vs 0.5)
        self.lr_reduce_patience = 5  # Shorter patience for learning rate reduction