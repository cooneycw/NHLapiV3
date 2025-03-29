class ConfigModel:
    def __init__(self):
        # Architecture parameters
        self.hidden_channels = 512  # Keeping the increased expressivity
        self.num_layers = 3  # Keeping the 3 GCN layers

        # Training parameters
        self.learning_rate = 0.0005  # Slightly reduced for more stability with reweighted loss
        self.num_epochs = 450  # Extended to allow for better convergence with new focus
        self.batch_size = 64  # Keeping the larger batch size for stability

        # Regularization parameters
        self.dropout_rate1 = 0.5  # Keeping strong dropout for first layers
        self.dropout_rate2 = 0.5  # Keeping dropout for later layers
        self.weight_decay = 2e-4  # Maintaining L2 regularization strength

        # Early stopping parameters
        self.patience = 50  # Increased patience to allow finding better outcome prediction
        self.lr_reduce_factor = 0.3  # Less aggressive reduction (0.3→0.5) for smoother convergence
        self.lr_reduce_patience = 5  # Extended patience for learning rate reduction

        # Multi-task loss weights - MAJOR CHANGES HERE
        self.alpha = 0.15  # Reduced weight for goals prediction (0.6→0.35)
        self.beta = 0.05  # Reduced weight for shots prediction (0.3→0.15)
        self.gamma = 0.80  # Significantly increased weight for outcome prediction (0.1→0.5)

        # New outcome-specific parameters
        self.use_outcome_head = True  # Explicitly use the dedicated outcome classification head
        self.class_weights = [1.0, 3.5, 1.5]  # Weight draw outcomes higher (home_win, draw, away_win)
        self.outcome_metric_priority = True  # Prioritize outcome accuracy for early stopping
        self.outcome_head_layers = [256, 128, 48]  # Deeper outcome prediction head

        # Additional parameters
        self.validation_split = 0.1  # Keeping the same validation split
        self.test_split = 0.2  # Keeping the same test split
        self.random_seed = 42  # Keeping the same random seed
        self.use_focal_loss = True
        self.focal_loss_gamma = 2.0  # Focusing parameter
        self.use_stratified_sampling = True