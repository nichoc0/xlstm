
"""
Centralized hyperparameter configurations for structured experimentation
"""

import os
import json
from datetime import datetime

# Base configuration
BASE_CONFIG = {
    # Data parameters
    "seq_length": 64,
    "batch_size": 32,
    "epochs": 25,
    
    # Model architecture
    "hidden_size": 32,
    "matrix_size": 3,
    "num_layers": 2,
    "dropout": 0.2,
    "expansion_factor": 2,
    
    # Training parameters
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "l1_lambda": 1e-5,
    "grad_clip": 0.5,
    "patience": 7,
    
    # Memory optimizations
    "use_checkpointing": True,
    "max_samples": 5000,
    "val_samples": 1000,
    "use_amp": True,  # Automatic mixed precision
}

# Scaled-up configuration
LARGE_CONFIG = {
    **BASE_CONFIG,
    "hidden_size": 64,
    "matrix_size": 4,
    "dropout": 0.3,
    "expansion_factor": 2,
}

# Production configuration
PRODUCTION_CONFIG = {
    **LARGE_CONFIG,
    "seq_length": 96,
    "num_layers": 3,
    "learning_rate": 1e-4,
    "weight_decay": 2e-4,
}

# Experimental configurations
EXPERIMENT_CONFIGS = {
    "deeper_model": {
        **BASE_CONFIG,
        "num_layers": 3,
        "hidden_size": 32,
    },
    "wider_model": {
        **BASE_CONFIG,
        "hidden_size": 64,
        "matrix_size": 2,
    },
    "longer_sequence": {
        **BASE_CONFIG,
        "seq_length": 96,
    }
}

def save_config(config, experiment_name=None):
    """Save configuration to disk for reproducibility"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
        
    os.makedirs("experiments", exist_ok=True)
    config_path = os.path.join("experiments", f"{experiment_name}_config.json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return experiment_name, config_path

def load_config(experiment_name):
    """Load configuration from disk"""
    config_path = os.path.join("experiments", f"{experiment_name}_config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config
