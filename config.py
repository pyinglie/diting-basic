import os
import torch

class Config:
    # General settings
    SEED = 17
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data settings
    DATA_PATH = './data/eeg_5_95_std.pth'
    PROCESSED_DATA_DIR = './processed_data/'
    CHECKPOINT_DIR = './checkpoints/'
    
    # EEG settings
    SAMPLING_RATE = 1024  # Hz
    
    # Model hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 150
    HIDDEN_DIM = 128
    PROJECTION_DIM = 128
    TEMPERATURE = 0.07
    CONTRASTIVE_WEIGHT = 0.5
    
    # Augmentation settings
    TIME_MASK_PROB = 0.2
    CHANNEL_MASK_PROB = 0.1
    NOISE_LEVEL = 0.05
    SCALING_PROB = 0.2
    SCALING_RANGE = (0.8, 1.2)
    TIME_SHIFT_PROB = 0.2
    TIME_SHIFT_MAX = 20
    
    # Frequency bands (Hz)
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    # Ensure necessary directories exist
    @classmethod
    def setup(cls):
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        
        print(f"Using device: {cls.DEVICE}")
        
        return cls