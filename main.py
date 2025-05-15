import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from config import Config
from utils import set_seed, build_labels_list
from data import get_data_loaders, EEGDataset, EEGAugmentation
from models import UnifiedEEGKeywordPredictor, EnhancedMultiViewContrastiveLoss
from train import train_model, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='EEG Classification with Multi-View Learning')
    # DATA_PATH argument might be repurposed or removed if paths are hardcoded for npy
    parser.add_argument('--data_base_dir', type=str, default='./data/raw/',
                        help='Base directory for raw train/val/test .npy file folders')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of epochs to train')
    # ... (other arguments remain the same) ...
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY,
                        help='Weight decay for optimizer')
    parser.add_argument('--hidden_dim', type=int, default=Config.HIDDEN_DIM,
                        help='Hidden dimension for the model')
    parser.add_argument('--contrastive_weight', type=float, default=Config.CONTRASTIVE_WEIGHT,
                        help='Weight for contrastive loss')
    parser.add_argument('--seed', type=int, default=Config.SEED,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default=Config.CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation, no training')
    return parser.parse_args()


def main():
    args = parse_args()

    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    Config.WEIGHT_DECAY = args.weight_decay
    Config.HIDDEN_DIM = args.hidden_dim
    Config.CONTRASTIVE_WEIGHT = args.contrastive_weight
    Config.SEED = args.seed
    Config.CHECKPOINT_DIR = args.checkpoint_dir
    Config.setup()

    set_seed(Config.SEED)

    print(f"Starting EEG classification with multi-view learning")
    print(f"Using device: {Config.DEVICE}")

    # Define data paths for .npy files
    train_path_pattern = os.path.join(args.data_base_dir, 'train/*.npy')
    val_path_pattern = os.path.join(args.data_base_dir, 'val/*.npy')
    test_path_pattern = os.path.join(args.data_base_dir, 'test/*.npy')

    print("\n1. Preparing data and labels...")
    # Build labels_list from training .npy files
    # This list maps integer labels to string names, e.g., labels_list[0] = "classNameForLabel0"
    labels_list = build_labels_list(train_path_pattern)
    num_labels = len(labels_list)

    # Create EEG Augmentation object
    eeg_augmentation = EEGAugmentation()

    # Instantiate Datasets
    print("Instantiating training dataset...")
    train_dataset = EEGDataset(dataset_path_pattern=train_path_pattern,
                               contrastive_transforms=eeg_augmentation,
                               is_cnn=False)  # Assuming is_cnn=False, adjust if needed
    print("Instantiating validation dataset...")
    dev_dataset = EEGDataset(dataset_path_pattern=val_path_pattern,
                             contrastive_transforms=None,
                             is_cnn=False)
    print("Instantiating test dataset...")
    test_dataset = EEGDataset(dataset_path_pattern=test_path_pattern,
                              contrastive_transforms=None,
                              is_cnn=False)

    print("\n2. Creating data loaders...")
    train_loader, dev_loader, test_loader = get_data_loaders(
        train_dataset, dev_dataset, test_dataset, Config.BATCH_SIZE
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check data paths and .npy file contents.")

    # Get model dimensions from data
    sample = train_dataset[0]  # Get a sample to determine input channels
    in_channels = sample['eeg'].shape[0]

    print(f"\nData dimensions:")
    print(f"Input channels: {in_channels}")
    print(f"Number of classes (from labels_list): {num_labels}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(dev_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\n3. Building model...")
    model = UnifiedEEGKeywordPredictor(
        in_channels=in_channels,
        num_labels=num_labels,  # Use num_labels derived from labels_list
        hidden_dim=Config.HIDDEN_DIM,
        projection_dim=Config.PROJECTION_DIM,
        temperature=Config.TEMPERATURE
    ).to(Config.DEVICE)

    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    contrastive_criterion = EnhancedMultiViewContrastiveLoss(temperature=Config.TEMPERATURE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    if args.eval_only:
        print("\n4. Running evaluation only...")
        # Ensure labels_list is passed to evaluate for printing purposes
        test_loss, test_acc, test_prec, test_rec, test_f1, test_top_k_acc, test_predictions = evaluate(
            model, test_loader, criterion, Config.DEVICE, labels_list
        )
        print("\n===== Test Results =====")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        # ... (print other metrics)
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Top-10 Accuracy: {test_top_k_acc:.4f}")

    else:
        print("\n4. Training model...")
        # Ensure labels_list is passed to train_model if it's used for evaluation within epochs
        history, test_predictions = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=Config.DEVICE,
            num_epochs=Config.NUM_EPOCHS,
            labels_list=labels_list,  # Pass labels_list here
            checkpoint_dir=Config.CHECKPOINT_DIR,
            contrastive_weight=Config.CONTRASTIVE_WEIGHT
        )
    print("\nDone!")


if __name__ == "__main__":
    main()