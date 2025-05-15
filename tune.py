import optuna
import torch
import os
import random
import numpy as np
import argparse
import copy  # For deep copying config

# --- Import project-specific modules ---
from config import Config as DefaultConfig  # Rename to avoid conflict
from utils import set_seed, build_labels_list
from data import get_data_loaders, EEGDataset, EEGAugmentation
from models import UnifiedEEGKeywordPredictor, EnhancedMultiViewContrastiveLoss
from train import train_model  # We will call train_model directly

# Global counter for trials
trial_counter = 0


def execute_trial_training(current_config, optuna_trial):
    """
    Encapsulates the training and evaluation for a single Optuna trial.
    Args:
        current_config: A Config object instance with hyperparams for this trial.
        optuna_trial: The Optuna trial object, for reporting/pruning.

    Returns:
        The metric to be optimized (e.g., best validation accuracy).
    """
    global trial_counter
    trial_counter += 1
    print(f"\n🚀 Starting Optuna Trial: {trial_counter} 🚀")
    print("Hyperparameters for this trial:")
    for key, value in optuna_trial.params.items():
        print(f"  {key}: {value}")
    print(f"  NUM_EPOCHS (for this trial): {current_config.NUM_EPOCHS}")
    print(f"  BATCH_SIZE (for this trial): {current_config.BATCH_SIZE}")

    # 1. Set seed for reproducibility within this trial
    set_seed(current_config.SEED)  # Uses the SEED from the config for this trial

    # 2. Setup (directories, device from current_config)
    current_config.setup()  # Creates checkpoint_dir etc.

    # 3. Data preparation
    #    Paths are assumed to be correctly set in current_config.DATA_BASE_DIR
    #    (or you can make data_base_dir a tunable parameter if needed)
    train_path_pattern = os.path.join(current_config.DATA_BASE_DIR, 'train/*.npy')
    val_path_pattern = os.path.join(current_config.DATA_BASE_DIR, 'val/*.npy')
    test_path_pattern = os.path.join(current_config.DATA_BASE_DIR, 'test/*.npy')  # For final eval by train_model

    print("\n[Trial Data] Preparing data and labels...")
    try:
        labels_list = build_labels_list(train_path_pattern)
        num_labels = len(labels_list)
    except Exception as e:
        print(f"Error building labels list: {e}. Pruning trial.")
        raise optuna.TrialPruned(f"Failed to build labels_list: {e}")

    eeg_augmentation = EEGAugmentation(
        time_mask_prob=current_config.TIME_MASK_PROB,
        channel_mask_prob=current_config.CHANNEL_MASK_PROB,
        noise_level=current_config.NOISE_LEVEL,
        scaling_prob=current_config.SCALING_PROB,
        scaling_range=current_config.SCALING_RANGE,
        time_shift_prob=current_config.TIME_SHIFT_PROB,
        time_shift_max=current_config.TIME_SHIFT_MAX
    )

    try:
        print("[Trial Data] Instantiating training dataset...")
        train_dataset = EEGDataset(dataset_path_pattern=train_path_pattern,
                                   contrastive_transforms=eeg_augmentation)
        print("[Trial Data] Instantiating validation dataset...")
        dev_dataset = EEGDataset(dataset_path_pattern=val_path_pattern)
        print("[Trial Data] Instantiating test dataset...")
        test_dataset = EEGDataset(dataset_path_pattern=test_path_pattern)  # Needed for train_model's final test run
    except FileNotFoundError as e:
        print(f"Data files not found: {e}. Pruning trial.")
        raise optuna.TrialPruned(f"Data files not found: {e}")
    except ValueError as e:  # Catch empty dataset error from EEGDataset
        print(f"Dataset error: {e}. Pruning trial.")
        raise optuna.TrialPruned(f"Dataset error: {e}")

    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        print("Error: Training or validation dataset is empty. Pruning trial.")
        raise optuna.TrialPruned("Empty train/dev dataset.")

    print("\n[Trial Data] Creating data loaders...")
    train_loader, dev_loader, test_loader = get_data_loaders(
        train_dataset, dev_dataset, test_dataset, current_config.BATCH_SIZE
    )

    sample = train_dataset[0]
    in_channels = sample['eeg'].shape[0]
    print(f"[Trial Data] Input channels: {in_channels}, Num labels: {num_labels}")

    # 4. Model Initialization
    print("\n[Trial Model] Building model...")
    model = UnifiedEEGKeywordPredictor(
        in_channels=in_channels,
        num_labels=num_labels,
        hidden_dim=current_config.HIDDEN_DIM,
        projection_dim=current_config.PROJECTION_DIM,
        temperature=current_config.TEMPERATURE  # This is for the classification softmax temp
    ).to(current_config.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    # The temperature for SupCon loss is also a hyperparameter we are tuning
    contrastive_criterion = EnhancedMultiViewContrastiveLoss(
        temperature=current_config.SUPCON_TEMPERATURE  # Use a distinct name for SupCon temp
    ).to(current_config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=current_config.LEARNING_RATE,
        weight_decay=current_config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=current_config.SCHEDULER_PATIENCE  # Tunable patience
    )

    # 5. Training
    print("\n[Trial Training] Starting training model for this trial...")
    # `train_model` will run for `current_config.NUM_EPOCHS`
    # It saves checkpoints in `current_config.CHECKPOINT_DIR` (maybe make this trial-specific)
    # and returns history. We need the best validation accuracy from this history.

    # Create a trial-specific checkpoint directory to avoid conflicts
    trial_checkpoint_dir = os.path.join(current_config.CHECKPOINT_DIR, f"trial_{optuna_trial.number}")
    os.makedirs(trial_checkpoint_dir, exist_ok=True)

    history, _ = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,  # train_model performs a final test, ensure test_loader is ready
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=current_config.DEVICE,
        num_epochs=current_config.NUM_EPOCHS,  # Use epochs set for this trial
        labels_list=labels_list,
        checkpoint_dir=trial_checkpoint_dir,  # Use trial-specific checkpoint dir
        contrastive_weight=current_config.CONTRASTIVE_WEIGHT
    )

    best_val_accuracy_for_this_trial = 0.0
    if history and 'val_acc' in history and history['val_acc']:
        best_val_accuracy_for_this_trial = max(history['val_acc'])

    print(f"Trial {trial_counter} finished. Best Validation Accuracy: {best_val_accuracy_for_this_trial:.4f}")

    # Optuna Pruning (optional but recommended)
    # Report the best val_acc achieved in this trial at its final epoch
    optuna_trial.report(best_val_accuracy_for_this_trial, step=current_config.NUM_EPOCHS - 1)
    if optuna_trial.should_prune():
        print(f"Trial {trial_counter} pruned.")
        raise optuna.TrialPruned()

    return best_val_accuracy_for_this_trial


def objective(trial, args):
    """
    Optuna objective function.
    Args:
        trial: Optuna trial object.
        args: Command line arguments (for fixed settings like data_base_dir).
    """
    # Create a fresh config for this trial by deep copying the default
    # This is important to avoid state leakage between trials if Config is a class with mutable members
    cfg = copy.deepcopy(DefaultConfig())

    # --- Suggest Hyperparameters ---
    cfg.LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64])  # Based on your default
    cfg.CONTRASTIVE_WEIGHT = trial.suggest_float("contrastive_weight", 0.1, 1.0)

    # Temperature for the classification layer's softmax (from original Config.TEMPERATURE)
    cfg.TEMPERATURE = trial.suggest_float("classification_temp", 0.05, 0.2)
    # Distinct temperature for the Supervised Contrastive Loss
    cfg.SUPCON_TEMPERATURE = trial.suggest_float("supcon_temp", 0.05, 0.5)

    cfg.HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [64, 128, 192, 256])
    # PROJECTION_DIM could be tied to HIDDEN_DIM or tuned independently
    # For simplicity, let's tie it for now, or make it a separate choice
    # cfg.PROJECTION_DIM = cfg.HIDDEN_DIM
    cfg.PROJECTION_DIM = trial.suggest_categorical("projection_dim", [64, 128, 192])

    cfg.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Scheduler patience
    cfg.SCHEDULER_PATIENCE = trial.suggest_int("scheduler_patience", 3, 10)

    # Number of epochs for each Optuna trial (can be fixed or tuned)
    # For faster initial tuning, use fewer epochs than the final training.
    cfg.NUM_EPOCHS = trial.suggest_int("num_epochs_trial", 25, 50)  # e.g., 25-50 epochs for tuning

    # Data Augmentation (Optional to tune, can be complex)
    # Example for one augmentation parameter:
    cfg.NOISE_LEVEL = trial.suggest_float("noise_level", 0.01, 0.1, log=True)
    # cfg.TIME_MASK_PROB = trial.suggest_float("time_mask_prob", 0.1, 0.4)

    # Pass fixed arguments from command line to config
    cfg.DATA_BASE_DIR = args.data_base_dir
    cfg.SEED = args.seed  # Use the overall seed for Optuna's sampler, trial execution uses its own.

    # --- Execute Training for this Trial ---
    try:
        val_metric = execute_trial_training(cfg, trial)
    except optuna.TrialPruned as e:
        print(f"Trial pruned: {e}")
        raise  # Re-raise to let Optuna handle it
    except Exception as e:
        print(f"An error occurred during trial execution: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # Return a poor score for failed trials not caught by pruning

    return val_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for EEG Classification")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument('--study_name', type=str, default="eeg_supcon_tuning_zhao_xiyuan_v1",
                        help="Name for the Optuna study")
    parser.add_argument('--data_base_dir', type=str, default='./data/raw/',
                        help='Base directory for raw train/val/test .npy file folders')
    parser.add_argument('--seed', type=int, default=DefaultConfig.SEED,
                        help='Base random seed for Optuna sampler and reproducibility.')
    parser.add_argument('--timeout', type=int, default=None,
                        help="Timeout for the Optuna study in seconds (e.g., 3600*8 for 8 hours)")

    cli_args = parser.parse_args()

    # Set the base seed for Optuna's sampler. Individual trials will also use this seed.
    set_seed(cli_args.seed)  # Sets for torch, numpy, random for the main script.

    storage_name = f"sqlite:///{cli_args.study_name}.db"

    # Sampler: TPESampler is a good default. Adding a seed makes its behavior deterministic.
    sampler = optuna.samplers.TPESampler(seed=cli_args.seed)

    # Pruner: MedianPruner can stop unpromising trials early.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10,
                                         interval_steps=1)  # Adjust warmup_steps based on num_epochs_trial

    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",  # We want to maximize validation accuracy
        sampler=sampler,
        pruner=pruner
    )

    print(f"Starting Optuna study: {cli_args.study_name}")
    print(f"  Number of trials: {cli_args.n_trials}")
    print(f"  Storage: {storage_name}")
    print(f"  Sampler: TPESampler (seeded)")
    print(f"  Pruner: MedianPruner")
    print(f"  Timeout: {cli_args.timeout} seconds" if cli_args.timeout else "No timeout")
    print(f"  Base data directory: {cli_args.data_base_dir}")
    print(f"  Default device: {DefaultConfig.DEVICE}")

    try:
        study.optimize(lambda trial: objective(trial, cli_args),
                       n_trials=cli_args.n_trials,
                       timeout=cli_args.timeout)
    except KeyboardInterrupt:
        print("Optimization stopped by user.")
    except Exception as e:
        print(f"An critical error occurred during study.optimize: {e}")
        import traceback

        traceback.print_exc()

    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Max Validation Accuracy): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save best params to a file
        best_params_file = os.path.join(DefaultConfig.CHECKPOINT_DIR, f"{cli_args.study_name}_best_params.txt")
        with open(best_params_file, 'w') as f:
            f.write(f"Best Validation Accuracy: {best_trial.value:.4f}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")
        print(f"Best parameters saved to {best_params_file}")

    except ValueError:
        print("No trials were completed successfully, or study was empty.")
    except Exception as e:
        print(f"Error retrieving best trial: {e}")

    # For more detailed analysis:
    # df = study.trials_dataframe()
    # print("\nAll trials DataFrame:")
    # print(df.sort_values(by="value", ascending=False))

    print(f"\nOptuna study '{cli_args.study_name}' complete. Results stored in '{storage_name}'.")