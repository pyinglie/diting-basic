import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import scipy.signal as signal
import pickle
import os
from config import Config
from glob import glob
from natsort import natsorted
from tqdm import tqdm

def set_seed(seed=Config.SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_frequency_bands(eeg_signal, fs=Config.SAMPLING_RATE, order=5):
    """
    Extract different frequency bands from EEG signal.
    Returns a dictionary with different frequency bands.
    """
    filtered_signals = {}

    for band_name, (low_freq, high_freq) in Config.FREQ_BANDS.items():
        try:
            # Normalize frequencies to Nyquist frequency
            nyq = 0.5 * fs
            low = low_freq / nyq
            high = high_freq / nyq

            if high >= 1.0:  # Ensure high frequency doesn't exceed Nyquist
                high = 0.999

            # Design filter
            b, a = signal.butter(order, [low, high], btype='band')

            # Apply filter along time dimension
            filtered = torch.from_numpy(
                signal.filtfilt(b, a, eeg_signal.numpy(), axis=1)
            ).float()

            filtered_signals[band_name] = filtered
        except Exception as e:
            print(f"Error filtering {band_name} band: {str(e)}")
            # Provide empty tensor as fallback
            filtered_signals[band_name] = torch.zeros_like(eeg_signal)

    return filtered_signals

def save_checkpoint(model, optimizer, epoch, val_metrics, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics
    }, filepath)

def load_checkpoint(model, optimizer, filepath, device=Config.DEVICE):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_metrics']

def plot_training_history(history, save_path):
    """Plot comprehensive training history"""
    plt.figure(figsize=(20, 15))

    # Plot 1: Main training metrics
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot 2: Accuracy metrics
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_top10_acc'], label='Val Top-10 Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot 3: F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot 4: Loss Components
    plt.subplot(2, 3, 4)
    plt.plot(history['train_cls_loss'], label='Classification')
    plt.plot(history['train_contrastive_loss'], label='Contrastive')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot 5: Contrastive Loss Components
    plt.subplot(2, 3, 5)
    plt.plot(history['train_global_cl_loss'], label='Global CL')
    plt.plot(history['train_view_cl_loss'], label='View CL')
    plt.plot(history['train_view_global_cl_loss'], label='View-Global CL')
    plt.title('Contrastive Loss Details')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_predictions(predictions, labels_list, filepath):
    """Save model predictions with metadata"""
    all_top_k_preds, all_top_k_probs, all_labels = predictions

    prediction_data = {
        'top_k_predictions': all_top_k_preds,
        'top_k_probabilities': all_top_k_probs,
        'true_labels': all_labels,
        'label_names': labels_list,
        'timestamp': torch.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }

    with open(filepath, 'wb') as f:
        pickle.dump(prediction_data, f)


def build_labels_list(train_path_pattern):
    """
    Builds a sorted list of unique integer labels found in the training data.
    It expects labels to be 0-indexed and contiguous.
    """
    print(f"Attempting to build labels list from training data: {train_path_pattern}")
    unique_labels = set()

    file_paths = natsorted(glob(train_path_pattern))
    if not file_paths:
        raise FileNotFoundError(f"No training files found for pattern: {train_path_pattern} to build labels_list.")

    print(f"Found {len(file_paths)} files for building labels list.")
    for path in tqdm(file_paths, desc="Building labels list from integer labels"):
        try:
            loaded_array = np.load(path, allow_pickle=True)
            if len(loaded_array) < 3:  # Need at least up to index 2 for the label
                # print(f"Warning: File {path} has fewer than 3 elements, skipping for label list building.")
                continue

            int_label = int(loaded_array[2])

            if int_label < 0:
                # print(f"Warning: File {path} has a negative integer label {int_label}. Skipping.")
                continue
            unique_labels.add(int_label)
        except ValueError as ve:
            # print(f"Warning: ValueError processing file {path} for label list (e.g. non-integer label at index 2): {ve}. Skipping.")
            continue
        except Exception as e:
            # print(f"Warning: Could not process file {path} for label list building: {e}. Skipping.")
            continue

    if not unique_labels:
        raise ValueError("No valid integer labels found in training data. Cannot build labels_list.")

    sorted_labels = sorted(list(unique_labels))

    # Check if labels are 0-indexed and contiguous
    expected_labels = list(range(len(sorted_labels)))
    if sorted_labels != expected_labels:
        raise ValueError(
            f"Labels are not 0-indexed and contiguous. Found: {sorted_labels}. Expected: {expected_labels}. "
            "Please ensure your integer labels in loaded_array[2] are 0, 1, 2, ..., N-1."
        )

    labels_list_out = sorted_labels

    print(f"Successfully built labels_list with {len(labels_list_out)} unique integer labels.")
    if len(labels_list_out) > 10:
        print(f"Labels (first 10): {labels_list_out[:10]} ... {labels_list_out[-1]}")
    else:
        print(f"Labels: {labels_list_out}")
    return labels_list_out
