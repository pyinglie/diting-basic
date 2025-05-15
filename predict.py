import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np

# --- Import project-specific modules ---
from config import Config as DefaultConfig
from utils import set_seed  # build_labels_list might not be directly used if we use the provided n_codes_list
from data import create_multiview_batch
from models import UnifiedEEGKeywordPredictor

# Provided list of n-codes, assuming this is the order for labels 0, 1, 2...
# This list should ideally be loaded from a file or saved with the model if it's fixed for a dataset.
# For now, embedding it directly as per your request.
N_CODES_ORDERED_LIST = [
    "n02389026", "n03888257", "n03584829", "n02607072", "n03297495", "n03063599", "n03792782", "n04086273",
    "n02510455", "n11939491", "n02951358", "n02281787", "n02106662", "n04120489", "n03590841", "n02992529",
    "n03445777", "n03180011", "n02906734", "n07873807", "n03773504", "n02492035", "n03982430", "n03709823",
    "n03100240", "n03376595", "n03877472", "n03775071", "n03272010", "n04069434", "n03452741", "n03792972",
    "n07753592", "n13054560", "n03197337", "n02504458", "n02690373", "n03272562", "n04044716", "n02124075"
]


def load_synset_mapping(mapping_file_path):
    """
    Loads the synset ID to actual label name mapping from the specified file.
    Example line: "n03792972 mountain tent"
    """
    if not os.path.exists(mapping_file_path):
        print(f"Warning: Synset mapping file not found at {mapping_file_path}. Actual names will not be available.")
        return None

    mapping = {}
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)  # Split only on the first space
                if len(parts) == 2:
                    n_code, actual_name = parts
                    mapping[n_code] = actual_name
        print(f"Successfully loaded synset mapping from {mapping_file_path} with {len(mapping)} entries.")
        return mapping
    except Exception as e:
        print(f"Error loading synset mapping file {mapping_file_path}: {e}")
        return None


def predict_single_npy(checkpoint_path, npy_file_path, synset_mapping_file_path, config, top_k=3):
    """
    Loads a trained model and predicts the label for a single .npy file,
    mapping the output to actual names using the synset mapping file.
    """
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Synset to Name Mapping
    print("\n1. Loading synset to actual name mapping...")
    synset_to_name_map = load_synset_mapping(synset_mapping_file_path)

    # The `labels_list` will now be our N_CODES_ORDERED_LIST.
    # The model outputs an integer index (0, 1, ...), which maps to an n-code in this list.
    labels_list_n_codes = N_CODES_ORDERED_LIST
    num_labels = len(labels_list_n_codes)
    print(f"Using predefined list of {num_labels} n-codes for label mapping.")

    # 2. Load Model
    print(f"\n2. Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Determine model parameters ---
    # num_labels is now determined by len(N_CODES_ORDERED_LIST)
    # We should verify if the checkpoint's num_labels (if saved) matches this.
    if 'num_labels' in checkpoint and checkpoint['num_labels'] != num_labels:
        print(f"Warning: Number of labels in checkpoint ({checkpoint['num_labels']}) "
              f"does not match the provided N_CODES_ORDERED_LIST length ({num_labels}). "
              "Using length of N_CODES_ORDERED_LIST.")
    elif 'num_labels' not in checkpoint:
        print(f"Num_labels not found in checkpoint, using length of N_CODES_ORDERED_LIST: {num_labels}")

    # Load the NPY file to get its shape for in_channels
    print(f"\n3. Loading and preprocessing NPY file: {npy_file_path}")
    if not os.path.exists(npy_file_path):
        print(f"Error: NPY file not found at {npy_file_path}")
        return

    try:
        loaded_array = np.load(npy_file_path, allow_pickle=True)
        eeg_np = loaded_array[1]

        # Ensure eeg_np is [Channels, TimeSteps]
        # This logic needs to be robust based on how NPY files are saved
        # and how EEGDataset preprocesses them.
        # Assuming EEGDataset expects raw NPY as [TimeSteps, Channels] and applies .T
        if eeg_np.shape[0] == config.EXPECTED_TIME_STEPS and eeg_np.shape[1] == config.EXPECTED_CHANNELS:
            eeg_np = eeg_np.T  # Make it [Channels, TimeSteps]
        elif eeg_np.shape[0] == config.EXPECTED_CHANNELS and eeg_np.shape[1] == config.EXPECTED_TIME_STEPS:
            pass  # Already in correct format
        else:
            print(
                f"Warning: Unexpected NPY shape {eeg_np.shape}. Expected C={config.EXPECTED_CHANNELS}, T={config.EXPECTED_TIME_STEPS} (or transposed). Attempting to proceed.")
            # Heuristic if dimensions are swapped and one matches expected channels/timesteps
            if eeg_np.ndim == 2:
                if eeg_np.shape[0] == config.EXPECTED_TIME_STEPS and eeg_np.shape[1] == config.EXPECTED_CHANNELS:
                    eeg_np = eeg_np.T
                elif eeg_np.shape[1] == config.EXPECTED_TIME_STEPS and eeg_np.shape[0] == config.EXPECTED_CHANNELS:
                    pass  # Correct
                # Add more sophisticated checks or rely on fixed input format for prediction

        if eeg_np.ndim == 1:
            eeg_np = np.expand_dims(eeg_np, axis=0)

        eeg_tensor = torch.from_numpy(eeg_np.astype(np.float32))

        if eeg_tensor.numel() > 0:
            norm_val = torch.max(torch.abs(eeg_tensor))
            if norm_val > 1e-8:
                eeg_tensor = eeg_tensor / norm_val

        in_channels_from_npy = eeg_tensor.shape[0]
        print(f"NPY data loaded. Shape: {eeg_tensor.shape} (Channels, TimeSteps)")

    except Exception as e:
        print(f"Error loading or processing NPY file {npy_file_path}: {e}")
        return

    # Initialize model
    model_hyperparams = checkpoint.get('model_hyperparams', {})
    model = UnifiedEEGKeywordPredictor(
        in_channels=in_channels_from_npy,
        num_labels=num_labels,  # Crucially, this must match the model's output layer size
        hidden_dim=model_hyperparams.get('hidden_dim', config.HIDDEN_DIM),
        projection_dim=model_hyperparams.get('projection_dim', config.PROJECTION_DIM),
        temperature=model_hyperparams.get('temperature', config.TEMPERATURE)  # Classification softmax temp
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 4. Prepare data for model input
    eeg_batch = eeg_tensor.unsqueeze(0).to(device)
    views_single = create_multiview_batch(eeg_batch)
    views_single_device = {k: v.to(device) for k, v in views_single.items() if v.numel() > 0 and v.size(1) > 0}

    # 5. Prediction
    print("\n4. Making prediction...")
    with torch.no_grad():
        outputs = model(eeg_batch, views=views_single_device)

    probabilities = F.softmax(outputs, dim=1).squeeze()

    # 6. Display results
    print("\n--- Prediction Results ---")
    actual_k = min(top_k, probabilities.size(0))
    top_k_probs, top_k_indices = torch.topk(probabilities, k=actual_k)

    for i in range(actual_k):
        pred_idx = top_k_indices[i].item()  # This is the integer label (0, 1, ...)
        prob = top_k_probs[i].item()

        predicted_n_code = "<Unknown N-Code>"
        actual_name_display = "<Actual Name Not Found>"

        if 0 <= pred_idx < len(labels_list_n_codes):
            predicted_n_code = labels_list_n_codes[pred_idx]
            if synset_to_name_map and predicted_n_code in synset_to_name_map:
                actual_name_display = synset_to_name_map[predicted_n_code]
            else:
                actual_name_display = f"<Name not in mapping for {predicted_n_code}>"

        print(
            f"Top {i + 1}: Index={pred_idx}, N-Code='{predicted_n_code}', Name='{actual_name_display}', Probability={prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict EEG label from a single NPY file and map to actual names.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("npy_file_path", type=str, help="Path to the .npy file for prediction.")
    parser.add_argument("synset_mapping_file", type=str, help="Path to LOC_synset_mapping.txt file.")

    parser.add_argument("--config_hidden_dim", type=int, default=DefaultConfig.HIDDEN_DIM,
                        help="Model hidden dimension (if not in checkpoint).")
    parser.add_argument("--config_projection_dim", type=int, default=DefaultConfig.PROJECTION_DIM,
                        help="Model projection dimension (if not in checkpoint).")
    parser.add_argument("--config_temperature", type=float, default=DefaultConfig.TEMPERATURE,
                        help="Model classification softmax temperature (if not in checkpoint).")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display.")

    # Parameters for NPY data orientation, if needed
    parser.add_argument("--expected_channels", type=int, default=128,
                        help="Expected number of channels in NPY files for orientation check.")
    parser.add_argument("--expected_time_steps", type=int, default=440,
                        help="Expected number of time steps in NPY files for orientation check.")

    args = parser.parse_args()

    pred_config = DefaultConfig()  # Use a fresh config object
    pred_config.HIDDEN_DIM = args.config_hidden_dim
    pred_config.PROJECTION_DIM = args.config_projection_dim
    pred_config.TEMPERATURE = args.config_temperature  # This is for classification softmax
    pred_config.EXPECTED_CHANNELS = args.expected_channels
    pred_config.EXPECTED_TIME_STEPS = args.expected_time_steps
    # pred_config.DATA_BASE_DIR is not strictly needed here unless some util relies on it.

    predict_single_npy(
        args.checkpoint_path,
        args.npy_file_path,
        args.synset_mapping_file,
        pred_config,
        args.top_k
    )