import gradio as gr
import torch
import torch.nn.functional as F
import os
import numpy as np

# --- Import project-specific modules ---
from config import Config as DefaultConfig
from utils import set_seed
from data import create_multiview_batch
from models import UnifiedEEGKeywordPredictor

N_CODES_ORDERED_LIST = [
    "n02389026", "n03888257", "n03584829", "n02607072", "n03297495", "n03063599", "n03792782", "n04086273",
    "n02510455", "n11939491", "n02951358", "n02281787", "n02106662", "n04120489", "n03590841", "n02992529",
    "n03445777", "n03180011", "n02906734", "n07873807", "n03773504", "n02492035", "n03982430", "n03709823",
    "n03100240", "n03376595", "n03877472", "n03775071", "n03272010", "n04069434", "n03452741", "n03792972",
    "n07753592", "n13054560", "n03197337", "n02504458", "n02690373", "n03272562", "n04044716", "n02124075"
]

FIXED_CHECKPOINT_PATH = "./checkpoints/best_acc_model.pth"
FIXED_MAPPING_FILE_PATH = "./LOC_synset_mapping.txt"


def load_synset_mapping(mapping_file_path):
    if not os.path.exists(mapping_file_path):
        # This print is for server-side logging, not returned to Gradio UI for this particular error.
        print(f"Warning: Synset mapping file not found at the fixed path: {mapping_file_path}.")
        return None  # Function will return None if file not found
    mapping = {}
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    n_code, actual_name = parts
                    mapping[n_code] = actual_name
        # print(f"Successfully loaded synset mapping from {mapping_file_path} with {len(mapping)} entries.") # Server-side log
        return mapping
    except Exception as e:
        print(f"Error loading synset mapping file {mapping_file_path}: {e}")  # Server-side log
        return None


def perform_prediction(npy_file_path_temp, top_k_value,
                       config_hidden_dim, config_projection_dim, config_temperature,
                       expected_channels, expected_time_steps):
    # Early exit errors will return plain strings
    if not os.path.exists(FIXED_CHECKPOINT_PATH):
        return f"Error: Fixed checkpoint path not found: {FIXED_CHECKPOINT_PATH}."
    if not os.path.exists(FIXED_MAPPING_FILE_PATH):
        return f"Error: Fixed synset mapping file path not found: {FIXED_MAPPING_FILE_PATH}."

    if not npy_file_path_temp:
        return "Error: Please upload EEG Data (.npy)."

    device = torch.device("cpu")  # Force CPU
    checkpoint_path = FIXED_CHECKPOINT_PATH
    synset_mapping_file_path = FIXED_MAPPING_FILE_PATH
    npy_file_path = npy_file_path_temp

    pred_config = DefaultConfig()
    pred_config.HIDDEN_DIM = config_hidden_dim
    pred_config.PROJECTION_DIM = config_projection_dim
    pred_config.TEMPERATURE = config_temperature
    pred_config.EXPECTED_CHANNELS = expected_channels
    pred_config.EXPECTED_TIME_STEPS = expected_time_steps
    pred_config.DEVICE = device

    set_seed(pred_config.SEED)

    synset_to_name_map = load_synset_mapping(synset_mapping_file_path)
    if synset_to_name_map is None:
        # If mapping file is essential for the desired output format, this might be a hard error.
        # For now, it will result in "<Actual Name Not Found>" in the list.
        print("Warning: Synset mapping could not be loaded. Predictions will lack actual names.")

    labels_list_n_codes = N_CODES_ORDERED_LIST
    num_labels = len(labels_list_n_codes)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        return f"Error loading checkpoint: {e}"

    # Optional: server-side logging for num_labels mismatch, not part of Gradio output
    if 'num_labels' in checkpoint and checkpoint['num_labels'] != num_labels:
        print(
            f"Server Info: Checkpoint num_labels ({checkpoint['num_labels']}) != N_CODES_ORDERED_LIST length ({num_labels}).")
    elif 'num_labels' not in checkpoint:
        print(f"Server Info: Num_labels not in checkpoint, using N_CODES_ORDERED_LIST length: {num_labels}")

    try:
        loaded_array = np.load(npy_file_path, allow_pickle=True)
        eeg_np = loaded_array[1]
        # ... (NPY processing logic remains the same)
        if eeg_np.shape[0] == pred_config.EXPECTED_TIME_STEPS and eeg_np.shape[1] == pred_config.EXPECTED_CHANNELS:
            eeg_np = eeg_np.T
        elif eeg_np.shape[0] == pred_config.EXPECTED_CHANNELS and eeg_np.shape[1] == pred_config.EXPECTED_TIME_STEPS:
            pass
        else:
            print(f"Server Warning: NPY shape {eeg_np.shape} doesn't directly match. Applying heuristic if applicable.")
            if eeg_np.ndim == 2 and eeg_np.shape[0] > eeg_np.shape[1] and eeg_np.shape[1] < 200:
                eeg_np = eeg_np.T
        if eeg_np.ndim == 1: eeg_np = np.expand_dims(eeg_np, axis=0)
        eeg_tensor = torch.from_numpy(eeg_np.astype(np.float32))
        if eeg_tensor.numel() > 0:
            norm_val = torch.max(torch.abs(eeg_tensor))
            if norm_val > 1e-8: eeg_tensor = eeg_tensor / norm_val
        in_channels_from_npy = eeg_tensor.shape[0]
    except Exception as e:
        return f"Error loading/processing NPY file: {e}"

    model_hyperparams = checkpoint.get('model_hyperparams', {})
    model = UnifiedEEGKeywordPredictor(
        in_channels=in_channels_from_npy, num_labels=num_labels,
        hidden_dim=model_hyperparams.get('hidden_dim', pred_config.HIDDEN_DIM),
        projection_dim=model_hyperparams.get('projection_dim', pred_config.PROJECTION_DIM),
        temperature=model_hyperparams.get('temperature', pred_config.TEMPERATURE)
    ).to(device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        return f"Error loading model state_dict: {e}."

    try:
        eeg_batch_for_model = eeg_tensor.unsqueeze(0).to(device)
        views_for_model = create_multiview_batch(eeg_batch_for_model)
        views_for_model_device = {
            k: v.to(device) for k, v in views_for_model.items()
            if isinstance(v, torch.Tensor) and v.numel() > 0 and (v.ndim < 2 or v.size(1) > 0)
        }
        with torch.no_grad():
            outputs = model(eeg_batch_for_model, views=views_for_model_device)
        probabilities = F.softmax(outputs, dim=1).squeeze()
    except Exception as e:
        return f"Error during model inference: {e}"

    # --- MODIFIED OUTPUT SECTION ---
    top_k_names_list = []
    actual_k_val = min(top_k_value, probabilities.size(0) if probabilities.ndim > 0 else 0)

    if probabilities.ndim == 0 or actual_k_val == 0:
        # If no valid predictions, return an empty list string or an error message
        # For consistency with the requested format, an empty list string is better.
        return "[]"
        # Alternatively: return "Error: No valid predictions to display."

    top_k_probs, top_k_indices = torch.topk(probabilities, k=actual_k_val)
    for i in range(actual_k_val):
        pred_idx = top_k_indices[i].item()

        predicted_n_code = "<Unknown N-Code>"
        actual_name_display = "<Name Not Found>"

        if 0 <= pred_idx < len(labels_list_n_codes):
            predicted_n_code = labels_list_n_codes[pred_idx]
            if synset_to_name_map and predicted_n_code in synset_to_name_map:
                actual_name_display = synset_to_name_map[predicted_n_code]
            elif synset_to_name_map is None:  # Mapping file failed to load
                actual_name_display = "<Mapping File Error>"
            # If mapping loaded but code not in it, it remains "<Name Not Found>"

        top_k_names_list.append(actual_name_display)

    # Format the list of names into the string "[name1, name2, ...]"
    # Ensure names with spaces or special characters are handled (quotes not strictly needed for display but good for true list repr)
    # For simple display, f"[{', '.join(top_k_names_list)}]" is fine.
    # If the names themselves might contain commas, a more robust joiner or quoting would be needed,
    # but for typical synset names, this should be okay.
    if not top_k_names_list:  # Should be caught by earlier check, but as a safeguard
        return "[]"

    # To make it look more like a list of strings, add quotes around each name
    quoted_names = [f'"{name}"' for name in top_k_names_list]
    return f"[{', '.join(quoted_names)}]"


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# EEG Signal Prediction")
    gr.Markdown(
        "Upload an EEG NPY file to get predictions."
    )

    with gr.Row():
        with gr.Column(scale=1):

            npy_file = gr.File(label="Upload EEG Data (.npy)", type="filepath")

            gr.Markdown("### Model Configuration (Defaults if not in checkpoint)")
            hidden_dim_input = gr.Number(label="Hidden Dimension", value=DefaultConfig.HIDDEN_DIM)
            projection_dim_input = gr.Number(label="Projection Dimension", value=DefaultConfig.PROJECTION_DIM)
            temperature_input = gr.Number(label="Classification Softmax Temperature", value=DefaultConfig.TEMPERATURE)

            gr.Markdown("### NPY Data Orientation")
            expected_channels_input = gr.Number(label="Expected Channels in NPY", value=128)
            expected_time_steps_input = gr.Number(label="Expected Time Steps in NPY", value=440)

        with gr.Column(scale=2):
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K Predictions")
            predict_button = gr.Button("Predict Label", variant="primary")
            # Output text box will now display the list-like string, e.g., "[name1, name2]"
            output_text = gr.Textbox(label="Predicted Names (Top-K)", lines=3, interactive=False)

    predict_button.click(
        fn=perform_prediction,
        inputs=[
            npy_file,
            top_k_slider,
            hidden_dim_input,
            projection_dim_input,
            temperature_input,
            expected_channels_input,
            expected_time_steps_input
        ],
        outputs=output_text
    )

    gr.Markdown("---")
    gr.Markdown("#### Notes:")
    gr.Markdown("- Ensure the uploaded NPY file contains EEG data at index `1` of the loaded array.")

if __name__ == "__main__":
    critical_error = False
    if not os.path.exists(FIXED_CHECKPOINT_PATH):
        print(f"CRITICAL ERROR: Fixed checkpoint file does not exist: {os.path.abspath(FIXED_CHECKPOINT_PATH)}")
        critical_error = True
    if not os.path.exists(FIXED_MAPPING_FILE_PATH):
        print(f"CRITICAL ERROR: Fixed synset mapping file does not exist: {os.path.abspath(FIXED_MAPPING_FILE_PATH)}")
        critical_error = True

    if critical_error:
        print("Please ensure the file(s) are at the specified location(s) or update the fixed paths in the script.")
    else:
        print(f"Gradio app will use checkpoint: {os.path.abspath(FIXED_CHECKPOINT_PATH)}")
        print(f"Gradio app will use synset mapping: {os.path.abspath(FIXED_MAPPING_FILE_PATH)}")
        print("INFO: Predictions will be run on CPU.")
        demo.launch()