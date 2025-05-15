import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.fft as fft
from config import Config
from utils import extract_frequency_bands
from glob import glob
from natsort import natsorted
class EEGAugmentation:
    """EEG data augmentation for contrastive learning"""
    def __init__(self,
                time_mask_prob=Config.TIME_MASK_PROB,
                channel_mask_prob=Config.CHANNEL_MASK_PROB,
                noise_level=Config.NOISE_LEVEL,
                scaling_prob=Config.SCALING_PROB,
                scaling_range=Config.SCALING_RANGE,
                time_shift_prob=Config.TIME_SHIFT_PROB,
                time_shift_max=Config.TIME_SHIFT_MAX):
        """
        Initialize augmentation parameters:
        - Time mask: Zero out random time segments
        - Channel mask: Zero out random channels
        - Noise: Add Gaussian noise
        - Scaling: Scale amplitude by random factor
        - Time shift: Shift signal in time
        """
        self.time_mask_prob = time_mask_prob
        self.channel_mask_prob = channel_mask_prob
        self.noise_level = noise_level
        self.scaling_prob = scaling_prob
        self.scaling_range = scaling_range
        self.time_shift_prob = time_shift_prob
        self.time_shift_max = time_shift_max

    def __call__(self, eeg):
        """Apply a series of augmentations to EEG data"""
        eeg = eeg.clone()  # Don't modify original data

        # Time masking
        if random.random() < self.time_mask_prob:
            mask_length = random.randint(5, min(50, eeg.shape[1] // 5))
            start_idx = random.randint(0, eeg.shape[1] - mask_length)
            eeg[:, start_idx:start_idx + mask_length] = 0

        # Channel masking
        if random.random() < self.channel_mask_prob:
            num_channels = random.randint(1, max(1, int(eeg.shape[0] * 0.1)))
            channel_indices = random.sample(range(eeg.shape[0]), num_channels)
            eeg[channel_indices, :] = 0

        # Add noise
        if random.random() < 0.5:  # 50% chance to add noise
            eeg += torch.randn_like(eeg) * self.noise_level * torch.std(eeg)

        # Scaling
        if random.random() < self.scaling_prob:
            scale = random.uniform(*self.scaling_range)
            eeg *= scale

        # Time shift
        if random.random() < self.time_shift_prob and eeg.shape[1] > self.time_shift_max * 2:
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            if shift > 0:
                eeg = torch.cat([torch.zeros_like(eeg[:, :shift]), eeg[:, :-shift]], dim=1)
            elif shift < 0:
                eeg = torch.cat([eeg[:, -shift:], torch.zeros_like(eeg[:, :abs(shift)])], dim=1)

        return eeg

def create_multiview_batch(eeg_batch):
    """
    Create multiple views of the EEG batch:
    1. Frequency bands (delta, theta, alpha, beta, gamma)
    2. Time domain segments (early, mid, late)
    3. Spatial groupings (frontal, central, parietal, occipital)
    """
    batch_size, channels, time_steps = eeg_batch.size()
    views = {}

    # 1. Frequency band views using FFT filtering
    fft_tensor = torch.fft.rfft(eeg_batch, dim=2)
    freq_bins = fft_tensor.size(2)
    freqs = torch.fft.rfftfreq(time_steps) * Config.SAMPLING_RATE  # Assuming 512 Hz sampling rate

    # Define frequency bands
    delta_mask = (freqs >= 0.5) & (freqs <= 4)
    theta_mask = (freqs > 4) & (freqs <= 8)
    alpha_mask = (freqs > 8) & (freqs <= 13)
    beta_mask = (freqs > 13) & (freqs <= 30)
    gamma_mask = (freqs > 30) & (freqs <= 100)

    # Apply masks and inverse FFT to get filtered signals
    views['delta'] = torch.fft.irfft(fft_tensor * delta_mask.view(1, 1, -1), n=time_steps, dim=2)
    views['theta'] = torch.fft.irfft(fft_tensor * theta_mask.view(1, 1, -1), n=time_steps, dim=2)
    views['alpha'] = torch.fft.irfft(fft_tensor * alpha_mask.view(1, 1, -1), n=time_steps, dim=2)
    views['beta'] = torch.fft.irfft(fft_tensor * beta_mask.view(1, 1, -1), n=time_steps, dim=2)
    views['gamma'] = torch.fft.irfft(fft_tensor * gamma_mask.view(1, 1, -1), n=time_steps, dim=2)

    # 2. Time domain segments
    segment_size = time_steps // 3
    views['early'] = eeg_batch[:, :, :segment_size]
    views['mid'] = eeg_batch[:, :, segment_size:2 * segment_size]
    views['late'] = eeg_batch[:, :, 2 * segment_size:]

    # 3. Spatial groupings (approximate for standard EEG)
    # These groups are approximate and should be adjusted based on the specific montage
    if channels >= 64:  # Only create spatial groups if we have enough channels
        frontal_channels = list(range(0, min(16, channels)))
        central_channels = list(range(16, min(32, channels)))
        parietal_channels = list(range(32, min(48, channels)))
        occipital_channels = list(range(48, min(64, channels)))

        views['frontal'] = eeg_batch[:, frontal_channels, :]
        views['central'] = eeg_batch[:, central_channels, :]
        views['parietal'] = eeg_batch[:, parietal_channels, :]
        views['occipital'] = eeg_batch[:, occipital_channels, :]

    return views


def multiview_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {
            'eeg': torch.empty(0), 'eeg_augmented': torch.empty(0),
            'views': {}, 'views_aug': {},
            'label': torch.empty(0, dtype=torch.long), 'image_idx': [],
            'subject': torch.empty(0, dtype=torch.long)
        }

    # Determine max_time_steps safely
    valid_eegs = [item['eeg'] for item in batch if item['eeg'].numel() > 0 and item['eeg'].shape[1] > 0]
    if not valid_eegs:  # All EEG tensors are empty or have 0 time steps
        max_time_steps = 0
    else:
        max_time_steps = max([eeg.shape[1] for eeg in valid_eegs])

    eeg_tensors = []
    eeg_aug_tensors = []
    labels = []
    image_indices = []  # This will remain a list of strings
    subjects = []

    for item in batch:
        eeg = item['eeg']
        eeg_aug = item.get('eeg_augmented', eeg.clone())
        current_channels = eeg.shape[0]
        current_time_steps = eeg.shape[1]

        if max_time_steps > 0 and current_time_steps > 0:  # Only pad/truncate if there's a target length and current has length
            padding_needed = max_time_steps - current_time_steps
            if padding_needed > 0:
                # Pad with zeros, ensure correct channel dimension
                padding = torch.zeros((current_channels, padding_needed), device=eeg.device)
                eeg_padded = torch.cat([eeg, padding], dim=1)
                eeg_aug_padded = torch.cat([eeg_aug, padding], dim=1)
            elif padding_needed < 0:
                eeg_padded = eeg[:, :max_time_steps]
                eeg_aug_padded = eeg_aug[:, :max_time_steps]
            else:
                eeg_padded = eeg
                eeg_aug_padded = eeg_aug
        elif max_time_steps == 0 and current_time_steps == 0:  # Both target and current are 0 time steps
            eeg_padded = eeg  # Keep as is (e.g. [channels, 0])
            eeg_aug_padded = eeg_aug
        elif max_time_steps > 0 and current_time_steps == 0:  # Current is empty, target is not
            padding = torch.zeros((current_channels, max_time_steps), device=eeg.device)
            eeg_padded = padding
            eeg_aug_padded = padding
        else:  # Should not happen if logic above is correct, but as fallback
            eeg_padded = eeg
            eeg_aug_padded = eeg_aug

        eeg_tensors.append(eeg_padded)
        eeg_aug_tensors.append(eeg_aug_padded)
        labels.append(item['label'])
        image_indices.append(item['image_idx'])  # Keep as string
        subjects.append(item['subject'])

    eeg_batch = torch.stack(eeg_tensors) if eeg_tensors else torch.empty(0)
    eeg_aug_batch = torch.stack(eeg_aug_tensors) if eeg_aug_tensors else torch.empty(0)

    # Ensure labels and subjects are tensors even if the list is empty, for consistency
    label_batch = torch.stack(labels) if labels else torch.empty(0, dtype=torch.long)
    subject_batch = torch.stack(subjects) if subjects else torch.empty(0, dtype=torch.long)

    batch_views = create_multiview_batch(eeg_batch) if eeg_batch.numel() > 0 and eeg_batch.shape[1] > 0 else {}
    batch_views_aug = create_multiview_batch(eeg_aug_batch) if eeg_aug_batch.numel() > 0 and eeg_aug_batch.shape[
        1] > 0 else {}

    return {
        'eeg': eeg_batch,
        'eeg_augmented': eeg_aug_batch,
        'views': batch_views,
        'views_aug': batch_views_aug,
        'label': label_batch,
        'image_idx': image_indices,  # THIS IS NOW CORRECTLY A LIST OF STRINGS
        'subject': subject_batch
    }


class EEGDataset(Dataset):
    """
    EEG Dataset that loads data from .npy files.
    Each .npy file is expected to contain an array-like structure where:
    - loaded_array[0]: image data (explicitly skipped)
    - loaded_array[1]: EEG data
    - loaded_array[2]: label (integer)
    - loaded_array[3]: class name (string, used as image_idx by this dataset)
    - loaded_array[4]: subject number (integer, optional)
    """

    def __init__(self, dataset_path_pattern, contrastive_transforms=None, is_cnn=False):
        self.eegs_list = []
        self.labels_list = []
        self.image_indices_list = []  # Stores class name strings
        self.subjects_list = []
        self.contrastive_transforms = contrastive_transforms
        self.is_cnn = is_cnn

        file_paths = natsorted(glob(dataset_path_pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files found for pattern: {dataset_path_pattern}")

        print(f'Loading dataset from {len(file_paths)} files matching pattern {dataset_path_pattern}...')
        for path in tqdm(file_paths, desc=f"Loading {os.path.basename(dataset_path_pattern)}"):
            try:
                loaded_array = np.load(path, allow_pickle=True)

                eeg_np = loaded_array[1]
                if self.is_cnn:
                    eeg_np = np.expand_dims(eeg_np, axis=0)

                eeg_tensor = torch.from_numpy(eeg_np).to(torch.float32)

                if eeg_tensor.numel() > 0:
                    norm_val = torch.max(torch.abs(eeg_tensor))  # More robust normalization
                    if norm_val > 1e-8:
                        normalized_eeg_tensor = eeg_tensor / norm_val
                    else:
                        normalized_eeg_tensor = eeg_tensor
                else:
                    normalized_eeg_tensor = eeg_tensor

                self.eegs_list.append(normalized_eeg_tensor)
                self.labels_list.append(torch.tensor(int(loaded_array[2]), dtype=torch.long))
                self.image_indices_list.append(str(loaded_array[3]))  # class name

                if len(loaded_array) > 4:
                    subject_val = int(loaded_array[4])
                else:
                    subject_val = -1
                self.subjects_list.append(torch.tensor(subject_val, dtype=torch.long))

            except IndexError:
                print(f"Warning: File {path} does not have the expected structure. Skipping.")
                continue
            except Exception as e:
                print(f"Warning: Error loading or processing file {path}: {e}. Skipping.")
                continue

        if not self.eegs_list:
            raise ValueError(
                f"Dataset is empty after processing pattern {dataset_path_pattern}. Check file contents or path.")
        print(f'Successfully loaded {len(self.eegs_list)} samples from {dataset_path_pattern}.')

    def __len__(self):
        return len(self.eegs_list)

    def __getitem__(self, idx):
        eeg = self.eegs_list[idx]
        label = self.labels_list[idx]
        image_idx = self.image_indices_list[idx]  # This is the class name string
        subject = self.subjects_list[idx]

        eeg_augmented = eeg.clone()
        if self.contrastive_transforms is not None:
            eeg_augmented = self.contrastive_transforms(eeg_augmented)

        return {
            'eeg': eeg,
            'eeg_augmented': eeg_augmented,
            'label': label,
            'image_idx': image_idx,
            'subject': subject
        }

def prepare_data(data_path, save_dir=Config.PROCESSED_DATA_DIR):
    """
    Load EEG data and split into train/dev/test sets using stratified sampling
    """
    print(f"Loading data from {data_path}...")
    try:
        data = torch.load(data_path)
        dataset = data['dataset']
        labels_list = data['labels']
        images_list = data['images']
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of labels: {len(labels_list)}")
    print(f"Number of images: {len(images_list)}")

    # Check EEG data dimensions
    channels = dataset[0]['eeg'].shape[0]
    time_shapes = [item['eeg'].shape[1] for item in dataset[:10]]
    print(f"EEG data has {channels} channels")
    print(f"Sample time dimensions: {time_shapes}")

    # Group samples by label for stratified sampling
    label_indices = {}
    for idx, item in enumerate(dataset):
        label = item['label']
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)

    # Perform stratified split
    train_indices = []
    dev_indices = []
    test_indices = []

    for label, indices in label_indices.items():
        n_samples = len(indices)
        n_train = int(0.7 * n_samples)
        n_dev = int(0.15 * n_samples)

        # Shuffle indices
        random.shuffle(indices)

        # Split into train, dev, and test
        train_indices.extend(indices[:n_train])
        dev_indices.extend(indices[n_train:n_train + n_dev])
        test_indices.extend(indices[n_train + n_dev:])

    # Create split datasets
    train_data = [dataset[i] for i in train_indices]
    dev_data = [dataset[i] for i in dev_indices]
    test_data = [dataset[i] for i in test_indices]

    print(f"Train set size: {len(train_data)}")
    print(f"Dev set size: {len(dev_data)}")
    print(f"Test set size: {len(test_data)}")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save split datasets
    torch.save({
        'dataset': train_data,
        'labels': labels_list,
        'images': images_list
    }, os.path.join(save_dir, 'eeg_train.pth'))

    torch.save({
        'dataset': dev_data,
        'labels': labels_list,
        'images': images_list
    }, os.path.join(save_dir, 'eeg_dev.pth'))

    torch.save({
        'dataset': test_data,
        'labels': labels_list,
        'images': images_list
    }, os.path.join(save_dir, 'eeg_test.pth'))

    print(f"Datasets saved to {save_dir}")

    return train_data, dev_data, test_data, labels_list, images_list

def get_data_loaders(train_dataset_instance, dev_dataset_instance, test_dataset_instance, batch_size=Config.BATCH_SIZE):
    """
    Create PyTorch DataLoaders for train, dev, and test sets.
    Args:
        train_dataset_instance: An already instantiated EEGDataset for training.
        dev_dataset_instance: An already instantiated EEGDataset for validation.
        test_dataset_instance: An already instantiated EEGDataset for testing.
        batch_size: The batch size for the DataLoaders.
    """
    # Datasets are now passed as already instantiated objects.
    # The EEGAugmentation should have been applied to train_dataset_instance
    # during its instantiation in main.py.

    train_loader = DataLoader(
        train_dataset_instance,  # Use the passed instance
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Consider making this configurable
        pin_memory=True,
        collate_fn=multiview_collate,
        drop_last=True
    )

    dev_loader = DataLoader(
        dev_dataset_instance,    # Use the passed instance
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Consider making this configurable
        pin_memory=True,
        collate_fn=multiview_collate
    )

    test_loader = DataLoader(
        test_dataset_instance,   # Use the passed instance
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Consider making this configurable
        pin_memory=True,
        collate_fn=multiview_collate
    )

    return train_loader, dev_loader, test_loader