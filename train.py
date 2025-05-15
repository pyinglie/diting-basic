import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import numpy as np
from tqdm import tqdm
import pickle

from config import Config
from utils import save_checkpoint, plot_training_history, save_predictions


def train_epoch(model, train_loader, criterion, contrastive_criterion, optimizer,
                device, contrastive_weight=Config.CONTRASTIVE_WEIGHT):
    """Train the model for one epoch with Supervised Contrastive Learning"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_contrastive_loss = 0.0
    # The detailed CL losses (global, view, view-global) are removed as SupCon is now one value
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        eeg = batch['eeg'].to(device)
        eeg_augmented = batch['eeg_augmented'].to(device)
        views = {k: v.to(device) for k, v in batch['views'].items() if v.numel() > 0}  # Filter empty views
        views_aug = {k: v.to(device) for k, v in batch['views_aug'].items() if v.numel() > 0}  # Filter empty views
        labels = batch['label'].to(device)  # Crucial for SupCon

        # Forward pass
        optimizer.zero_grad()
        # Model's forward might need labels if it's used internally, but for loss, we pass it separately
        outputs, global_proj, aug_global_proj, _, _, _ = model(  # Unpack only what's needed for loss
            eeg, eeg_augmented, views, views_aug  # Pass labels to model if its forward expects it
        )

        # Classification loss
        cls_loss = criterion(outputs, labels)

        # Supervised Contrastive loss
        # Pass global_proj, aug_global_proj, and labels
        sup_contrastive_loss = contrastive_criterion(global_proj, aug_global_proj, labels)

        # Combined loss
        loss = cls_loss + contrastive_weight * sup_contrastive_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        running_contrastive_loss += sup_contrastive_loss.item()  # Store the SupCon loss

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    n_batches = len(train_loader)
    epoch_loss = running_loss / n_batches
    epoch_cls_loss = running_cls_loss / n_batches
    epoch_contrastive_loss = running_contrastive_loss / n_batches  # This is now SupCon loss

    # Return simplified loss components
    return (epoch_loss, epoch_cls_loss, epoch_contrastive_loss,
            accuracy, precision, recall, f1)


def evaluate(model, data_loader, criterion, device, labels_list, top_k=3):
    """Evaluate the model on validation or test data"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_top_k_preds = []  # Store top_k predicted indices
    all_top_k_probs = []  # Store top_k probabilities

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            eeg = batch['eeg'].to(device)
            # Views might be empty if not all samples produce all views, handle this:
            views = {k: v.to(device) for k, v in batch['views'].items() if v.numel() > 0 and v.size(1) > 0}
            labels = batch['label'].to(device)

            # Forward pass for inference (no augmented data needed)
            outputs = model(eeg, x_augmented=None, views=views, views_aug=None)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)

            # Ensure k is not greater than number of classes
            actual_k = min(top_k, probs.size(1))

            current_top_k_probs, current_top_k_indices = torch.topk(probs, k=actual_k, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top_k_preds.extend(current_top_k_indices.cpu().numpy())
            all_top_k_probs.extend(current_top_k_probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    top_k_correct = 0
    for i, label_val in enumerate(all_labels):
        if label_val in all_top_k_preds[i]:
            top_k_correct += 1
    top_k_accuracy = top_k_correct / len(all_labels) if len(all_labels) > 0 else 0

    epoch_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0

    # Print sample predictions
    # Ensure labels_list contains string names if you want to print names,
    # or it will print integer labels if labels_list is just [0, 1, ..., N-1]
    if labels_list and len(all_labels) > 0:
        print("\nSample predictions (Label mapping based on `labels_list` type):")
        for i in range(min(5, len(all_labels))):
            true_label_display = labels_list[all_labels[i]] if isinstance(labels_list[0], str) else all_labels[i]
            print(f"True label: {true_label_display}")
            print("Top predictions:")
            for j in range(min(actual_k, len(all_top_k_preds[i]))):  # Use actual_k
                pred_idx = all_top_k_preds[i][j]
                pred_label_display = labels_list[pred_idx] if isinstance(labels_list[0], str) else pred_idx
                prob = all_top_k_probs[i][j]
                print(f"  {j + 1}. {pred_label_display} (Prob: {prob:.4f})")
            print()

    return epoch_loss, accuracy, precision, recall, f1, top_k_accuracy, (all_top_k_preds, all_top_k_probs, all_labels)


def train_model(model, train_loader, dev_loader, test_loader, criterion, contrastive_criterion,
                optimizer, scheduler, device, num_epochs=Config.NUM_EPOCHS, labels_list=None,
                checkpoint_dir=Config.CHECKPOINT_DIR, contrastive_weight=Config.CONTRASTIVE_WEIGHT):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_cls_loss': [], 'train_contrastive_loss': [],  # Contrastive loss is now SupCon
        # Removed detailed CL losses: 'train_global_cl_loss', 'train_view_cl_loss', 'train_view_global_cl_loss'
        'val_top10_acc': [],
        'learning_rate': []
    }

    warmup_epochs = 3
    warmup_factor = 0.1

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        current_lr = Config.LEARNING_RATE
        if epoch < warmup_epochs:
            factor = warmup_factor + (1.0 - warmup_factor) * (epoch / warmup_epochs)
            current_lr = factor * Config.LEARNING_RATE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])  # Record actual LR

        # Train epoch returns: epoch_loss, epoch_cls_loss, epoch_contrastive_loss, accuracy, precision, recall, f1
        train_metrics = train_epoch(
            model, train_loader, criterion, contrastive_criterion, optimizer, device, contrastive_weight
        )
        train_loss, train_cls_loss, train_contrastive_loss, train_acc, _, _, train_f1 = train_metrics

        val_loss, val_acc, _, _, val_f1, val_top_k_acc, _ = evaluate(
            model, dev_loader, criterion, device, labels_list
        )

        if epoch >= warmup_epochs:  # Start scheduler after warmup
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_contrastive_loss'].append(train_contrastive_loss)  # SupCon loss
        history['val_top10_acc'].append(val_top_k_acc)

        print(f"Train Loss: {train_loss:.4f} (CLS: {train_cls_loss:.4f}, SupCL: {train_contrastive_loss:.4f})")
        print(f"Train Metrics - Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Top-3 Acc: {val_top_k_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, {
                'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_top_k_acc': val_top_k_acc
            }, os.path.join(checkpoint_dir, 'best_acc_model.pth'))
            print("Saved best accuracy model checkpoint")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {
                'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_top_k_acc': val_top_k_acc
            }, os.path.join(checkpoint_dir, 'best_loss_model.pth'))
            print("Saved best loss model checkpoint")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, {
                'val_loss': val_loss, 'val_acc': val_acc
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))

    # Load best model for final test evaluation (e.g., based on accuracy)
    best_model_path = os.path.join(checkpoint_dir, 'best_acc_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model (by accuracy) from epoch {checkpoint['epoch'] + 1} for final testing.")
        val_metrics = checkpoint['val_metrics']
        print(f"Best validation metrics - Acc: {val_metrics['val_acc']:.4f}, Loss: {val_metrics['val_loss']:.4f}, "
              f"F1: {val_metrics['val_f1']:.4f}, Top-10 Acc: {val_metrics['val_top_k_acc']:.4f}")
    else:
        print("\nNo best accuracy model checkpoint found. Using the model from the last epoch for testing.")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_top_k_acc, test_predictions = evaluate(
        model, test_loader, criterion, device, labels_list
    )

    print("\n===== Final Test Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Top-10 Accuracy: {test_top_k_acc:.4f}")

    save_predictions(test_predictions, labels_list, os.path.join(checkpoint_dir, 'test_predictions.pkl'))
    plot_training_history(history, os.path.join(checkpoint_dir, 'training_history.png'))

    return history, test_predictions