# run_single_model.py

import os
import numpy as np
# import matplotlib.pyplot as plt # Not directly used here anymore for history plots
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import gc
import argparse
import traceback
import json
# import pandas as pd # Not directly used here
# import matplotlib.pyplot as plt # Not directly used here

# Import necessary functions and variables from utils and config
import config
from lens_utils import (
    LensDataset, get_model, train_one_epoch, validate_one_epoch,
    test_model, collate_fn, cleanup_memory, save_results, load_results # Added load_results back if needed
)
# Import AutoImageProcessor if available
try:
    from transformers import AutoImageProcessor
except ImportError:
    AutoImageProcessor = None

from sklearn.model_selection import train_test_split

def run_training(model_desc, model_identifier, batch_size_override=None, epochs_override=None):
    """
    Trains, validates, and tests a single specified model.
    Saves results and best weights. Handles model-specific preprocessing.
    """
    print("\n" + "="*30 + f" Processing Model: {model_desc} ({model_identifier}) " + "="*30)
    iteration_start_time = time.time()

    # --- Determine effective hyperparameters ---
    batch_size = batch_size_override if batch_size_override else config.BATCH_SIZE
    epochs = epochs_override if epochs_override else config.EPOCHS
    print(f"Using Batch Size: {batch_size}, Epochs: {epochs}")

    # --- Setup Paths ---
    safe_model_desc = config.get_safe_filename(model_desc)
    best_model_path = os.path.join(config.WEIGHTS_DIR, f"best_model_{safe_model_desc}.pth")
    result_file_path = os.path.join(config.RESULTS_DIR, f"results_{safe_model_desc}.json")
    print(f"Best weights path: {best_model_path}")
    print(f"Results JSON path: {result_file_path}")

    # Create directories if they don't exist
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True) # Ensure plots dir exists

    # --- Initialize results dictionary ---
    current_run_results = {
        'model_desc': model_desc,
        'model_identifier': model_identifier,
        'status': 'Starting',
        'error': None,
        'config': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'img_size': config.IMG_SIZE, # Note: HF processor might use different internal size
            'seed': config.SEED,
            'device': str(config.DEVICE) # Store device info
        },
        'best_epoch': -1,
        'best_val_auc': -1.0, # Use -1.0 or float('-inf') as initial best
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'test_results': {},
        'training_time_s': 0,
    }

    # --- Set Seed ---
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED) # Seed all GPUs
        # Optional: CUDNN settings for reproducibility (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    model = None
    optimizer = None
    scheduler = None
    criterion = None
    cleanup_memory()

    try:
        # --- Data Loading and Splitting ---
        print("\nLoading file paths & splitting data...")
        # (Keep your existing data loading and splitting logic here)
        # ... (loading all_train_files, all_original_val_files etc.) ...
        all_train_files = []
        all_train_labels = []
        all_original_val_files = []
        all_original_val_labels = []
        train_dir = os.path.join(config.DATA_ROOT_ORIGINAL, 'train')
        original_val_dir = os.path.join(config.DATA_ROOT_ORIGINAL, 'val')

        if not os.path.isdir(train_dir) or not os.path.isdir(original_val_dir):
            raise FileNotFoundError(f"Train ({train_dir}) or Val ({original_val_dir}) directory not found in DATA_ROOT_ORIGINAL")

        for label, cls in enumerate(config.CLASSES):
            cls_train_dir = os.path.join(train_dir, cls)
            cls_val_dir = os.path.join(original_val_dir, cls)
            if os.path.isdir(cls_train_dir):
                files = glob.glob(os.path.join(cls_train_dir, '*.npy'))
                all_train_files.extend(files)
                all_train_labels.extend([label] * len(files))
            if os.path.isdir(cls_val_dir):
                files = glob.glob(os.path.join(cls_val_dir, '*.npy'))
                all_original_val_files.extend(files)
                all_original_val_labels.extend([label] * len(files))

        if not all_train_files or not all_original_val_files:
             raise ValueError("Failed to load training or original validation files.")

        train_files, train_labels = all_train_files, all_train_labels
        val_files, test_files, val_labels, test_labels = train_test_split(
            all_original_val_files, all_original_val_labels,
            test_size=(1.0 - config.VAL_SPLIT_RATIO), random_state=config.SEED, stratify=all_original_val_labels
        )
        print(f"Data splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        if not val_files or not test_files:
             raise ValueError("Validation or Test split resulted in zero samples.")


        # --- Define Transformations OR Load Image Processor ---
        image_processor = None
        train_transform = None
        val_test_transform = None

        # Check if it's likely a Hugging Face model identifier
        # More robust: Use a list in config or check model source after loading
        is_hf_model_id = AutoImageProcessor and "/" in model_identifier

        if is_hf_model_id:
            try:
                print(f"Attempting to load AutoImageProcessor for {model_identifier}")
                image_processor = AutoImageProcessor.from_pretrained(model_identifier)
                print(f"  Successfully loaded ImageProcessor for {model_identifier}.")
                # Store processor size if available (for info, actual processing uses processor)
                if hasattr(image_processor, 'size'):
                     proc_size = image_processor.size.get('shortest_edge') or image_processor.size.get('height')
                     if proc_size:
                         current_run_results['config']['img_size_processor'] = proc_size
                # No separate transforms needed when using the processor in the Dataset
                train_transform = None
                val_test_transform = None
            except Exception as e:
                print(f"Warning: Could not load AutoImageProcessor for {model_identifier}. Error: {e}")
                print("         Falling back to default torchvision transforms.")
                image_processor = None # Ensure it's None if loading failed
                is_hf_model_id = False # Treat as non-HF for transform definition

        # Define transforms if not using an HF processor (or if HF processor failed)
        if not image_processor:
            print("Defining default torchvision transforms...")
            # Define standard transforms for torchvision/timm models
            # Note: Input to these transforms should be PIL Image now due to Dataset changes
            train_transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(), # Convert PIL to Tensor BEFORE Normalize
                transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
            ])
            val_test_transform = transforms.Compose([
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
                transforms.ToTensor(), # Convert PIL to Tensor BEFORE Normalize
                transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
            ])

        # --- Create Datasets & DataLoaders ---
        print("Creating Datasets...")
        # Pass EITHER the transform OR the image_processor to the dataset
        train_dataset = LensDataset(train_files, train_labels, transform=train_transform, image_processor=image_processor)
        val_dataset = LensDataset(val_files, val_labels, transform=val_test_transform, image_processor=image_processor)
        test_dataset = LensDataset(test_files, test_labels, transform=val_test_transform, image_processor=image_processor)

        print("Creating DataLoaders...")
        # Ensure persistent_workers is False if num_workers is 0
        persist_workers_flag = (config.NUM_WORKERS > 0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=config.PIN_MEMORY, drop_last=True, persistent_workers=persist_workers_flag)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=config.PIN_MEMORY, persistent_workers=persist_workers_flag)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=config.PIN_MEMORY, persistent_workers=persist_workers_flag)
        print("DataLoaders created.")

        # --- Initialize Model, Optimizer, Scheduler ---
        # get_model handles loading from different sources
        model = get_model(model_identifier, config.NUM_CLASSES, pretrained=True)
        model.to(config.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        # Scheduler expects AUC score, which should be maximized
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE, verbose=True)
        print(f"Initialized components for {model_desc}.")
        current_run_results['status'] = 'Initialized'

        # --- Training & Validation Loop ---
        print(f"\nStarting Training for {model_desc} ({epochs} epochs)...")
        current_run_results['status'] = 'Training'
        best_val_auc_so_far = -1.0 # Initialize best AUC tracker

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")

            # Training Step
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, model_desc)
            current_run_results['train_losses'].append(train_loss)

            # Validation Step
            val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, config.DEVICE, config.NUM_CLASSES, model_desc)
            current_run_results['val_losses'].append(val_loss)
            current_run_results['val_aucs'].append(val_auc) # Store raw AUC (can be NaN)

            # Use NaN-aware comparison for scheduler and checkpointing
            # Treat NaN AUC as worse than any valid AUC score for checkpointing/scheduling
            current_epoch_auc_for_logic = val_auc if (val_auc is not None and not np.isnan(val_auc)) else -1.0

            epoch_end_time = time.time()
            val_auc_str = f"{val_auc:.4f}" if (val_auc is not None and not np.isnan(val_auc)) else "N/A"
            print(f"Epoch {epoch+1} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc_str}, Time={epoch_end_time - epoch_start_time:.2f}s")

            # Step the scheduler based on the valid AUC score (or -1.0 if NaN)
            scheduler.step(current_epoch_auc_for_logic)

            # Checkpoint Saving Logic (NaN aware)
            # Save if current AUC is better than the best *valid* AUC seen so far
            if current_epoch_auc_for_logic > best_val_auc_so_far:
                 print(f"  New best Val AUC: {current_epoch_auc_for_logic:.4f} (Previous best: {best_val_auc_so_far:.4f})")
                 best_val_auc_so_far = current_epoch_auc_for_logic # Update best tracker
                 current_run_results['best_val_auc'] = best_val_auc_so_far # Update results dict
                 current_run_results['best_epoch'] = epoch + 1
                 try:
                     torch.save(model.state_dict(), best_model_path)
                     print(f"  ✅ Model weights saved to {best_model_path}")
                 except Exception as e_save:
                     print(f"  ERROR saving weights: {e_save}")
                     traceback.print_exc()
            # Optional: Save even if AUC is equal (updates weights to latest epoch with same best score)
            # elif current_epoch_auc_for_logic == best_val_auc_so_far and best_val_auc_so_far != -1.0:
            #      print(f"  Val AUC matched best ({best_val_auc_so_far:.4f}). Updating weights to Epoch {epoch+1}.")
            #      current_run_results['best_epoch'] = epoch + 1 # Update epoch number in results
            #      try:
            #          torch.save(model.state_dict(), best_model_path)
            #          print(f"  ✅ Model weights updated at {best_model_path}")
            #      except Exception as e_save: print(f"  ERROR saving weights on match: {e_save}")


        print(f"\nFinished Training Phase for {model_desc}.")
        current_run_results['status'] = 'Training Completed'

        # --- Testing Phase ---
        # Check if a best model was ever saved (i.e., best_val_auc_so_far is better than initial -1.0)
        if best_val_auc_so_far > -1.0:
            if os.path.exists(best_model_path):
                print(f"\nLoading best model weights (Epoch {current_run_results['best_epoch']}, Val AUC {best_val_auc_so_far:.4f}) for testing...")
                # Reload the model architecture and load the saved state dict
                model_test = None
                cleanup_memory() # Clean before loading new model
                try:
                    # Important: Load architecture again, then load weights
                    model_test = get_model(model_identifier, config.NUM_CLASSES, pretrained=False) # Don't need pretrained weights here
                    model_test.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
                    model_test.to(config.DEVICE)
                    model_test.eval() # Set to evaluation mode
                    print("Best model loaded successfully for testing.")
                    current_run_results['status'] = 'Testing'

                    # Run testing
                    test_results_dict = test_model(
                        model=model_test,
                        loader=test_loader,
                        device=config.DEVICE,
                        num_classes_test=config.NUM_CLASSES,
                        class_names_test=config.CLASSES,
                        plots_dir=config.PLOTS_DIR, # Pass plots directory
                        model_name_tag=model_desc
                    )
                    current_run_results['test_results'] = test_results_dict
                    current_run_results['status'] = 'Completed'
                    del model_test # Clean up test model

                except Exception as e_test:
                    print(f"\nERROR during testing phase for {model_desc}: {e_test}")
                    traceback.print_exc()
                    current_run_results['status'] = 'Testing Failed'
                    current_run_results['error'] = f"Testing Error: {str(e_test)}"
                    # Provide placeholder test results on error
                    current_run_results['test_results'] = {'accuracy': float('nan'), 'auc_ovr_macro': float('nan'), 'report': f'Testing Error: {e_test}'}
            else:
                print(f"\nWarning: Best weights file not found ({best_model_path}), though a best epoch ({current_run_results['best_epoch']}) was recorded. Skipping testing.")
                current_run_results['status'] = 'Testing Skipped (No Weights)'
                current_run_results['test_results'] = {'accuracy': float('nan'), 'auc_ovr_macro': float('nan'), 'report': 'N/A - Weights file missing'}
        else:
             print("\nWarning: No improvement in validation AUC observed during training. Skipping testing.")
             current_run_results['status'] = 'Testing Skipped (No Improvement)'
             current_run_results['test_results'] = {'accuracy': float('nan'), 'auc_ovr_macro': float('nan'), 'report': 'N/A - No improvement during validation'}

    except Exception as e_main:
        print(f"\nFATAL ERROR during processing of {model_desc}: {e_main}")
        traceback.print_exc()
        # Update status based on where the error likely occurred
        if current_run_results['status'] in ['Starting', 'Initialized']:
             current_run_results['status'] = 'Initialization Failed'
        elif current_run_results['status'] == 'Training':
             current_run_results['status'] = 'Training Failed'
        else: # Catch-all for errors during testing setup or other phases
             current_run_results['status'] = 'Failed'
        current_run_results['error'] = str(e_main)

    finally:
        # --- Record Time and Save Final Results ---
        iteration_end_time = time.time()
        total_time_seconds = iteration_end_time - iteration_start_time
        current_run_results['training_time_s'] = total_time_seconds
        print(f"\nTotal time for {model_desc}: {total_time_seconds:.2f}s ({total_time_seconds/60:.2f} min)")
        print(f"Final Status for {model_desc}: {current_run_results['status']}")

        # Save the results dictionary for this model
        save_results(current_run_results, result_file_path)

        # --- Clean Up Memory ---
        print(f"Cleaning up resources for {model_desc}...")
        del model, optimizer, scheduler, criterion # Remove references if they exist
        # model_test might exist if testing failed mid-way
        if 'model_test' in locals() and model_test is not None:
             del model_test
        cleanup_memory()
        print("-" * 60)


# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model for lens classification.")
    parser.add_argument("--model_desc", type=str, required=True, help="User-friendly model description (e.g., 'ResNet18')")
    parser.add_argument("--model_identifier", type=str, required=True, help="Model identifier string (e.g., 'resnet18' or 'facebook/convnextv2-...')")
    parser.add_argument("--batch_size", type=int, default=None, help="Override default batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override default number of epochs")

    args = parser.parse_args()

    run_training(
        model_desc=args.model_desc,
        model_identifier=args.model_identifier,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs
    )

    print(f"\nFinished script for model: {args.model_desc}")