# lens_utils.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as vision_models # For ResNet
import json
import traceback # Added for better error reporting in __getitem__
from PIL import Image # Added: HF processors often work well with PIL

try:
    import timm
except ImportError:
    print("Warning: Timm library not found. Install with 'pip install timm'")
    timm = None
try:
    # Ensure both are imported for type checking and usage
    from transformers import ConvNextV2ForImageClassification, AutoImageProcessor
except ImportError:
    print("Warning: Transformers library not found. Install with 'pip install transformers'")
    ConvNextV2ForImageClassification = None
    AutoImageProcessor = None # Explicitly set to None if import fails

from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle
import glob
import gc

# --- JSON Serialization Helper for NumPy arrays ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj): 
                return None
            if np.isinf(obj): 
                return str(obj)
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.encode_list(obj.tolist())
        elif isinstance(obj, (np.bool_, bool)):
             return bool(obj)
        return json.JSONEncoder.default(self, obj)

    def encode_list(self, lst):
        new_list = []
        for item in lst:
            if isinstance(item, list):
                new_list.append(self.encode_list(item))
            elif isinstance(item, np.floating):
                 if np.isnan(item): 
                     new_list.append(None)
                 elif np.isinf(item): 
                     new_list.append(str(item))
                 else: 
                     new_list.append(float(item))
            elif isinstance(item, np.integer):
                 new_list.append(int(item))
            elif isinstance(item, (np.bool_, bool)):
                 new_list.append(bool(item))
            else:
                 new_list.append(item)
        return new_list

def save_results(data, filepath):
    """Saves dictionary data to a JSON file, handling numpy types."""
    print(f"  Saving results to {filepath}...")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        print(f"  Successfully saved results.")
    except TypeError as e:
         print(f"  ERROR saving results to {filepath}: TypeError - {e}")
         print("    Check if data contains non-standard types.")
    except Exception as e:
        print(f"  ERROR saving results to {filepath}: {e}")


def load_results(filepath):
    """Loads dictionary data from a JSON file."""
    if not os.path.exists(filepath):
        return None
    print(f"  Loading results from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print("  Successfully loaded results.")
        return data
    except json.JSONDecodeError as e:
        print(f"  ERROR loading results from {filepath}: Invalid JSON - {e}")
        return None
    except Exception as e:
        print(f"  ERROR loading results from {filepath}: {e}")
        return None


# --- Updated Dataset Class ---
class LensDataset(Dataset):
    # Added image_processor argument
    def __init__(self, file_paths, labels, transform=None, image_processor=None):
        self.file_paths = file_paths
        self.labels = labels
        # Store either the torchvision transform or the HF image processor
        self.transform = transform
        self.image_processor = image_processor
        # Ensure only one preprocessing method is active
        if self.transform and self.image_processor:
            raise ValueError("LensDataset received both transform and image_processor. Only one should be provided.")
        if not self.transform and not self.image_processor:
             print("Warning: LensDataset created without transform or image_processor. Returning raw data.")


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if idx >= len(self.file_paths):
            print(f"Warning: Index {idx} out of bounds for LensDataset.")
            return None, None
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                return None, None

            # Load numpy array
            image_np = np.load(file_path).astype(np.float32)

            # --- Prepare image in a consistent format (e.g., PIL RGB) ---
            # HF processors often prefer PIL images or NumPy arrays (H, W, C)
            # Torchvision transforms usually take PIL or Tensor (C, H, W)
            image_pil = None
            if image_np.ndim == 2: # Grayscale (H, W)
                # Convert to PIL RGB
                image_pil = Image.fromarray(image_np).convert('RGB')
            elif image_np.ndim == 3 and image_np.shape[0] == 1: # (1, H, W)
                image_pil = Image.fromarray(np.squeeze(image_np, axis=0)).convert('RGB')
            elif image_np.ndim == 3 and image_np.shape[2] == 3: # Already (H, W, 3)
                 image_pil = Image.fromarray(image_np.astype(np.uint8)) # PIL prefers uint8 usually
            elif image_np.ndim == 3 and image_np.shape[0] == 3: # (C, H, W) - Less common for loading
                 # Convert to PIL (needs transpose)
                 image_pil = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8)).convert('RGB')
            else:
                print(f"Warning: Unexpected image dimensions {image_np.shape} for {file_path}")
                return None, None

            if image_pil is None:
                 print(f"Warning: Could not convert numpy array to PIL for {file_path}")
                 return None, None

            # --- Apply the appropriate preprocessing ---
            if self.image_processor:
                # Use Hugging Face processor
                # It handles resizing, normalization, and tensor conversion
                # `do_rescale=False` might be needed if .npy is not 0-255
                processed = self.image_processor(images=image_pil, return_tensors="pt")
                image_tensor = processed['pixel_values'][0] # Remove batch dimension
            elif self.transform:
                # Use torchvision transform pipeline
                image_tensor = self.transform(image_pil)
            else:
                # No preprocessing specified, return basic tensor (less useful)
                # Convert PIL back to numpy -> tensor if needed, or use ToTensor in transform
                image_tensor = transforms.ToTensor()(image_pil) # Example basic conversion

            if image_tensor is None:
                print(f"Warning: image_tensor is None after processing {file_path}")
                return None, None

            return image_tensor, label

        except Exception as e:
            print(f"ERROR processing file {file_path}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return None, None # Return None on error


# --- Collate Function (Unchanged) ---
def collate_fn(batch):
    # Filter out None items returned by __getitem__ due to errors
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        # print("Warning: collate_fn received an empty batch after filtering.")
        return torch.Tensor(), torch.Tensor() # Return empty tensors
    try:
        # Use default collate
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Collate Error: {e}")
        traceback.print_exc()
        return torch.Tensor(), torch.Tensor() # Return empty tensors on error

# --- Model Loading Function (Unchanged) ---
def get_model(model_identifier, num_classes, pretrained=True):
    print(f"Loading model: {model_identifier} (pretrained={pretrained})")
    model = None
    is_hf_model = False # Flag to track model type
    try:
        # Check if it's a Hugging Face model identifier (heuristic)
        # More robust: Check if model_identifier is in a predefined list of HF models or check source explicitly
        if AutoImageProcessor and "/" in model_identifier: # Simple check for HF path format
             try:
                 # Try loading as HF model first
                 if ConvNextV2ForImageClassification is None: 
                     raise ImportError("transformers not installed.")
                 # Check specific model types if needed, otherwise assume compatible base class
                 if 'convnextv2' in model_identifier.lower():
                     model = ConvNextV2ForImageClassification.from_pretrained(model_identifier, num_labels=num_classes, ignore_mismatched_sizes=True)
                     print(f" Loaded {model_identifier} as ConvNextV2 from Hugging Face.")
                     is_hf_model = True
                 # Add elif blocks here for other HF model types like ViT if they need specific classes
                 # elif 'vit' in model_identifier.lower():
                 #     from transformers import ViTForImageClassification # Example
                 #     model = ViTForImageClassification.from_pretrained(...)
                 #     is_hf_model = True

                 # Fallback generic loading if specific type isn't matched but looks like HF path
                 if not is_hf_model:
                      print(f" Attempting generic HF load for {model_identifier}...")
                      # This might require AutoModelForImageClassification
                      from transformers import AutoModelForImageClassification
                      model = AutoModelForImageClassification.from_pretrained(model_identifier, num_labels=num_classes, ignore_mismatched_sizes=True)
                      print(f" Loaded {model_identifier} generically from Hugging Face.")
                      is_hf_model = True # Assume success means it's HF compatible

             except Exception as hf_e:
                  print(f"  Info: Failed to load {model_identifier} via Hugging Face ({hf_e}). Trying other sources.")
                  is_hf_model = False # Reset flag if HF loading fails

        if not is_hf_model and model_identifier.startswith("resnet"):
            # Torchvision ResNets
            if model_identifier == "resnet18": 
                weights = vision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = vision_models.resnet18(weights=weights)
            elif model_identifier == "resnet34": 
                weights = vision_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                model = vision_models.resnet34(weights=weights)
            elif model_identifier == "resnet50": 
                weights = vision_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                model = vision_models.resnet50(weights=weights)
            elif model_identifier == "resnet101": 
                weights = vision_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
                model = vision_models.resnet101(weights=weights)
            elif model_identifier == "resnet152": 
                weights = vision_models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
                model = vision_models.resnet152(weights=weights)
            else: 
                raise ValueError(f"Unknown torchvision resnet: {model_identifier}")
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(f" Loaded {model_identifier} from Torchvision.")
        elif not is_hf_model: # Assume Timm if not HF or ResNet
            if timm is None: 
                raise ImportError("timm not installed.")
            print(f" Attempting {model_identifier} via Timm...")
            if not timm.is_model(model_identifier):
                 similar = timm.list_models(f'*{model_identifier.split("-")[0]}*')
                 raise ValueError(f"Model '{model_identifier}' not in timm. Similar: {similar[:15]}...")
            model = timm.create_model(model_identifier, pretrained=pretrained, num_classes=num_classes, in_chans=3)
            print(f" Loaded {model_identifier} from Timm.")

        if model is None: 
            raise ValueError("Model failed to load.")
        return model
    except ImportError as e: 
        print(f"Import Error: {e}")
        raise
    except Exception as e: 
        print(f"General Error loading {model_identifier}: {e}")
        raise


# --- Training Function (Unchanged - handles HF model input) ---
def train_one_epoch(model, loader, criterion, optimizer, device, model_id):
    model.train()
    running_loss = 0.0
    num_samples = 0
    loop = tqdm(loader, desc=f"Train Ep", leave=False)
    for images, labels in loop:
        if images.nelement() == 0 or labels.nelement() == 0: 
            continue
        batch_size = images.size(0)
        if batch_size == 0: 
            continue
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        try:
            # Check if model is likely a Hugging Face model (requires 'pixel_values')
            # More robust: Pass a flag from get_model or check class explicitly
            # Using isinstance check which is reliable if transformers is installed
            if ConvNextV2ForImageClassification and isinstance(model, ConvNextV2ForImageClassification):
                 outputs = model(pixel_values=images).logits
            # Add elif for other specific HF types if needed
            # elif ViTForImageClassification and isinstance(model, ViTForImageClassification):
            #      outputs = model(pixel_values=images).logits
            else: # Assume torchvision or timm model
                 outputs = model(images)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any(): 
                print(f"W:NaN/Inf train out {model_id}")
                continue
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss): 
                print(f"W:NaN/Inf train loss {model_id}")
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            loop.set_postfix(loss=f"{loss.item():.4f}")
        except Exception as e:
             print(f"\nErr train batch {model_id}: {e}")
             traceback.print_exc() # See where the error occurs
             continue # Skip batch on error
    if num_samples == 0: 
        print(f"W:0 samples trained {model_id}")
        return 0.0
    return running_loss / num_samples

# --- Validation Function (Unchanged - handles HF model input) ---
def validate_one_epoch(model, loader, criterion, device, num_classes_val, model_id):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    num_samples = 0
    loop = tqdm(loader, desc=f"Validate Ep", leave=False)
    with torch.no_grad():
        for images, labels in loop:
            if images.nelement() == 0 or labels.nelement() == 0: 
                continue
            batch_size = images.size(0)
            if batch_size == 0: 
                continue
            images = images.to(device)
            labels = labels.to(device)
            try:
                # Check if model is likely a Hugging Face model
                if ConvNextV2ForImageClassification and isinstance(model, ConvNextV2ForImageClassification):
                     outputs = model(pixel_values=images).logits
                # Add elif for other specific HF types if needed
                else: # Assume torchvision or timm model
                     outputs = model(images)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any(): 
                    print(f"W:NaN/Inf val out {model_id}")
                    continue
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss): 
                    print(f"W:NaN/Inf val loss {model_id}")
                    continue
                running_loss += loss.item() * batch_size
                num_samples += batch_size
                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                loop.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                 print(f"\nErr val batch {model_id}: {e}")
                 traceback.print_exc()
                 continue # Skip batch on error

    if num_samples == 0: 
        print(f"W:0 samples validated {model_id}")
        return 0.0, float('nan')
    avg_loss = running_loss / num_samples
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probs)
    auc_score_val = float('nan')

    # Ensure probabilities are valid before calculating AUC
    if probs_np.ndim == 2 and probs_np.shape[0] == len(labels_np) and probs_np.shape[1] == num_classes_val:
        if len(np.unique(labels_np)) >= 2: # Need at least 2 classes present in the labels
            if not (np.isnan(probs_np).any() or np.isinf(probs_np).any()):
                try:
                    auc_score_val = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')
                except ValueError as e:
                    print(f"  AUC calculation error during validation {model_id}: {e}")
                    # This can happen if only one class is present in a batch or the validation set subset used so far
                    auc_score_val = float('nan') # Set to NaN on error
            else:
                 print(f"W: NaN/Inf detected in validation probabilities for {model_id}. AUC set to NaN.")
                 auc_score_val = float('nan')
        else:
             # print(f" Info: Less than 2 classes present in validation labels for {model_id}. AUC set to NaN.")
             auc_score_val = float('nan') # Cannot compute AUC with only one class
    else:
         print(f"W: Probability array shape mismatch or empty for {model_id}. Probs shape: {probs_np.shape}, Labels len: {len(labels_np)}. AUC set to NaN.")
         auc_score_val = float('nan')

    return avg_loss, auc_score_val


# --- Testing Function (Unchanged - handles HF model input) ---
def test_model(model, loader, device, num_classes_test, class_names_test, plots_dir, model_name_tag="custom_model"):
    """Note: Added plots_dir argument"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    safe_model_name_tag = model_name_tag.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '')
    print(f"\n--- Evaluating {model_name_tag} on Test Set ---")
    start_test_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing", leave=False):
            if images.nelement() == 0 or labels.nelement() == 0: 
                continue
            batch_size = images.size(0)
            if batch_size == 0: 
                continue
            images = images.to(device)
            labels = labels.to(device)
            try:
                # Check if model is likely a Hugging Face model
                if ConvNextV2ForImageClassification and isinstance(model, ConvNextV2ForImageClassification):
                     outputs = model(pixel_values=images).logits
                # Add elif for other specific HF types if needed
                else: # Assume torchvision or timm model
                     outputs = model(images)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any(): 
                    print(f"W:NaN/Inf test out {model_name_tag}")
                    continue
                probs = torch.softmax(outputs, dim=1)
                if torch.isnan(probs).any() or np.isinf(probs.cpu().numpy()).any(): 
                    print(f"W:NaN/Inf test prob {model_name_tag}")
                    continue # Check numpy for inf
                preds = torch.argmax(probs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            except Exception as e:
                 print(f"\nErr test batch {model_name_tag}: {e}")
                 traceback.print_exc()
                 continue # Skip batch on error

    end_test_time = time.time()
    print(f"Testing time: {end_test_time - start_test_time:.2f}s.")
    results = {'accuracy': float('nan'), 'auc_ovr_macro': float('nan'), 'report': 'N/A', 'cm': None, 'roc_curve_data': None}
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)

    if len(labels_np) == 0:
        print(f"W: No results generated during testing for {model_name_tag}")
        return results # Return empty results

    print("Calculating test metrics...")
    try:
        results['accuracy'] = np.mean(preds_np == labels_np)

        # AUC Calculation
        if len(np.unique(labels_np)) >= 2 and probs_np.ndim == 2 and probs_np.shape[0] == len(labels_np) and probs_np.shape[1] == num_classes_test:
             if not (np.isnan(probs_np).any() or np.isinf(probs_np).any()):
                try:
                    # Binarize labels for OVR AUC calculation
                    labels_bin = np.zeros((len(labels_np), num_classes_test))
                    valid_indices = (labels_np >= 0) & (labels_np < num_classes_test) # Ensure labels are within range
                    if np.any(valid_indices):
                         labels_bin[np.arange(len(labels_np))[valid_indices], labels_np[valid_indices]] = 1
                         # Check if all classes are present in the binarized labels for macro average
                         if np.all(np.sum(labels_bin, axis=0) > 0):
                              results['auc_ovr_macro'] = roc_auc_score(labels_bin, probs_np, multi_class='ovr', average='macro')
                         else:
                              print(f"  Warning: Not all classes present in test labels for {model_name_tag}. Cannot compute macro AUC.")
                              results['auc_ovr_macro'] = float('nan')
                    else:
                         print(f"  Warning: No valid labels found for AUC calculation in {model_name_tag}.")
                         results['auc_ovr_macro'] = float('nan')

                except ValueError as e:
                    print(f"  AUC calculation error during testing {model_name_tag}: {e}")
                    results['auc_ovr_macro'] = float('nan')
             else:
                  print(f"W: NaN/Inf detected in test probabilities for {model_name_tag}. AUC set to NaN.")
                  results['auc_ovr_macro'] = float('nan')
        else:
             print(f"W: Conditions for AUC calculation not met for {model_name_tag} (labels/probs shape/content). AUC set to NaN.")
             results['auc_ovr_macro'] = float('nan')


        # Classification Report & Confusion Matrix
        report_labels = list(range(num_classes_test))
        
        target_names = [class_names_test[i] if i < len(class_names_test) else f"Class_{i}" for i in report_labels]
        results['report'] = classification_report(labels_np, preds_np, target_names=target_names, labels=report_labels, zero_division=0)
        cm_raw = confusion_matrix(labels_np, preds_np, labels=report_labels)
        results['cm_list'] = cm_raw.tolist() # Store CM as list for JSON

    except Exception as e:
        print(f"Error calculating metrics for {model_name_tag}: {e}")
        traceback.print_exc()
        # Reset results on error
        results = {'accuracy': float('nan'), 'auc_ovr_macro': float('nan'), 'report': f'Error calculating metrics: {e}', 'cm_list': None, 'roc_curve_data': None}

    print("Generating plots...")
    # --- Plotting Confusion Matrix ---
    cm_list_plot = results.get('cm_list')
    safe_cm_path = os.path.join(plots_dir, f"confusion_matrix_{safe_model_name_tag}.png")
    if cm_list_plot is not None:
        try:
            cm_array = np.array(cm_list_plot)
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 8}) # Use target_names
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {model_name_tag}')
            plt.tight_layout()
            plt.savefig(safe_cm_path)
            print(f"  Saved CM: {safe_cm_path}")
            plt.close()
        except Exception as e:
            print(f"  Err plot CM: {e}")
            plt.close()

    # --- Plotting ROC Curve ---
    safe_roc_path = os.path.join(plots_dir, f"roc_curve_{safe_model_name_tag}.png")
    # Check if AUC could be calculated and probabilities are valid
    auc_value = results.get('auc_ovr_macro')
    if auc_value is not None and not np.isnan(auc_value) and not (np.isnan(probs_np).any() or np.isinf(probs_np).any()):
        try:
            results['roc_curve_data'] = {} # Store calculated data as lists
            plt.figure(figsize=(8, 7))
            colors = cycle(['blue', 'red', 'green'])
            plot_successful = False
            # Re-binarize labels for plotting each class curve
            labels_bin = np.zeros((len(labels_np), num_classes_test))
            valid_indices = (labels_np >= 0) & (labels_np < num_classes_test)
            if np.any(valid_indices): 
                labels_bin[np.arange(len(labels_np))[valid_indices], labels_np[valid_indices]] = 1

            for i, color in zip(range(num_classes_test), colors):
                # Check if the class has both positive and negative samples in the test set
                if len(np.unique(labels_bin[:, i])) == 2:
                    fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_np[:, i])
                    class_auc = auc(fpr, tpr)
                    if not np.isnan(class_auc):
                        results['roc_curve_data'][i] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': class_auc}
                        plt.plot(fpr, tpr, color=color, lw=2, label=f'{target_names[i]} (AUC={class_auc:.4f})')
                        plot_successful = True
                    else:
                         results['roc_curve_data'][i] = {'fpr': [], 'tpr': [], 'auc': float('nan')}
                         print(f"  Info: AUC for class {i} ({target_names[i]}) is NaN.")
                else:
                     results['roc_curve_data'][i] = {'fpr': [], 'tpr': [], 'auc': float('nan')}
                     print(f"  Info: Skipping ROC curve for class {i} ({target_names[i]}) - only one class present in labels.")

            if plot_successful:
                plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve (OVR) - {model_name_tag}')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(safe_roc_path)
                print(f"  Saved ROC: {safe_roc_path}")
                plt.close()
            else:
                 print("  Skip ROC plot: No valid curves generated.")
                 plt.close()
                 results['roc_curve_data'] = None # Clear if no curves plotted

        except Exception as e:
             print(f"  Err plot ROC: {e}")
             traceback.print_exc()
             results['roc_curve_data'] = None
             plt.close()
    else:
         print(f"  Skip ROC plot (Test AUC is NaN or probabilities invalid for {model_name_tag}).")
         results['roc_curve_data'] = None

    print(f"--- Eval Done: {model_name_tag} ---")
    # Return results dictionary (cm_list is already JSON serializable)
    results.pop('cm', None) # Ensure raw numpy array isn't returned if it existed
    return results


# --- Helper to clean up memory ---
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()