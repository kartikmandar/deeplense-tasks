# config.py
import torch
import os

# --- Paths ---
DATA_ROOT_ORIGINAL = 'dataset'          # Pre-existing dataset folder
WEIGHTS_DIR = "model_weights_task1_comparison" # Where to save best weights per model
RESULTS_DIR = "model_results_task1_comparison" # Where to save individual JSON results per model
FINAL_RESULTS_FILE = "model_comparison_task1_results_final.csv" # Final aggregated results
PLOTS_DIR = "." # Save plots in the root directory for simplicity

# --- Data & Splitting ---
NUM_CLASSES = 3
CLASSES = ['no', 'sphere', 'vort']
SEED = 42
VAL_SPLIT_RATIO = 0.5 # Ratio of original 'val' set to use for the new validation set (rest becomes test)

# --- Execution Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Start with a conservative batch size, can be overridden per model if needed
# BATCH_SIZE = 128 # Reduced default
BATCH_SIZE = 8 # Even more conservative start for broader compatibility
try:
    num_cores = os.cpu_count()
    # Heuristic: Use slightly fewer than half the threads for data loading
    NUM_WORKERS = max(1, num_cores // 2 - 2 if num_cores and num_cores > 4 else (num_cores // 2 if num_cores else 1))
except:
    NUM_WORKERS = 4 # Fallback
PIN_MEMORY = False

# --- Training Hyperparameters (Defaults - can be overridden) ---
EPOCHS = 15
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
# Patience for ReduceLROnPlateau scheduler
SCHEDULER_PATIENCE = 3
# Factor for ReduceLROnPlateau scheduler
SCHEDULER_FACTOR = 0.2

# --- Image Preprocessing ---
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# --- Models to Run ---
# This dictionary defines the models the controller will attempt to run.
# Key: User-friendly description (used for filenames)
# Value: Model identifier string (used by get_model in lens_utils)
MODELS_TO_RUN = {
    # ResNets (Torchvision)
    "ResNet18":             "resnet18",
    "ResNet34":             "resnet34",
    "ResNet50":             "resnet50",
    "ResNet101":            "resnet101",
    "ResNet152":            "resnet152",

    # EfficientNets (Timm)
    "EfficientNet-B0":      "efficientnet_b0",
    "EfficientNet-B1":      "efficientnet_b1",
    "EfficientNet-B2":      "efficientnet_b2",
    "EfficientNet-B3":      "efficientnet_b3",
    "EfficientNet-B4":      "efficientnet_b4",
    "EfficientNet-B5":      "efficientnet_b5",

    # MobileNetV3 (Timm)
    "MobileNetV3-Small":    "mobilenetv3_small_100",
    "MobileNetV3-Large":    "mobilenetv3_large_100",

    # ConvNeXtV2 (Hugging Face Transformers) - 1k
    "ConvNeXtV2-Atto-1k":  "facebook/convnextv2-atto-1k-224",
    "ConvNeXtV2-Femto-1k": "facebook/convnextv2-femto-1k-224",
    "ConvNeXtV2-Pico-1k":  "facebook/convnextv2-pico-1k-224",
    "ConvNeXtV2-Nano-1k":  "facebook/convnextv2-nano-1k-224",
    "ConvNeXtV2-Tiny-1k":  "facebook/convnextv2-tiny-1k-224",
    # "ConvNeXtV2-Base-1k":  "facebook/convnextv2-base-1k-224",

    # ConvNeXtV2 (Hugging Face Transformers) - 22k
    "ConvNeXtV2-Nano-22k": "facebook/convnextv2-nano-22k-224",
    "ConvNeXtV2-Tiny-22k": "facebook/convnextv2-tiny-22k-224",
    # "ConvNeXtV2-Base-22k": "facebook/convnextv2-base-22k-224",

    # Vision Transformers (Timm)
    "ViT-Tiny-P16":        "vit_tiny_patch16_224",
    "ViT-Small-P16":       "vit_small_patch16_224",
    "ViT-Base-P16":        "vit_base_patch16_224",
}

# --- Helper Functions ---
def get_safe_filename(name):
    """Creates a filesystem-safe filename from a model description."""
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '')