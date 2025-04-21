# generate_history_plots.py

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
import traceback

# Import config and utils
import config
from lens_utils import load_results, NumpyEncoder # Need load_results

def generate_missing_history_plots():
    print("--- Generating Missing Training History Plots ---")

    # Ensure results and plots directories exist (though plots_dir might not be needed if saving to root)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Loop through all models defined in the config
    for model_desc, model_identifier in config.MODELS_TO_RUN.items():
        print(f"\nChecking model: {model_desc}")
        safe_model_desc = config.get_safe_filename(model_desc)
        result_file_path = os.path.join(config.RESULTS_DIR, f"results_{safe_model_desc}.json")
        hist_plot_path = os.path.join(config.PLOTS_DIR, f"training_history_{safe_model_desc}.png")

        # Check if result JSON exists
        if not os.path.exists(result_file_path):
            print(f"  Result file not found ({result_file_path}). Skipping plot generation.")
            continue

        # Load the result data
        result_data = load_results(result_file_path)
        if not result_data:
            print(f"  Failed to load result data from {result_file_path}. Skipping plot generation.")
            continue

        # Check if the run was completed (or at least finished training)
        status = result_data.get('status', 'Unknown')
        if status not in ['Completed', 'Training Completed', 'Testing Failed', 'Testing Skipped (No Weights)', 'Testing Skipped (No Improvement)']:
             print(f"  Model status is '{status}'. Skipping plot generation.")
             continue

        # Check if the history plot ALREADY exists
        if os.path.exists(hist_plot_path):
            print(f"  History plot already exists ({hist_plot_path}). Skipping.")
            continue

        # --- Try to generate the plot ---
        print(f"  History plot missing. Attempting to generate from results...")
        train_losses = result_data.get('train_losses')
        val_losses = result_data.get('val_losses')
        val_aucs = result_data.get('val_aucs')
        best_ep_num_hist = result_data.get('best_epoch', -1)
        best_val_auc_score_hist = result_data.get('best_val_auc', float('nan'))

        # Validate data needed for plotting
        if train_losses and val_losses and val_aucs and \
           len(train_losses) > 0 and \
           len(train_losses) == len(val_losses) == len(val_aucs):
            try:
                epochs_range = range(1, len(train_losses) + 1)
                plt.figure(figsize=(14, 6))
                plt.style.use('ggplot')

                # Loss Plot
                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, train_losses, label='Train Loss', marker='.', linestyle='-', ms=4)
                plt.plot(epochs_range, val_losses, label='Val Loss', marker='.', linestyle='-', ms=4)
                plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title(f'{model_desc} - Loss');
                plt.legend(); plt.grid(True); plt.ylim(bottom=0)

                # AUC Plot
                plt.subplot(1, 2, 2)
                val_auc_series = pd.Series(val_aucs, index=epochs_range).dropna()
                if not val_auc_series.empty:
                    plt.plot(val_auc_series.index, val_auc_series.values, label='Val AUC (Macro OVR)', marker='.', linestyle='-', ms=4, color='green')
                if best_ep_num_hist != -1 and not np.isnan(best_val_auc_score_hist) and best_ep_num_hist in val_auc_series.index:
                     if np.isclose(val_auc_series.loc[best_ep_num_hist], best_val_auc_score_hist):
                          plt.scatter([best_ep_num_hist], [best_val_auc_score_hist], color='red', s=100, label=f'Best (Ep {best_ep_num_hist}, AUC {best_val_auc_score_hist:.4f})', zorder=5)
                plt.xlabel('Epochs'); plt.ylabel('AUC Score'); plt.title(f'{model_desc} - Val AUC');
                plt.legend(); plt.grid(True); plt.ylim(0.5, 1.02)

                plt.suptitle(f"Training History: {model_desc}", fontsize=14);
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(hist_plot_path)
                print(f"  ✅ Successfully generated and saved history plot to {hist_plot_path}.")
                plt.close()

            except Exception as e_hist:
                print(f"  ERROR generating history plot for {model_desc}: {e_hist}")
                traceback.print_exc()
                plt.close()
        else:
            print(f"  Skipping history plot for {model_desc} (required data missing or inconsistent lengths in JSON).")

    print("\n--- Finished generating missing plots ---")

if __name__ == "__main__":
    generate_missing_history_plots()