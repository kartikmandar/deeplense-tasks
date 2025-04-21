# compare_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sns
import json # For loading results
import traceback

# Import config and utils
import config
from lens_utils import load_results # Only need load_results

def generate_comparison():
    print("--- Generating Final Comparison Report ---")
    all_results_data = []

    # Find all result files in the results directory
    result_files = glob.glob(os.path.join(config.RESULTS_DIR, "results_*.json"))

    if not result_files:
        print(f"ERROR: No result files found in '{config.RESULTS_DIR}'.")
        print("Please run 'main_controller.py' first to generate model results.")
        return

    print(f"Found {len(result_files)} result files.")

    # Load data from each result file
    for f_path in result_files:
        result = load_results(f_path)
        if result:
            # Extract relevant summary info
            summary = {
                'Model': result.get('model_desc', os.path.basename(f_path)), # Use filename as fallback
                'Status': result.get('status', 'Unknown'),
                'Best Epoch': result.get('best_epoch', -1),
                'Best Val AUC': result.get('best_val_auc', float('nan')),
                # Extract test metrics from the nested 'test_results' dictionary
                'Test Accuracy': result.get('test_results', {}).get('accuracy', float('nan')),
                'Test AUC (OVR Macro)': result.get('test_results', {}).get('auc_ovr_macro', float('nan')),
                'Training Time (min)': result.get('training_time_s', 0) / 60.0,
                # Store the path for later retrieval of full results if needed
                'Result File': f_path
            }
            all_results_data.append(summary)
        else:
            print(f"  Warning: Failed to load results from {f_path}")

    if not all_results_data:
        print("ERROR: Failed to load any valid results. Cannot generate comparison.")
        return

    # --- Create DataFrame ---
    results_df = pd.DataFrame(all_results_data)
    results_df = results_df.round(4)
    if 'Model' in results_df.columns:
        results_df.set_index('Model', inplace=True)
    else:
         print("Warning: 'Model' column missing after loading results.")


    print("\n--- Summary Comparison Table ---")
    try:
        print(results_df.to_markdown())
    except ImportError:
         print("Note: Install 'tabulate' library (`pip install tabulate`) for markdown table output.")
         print(results_df)

    # --- Save Summary CSV ---
    try:
        results_df.to_csv(config.FINAL_RESULTS_FILE)
        print(f"\nFinal comparison summary saved to {config.FINAL_RESULTS_FILE}")
    except Exception as e_csv:
        print(f"\nError saving final results CSV: {e_csv}")

    # --- Identify Best Model ---
    best_model_name = None
    best_model_result_file = None
    completed_runs_df = results_df[results_df['Status'] == 'Completed'].copy()
    completed_runs_df['Test AUC (OVR Macro)'] = pd.to_numeric(completed_runs_df['Test AUC (OVR Macro)'], errors='coerce')
    valid_auc_runs_df = completed_runs_df.dropna(subset=['Test AUC (OVR Macro)'])

    if not valid_auc_runs_df.empty:
         best_model_name = valid_auc_runs_df['Test AUC (OVR Macro)'].idxmax()
         print(f"\n🏆 Best Model based on Test AUC (OVR Macro): {best_model_name}")
         print("\n--- Details for Best Model ---")
         print(results_df.loc[best_model_name])
         # Store the result file path of the best model
         best_model_result_file = results_df.loc[best_model_name, 'Result File']
    else:
         print("\nCould not determine the best model (No completed runs with valid Test AUC).")


    # --- Plotting Comparisons ---
    print("\n--- Saving Comparison Plots ---")
    plots_saved = False
    if not completed_runs_df.empty:
        plt.style.use('ggplot')
        save_dir = config.PLOTS_DIR # Save plots to specified dir (root in this case)

        # Plot 1: Test AUC vs. Model
        try:
            plt.figure(figsize=(16, 8))
            plot_data = valid_auc_runs_df['Test AUC (OVR Macro)'].sort_values(ascending=False)
            if not plot_data.empty:
                plot_data.plot(kind='bar', color='skyblue')
                plt.title('Test AUC (OVR Macro) Comparison (Completed Runs)')
                plt.ylabel('AUC Score'); plt.xlabel('Model')
                plt.xticks(rotation=70, ha='right', fontsize=9)
                plt.grid(axis='y', linestyle='--'); plt.tight_layout()
                save_path = os.path.join(save_dir, "comparison_plot_test_auc.png")
                plt.savefig(save_path); print(f"  Saved: {save_path}"); plots_saved = True
            else: print("  Skipping Test AUC plot (no valid data).")
            plt.close()
        except Exception as e: print(f"  Error plotting Test AUC: {e}"); plt.close()

        # Plot 2: Test Accuracy vs. Model
        try:
            plt.figure(figsize=(16, 8))
            completed_runs_df['Test Accuracy'] = pd.to_numeric(completed_runs_df['Test Accuracy'], errors='coerce')
            plot_data = completed_runs_df.dropna(subset=['Test Accuracy'])['Test Accuracy'].sort_values(ascending=False)
            if not plot_data.empty:
                plot_data.plot(kind='bar', color='lightcoral')
                plt.title('Test Accuracy Comparison (Completed Runs)')
                plt.ylabel('Accuracy'); plt.xlabel('Model')
                plt.xticks(rotation=70, ha='right', fontsize=9)
                plt.grid(axis='y', linestyle='--'); plt.tight_layout()
                save_path = os.path.join(save_dir, "comparison_plot_test_accuracy.png")
                plt.savefig(save_path); print(f"  Saved: {save_path}"); plots_saved = True
            else: print("  Skipping Test Accuracy plot (no valid data).")
            plt.close()
        except Exception as e: print(f"  Error plotting Test Accuracy: {e}"); plt.close()

        # Plot 3: Training Time vs. Model
        try:
            plt.figure(figsize=(16, 8))
            plot_data = completed_runs_df['Training Time (min)'].sort_values()
            if not plot_data.empty:
                plot_data.plot(kind='bar', color='lightgreen')
                plt.title('Training Time Comparison (Completed Runs)')
                plt.ylabel('Time (minutes)'); plt.xlabel('Model')
                plt.xticks(rotation=70, ha='right', fontsize=9)
                plt.grid(axis='y', linestyle='--'); plt.tight_layout()
                save_path = os.path.join(save_dir, "comparison_plot_training_time.png")
                plt.savefig(save_path); print(f"  Saved: {save_path}"); plots_saved = True
            else: print("  Skipping Training Time plot (no valid data).")
            plt.close()
        except Exception as e: print(f"  Error plotting Training Time: {e}"); plt.close()

        # Plot 4: Test AUC vs Training Time (Scatter)
        try:
            plt.figure(figsize=(12, 8))
            plot_data = valid_auc_runs_df.dropna(subset=['Test AUC (OVR Macro)', 'Training Time (min)'])
            if not plot_data.empty:
                plt.scatter(plot_data['Training Time (min)'], plot_data['Test AUC (OVR Macro)'], alpha=0.7, s=50)
                for model_nm in plot_data.index:
                    plt.text(plot_data.loc[model_nm, 'Training Time (min)'] * 1.01, plot_data.loc[model_nm, 'Test AUC (OVR Macro)'], model_nm, fontsize=9)
                plt.title('Test AUC vs. Training Time (Completed Runs with Valid AUC)')
                plt.xlabel('Training Time (minutes)'); plt.ylabel('Test AUC (OVR Macro)')
                plt.grid(True, linestyle='--'); plt.tight_layout()
                save_path = os.path.join(save_dir, "comparison_plot_auc_vs_time.png")
                plt.savefig(save_path); print(f"  Saved: {save_path}"); plots_saved = True
            else: print("  Skipping AUC vs Time plot (no valid data points).")
            plt.close()
        except Exception as e: print(f"  Error plotting AUC vs Time: {e}"); plt.close()

    if not plots_saved:
         print("No comparison plots were generated (likely no completed runs with valid data).")


    # --- Display Best Model's Details (from its individual result file) ---
    if best_model_name and best_model_result_file:
        print(f"\n--- Detailed Report/Plots for Best Model ({best_model_name}) ---")
        best_result_data = load_results(best_model_result_file)
        if best_result_data:
            print("\nTest Report:")
            report = best_result_data.get('test_results', {}).get('report', 'Report not available.')
            print(report if isinstance(report, str) else "Report data invalid.")

            # Check for the individual plots saved by test_model
            safe_best_model_desc = config.get_safe_filename(best_model_name)
            cm_path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe_best_model_desc}.png")
            roc_path = os.path.join(config.PLOTS_DIR, f"roc_curve_{safe_best_model_desc}.png")
            # Construct history plot path (assuming it might be saved by run_single_model eventually or compare_results)
            hist_path = os.path.join(config.PLOTS_DIR, f"training_history_{safe_best_model_desc}.png")

            if os.path.exists(cm_path): print(f"\nConfusion Matrix plot exists at: {cm_path}")
            else: print(f"\nConfusion Matrix plot for {best_model_name} NOT found.")

            if os.path.exists(roc_path): print(f"ROC Curve plot exists at: {roc_path}")
            else: print(f"ROC Curve plot for {best_model_name} NOT found.")

            # Plot Training History for the Best Model (using loaded detailed results)
            train_losses = best_result_data.get('train_losses')
            val_losses = best_result_data.get('val_losses')
            val_aucs = best_result_data.get('val_aucs') # Contains NaNs

            if train_losses and val_losses and val_aucs and \
               len(train_losses) == len(val_losses) == len(val_aucs):
                try:
                    print(f"\n--- Saving Training History Plot for Best Model ({best_model_name}) ---")
                    epochs_range = range(1, len(train_losses) + 1)
                    plt.figure(figsize=(14, 6))
                    plt.subplot(1, 2, 1) # Loss
                    plt.plot(epochs_range, train_losses, label='Train Loss', marker='.', linestyle='-', ms=4)
                    plt.plot(epochs_range, val_losses, label='Val Loss', marker='.', linestyle='-', ms=4)
                    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title(f'{best_model_name} - Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
                    plt.subplot(1, 2, 2) # AUC
                    val_auc_series = pd.Series(val_aucs, index=epochs_range).dropna()
                    if not val_auc_series.empty: plt.plot(val_auc_series.index, val_auc_series.values, label='Val AUC (Macro OVR)', marker='.', linestyle='-', ms=4, color='g')
                    best_ep = best_result_data.get('best_epoch', -1)
                    best_auc = best_result_data.get('best_val_auc', float('nan'))
                    if best_ep != -1 and not np.isnan(best_auc) and best_ep in val_auc_series.index and np.isclose(val_auc_series.loc[best_ep], best_auc):
                         plt.scatter([best_ep], [best_auc], color='red', s=100, label=f'Best (Ep {best_ep}, AUC {best_auc:.4f})', zorder=5)
                    plt.xlabel('Epochs'); plt.ylabel('AUC Score'); plt.title(f'{best_model_name} - Val AUC'); plt.legend(); plt.grid(True); plt.ylim(0.5, 1.02)
                    plt.suptitle(f"Training History: {best_model_name}", fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    save_path = os.path.join(config.PLOTS_DIR, f"training_history_{safe_best_model_desc}.png")
                    plt.savefig(save_path); print(f"  Saved training history plot to {save_path}"); plt.close()
                except Exception as e_hist: print(f"  Error plotting training history: {e_hist}"); plt.close()
            else: print("  Training history data incomplete/missing in results file.")
        else:
             print(f"Could not load detailed results for best model from: {best_model_result_file}")

    print("\nComparison script finished.")

if __name__ == "__main__":
    generate_comparison()