# main_controller.py

import os
import subprocess
import time
import sys
import traceback

# Import config and utils
import config
from lens_utils import load_results # Only need load_results here

def main():
    print("--- Starting Model Training Controller ---")
    print(f"Found {len(config.MODELS_TO_RUN)} models configured.")

    models_to_process = list(config.MODELS_TO_RUN.items())
    total_models = len(models_to_process)
    start_controller_time = time.time()

    for i, (model_desc, model_identifier) in enumerate(models_to_process):
        print("\n" + "="*20 + f" Processing Model {i+1}/{total_models}: {model_desc} " + "="*20)

        safe_model_desc = config.get_safe_filename(model_desc)
        result_file_path = os.path.join(config.RESULTS_DIR, f"results_{safe_model_desc}.json")
        status = "Not Started"
        error_info = None

        # Check if results file exists and load status
        if os.path.exists(result_file_path):
            print(f"Found existing result file: {result_file_path}")
            existing_result = load_results(result_file_path)
            if existing_result:
                status = existing_result.get('status', 'Unknown Status')
                error_info = existing_result.get('error', None)
            else:
                 print("  Warning: Could not load existing result file, will attempt to run again.")
                 status = "Load Failed"

        if status == 'Completed':
            print(f"Model '{model_desc}' already completed. Skipping.")
            continue
        elif status in ['Failed', 'Initialization Failed', 'Training Failed', 'Testing Failed']:
             print(f"Model '{model_desc}' previously failed (Status: {status}). Skipping.")
             if error_info: print(f"  Previous error: {error_info[:200]}...")
             # To retry failed runs, comment out the 'continue' line below.
             continue
        elif status in ['Starting', 'Initialized', 'Training', 'Testing', 'Testing Skipped (No Weights)', 'Testing Skipped (No Improvement)']:
             print(f"Model '{model_desc}' was previously interrupted (Status: {status}). Re-running.")
        else:
             print(f"Model '{model_desc}' status is '{status}'. Starting run.")

        # --- Construct command ---
        command = [
            sys.executable,
            "run_single_model.py",
            "--model_desc", model_desc,
            "--model_identifier", model_identifier,
        ]
        # Add batch size override dynamically if needed
        # Define which models might need smaller batch size on your system
        large_models_needing_override = ["ResNet101", "ResNet152", "EfficientNet-B4", "EfficientNet-B5", "ViT-Base-P16"]
        if model_desc in large_models_needing_override:
             # Choose a smaller batch size for these models
             override_batch_size = "4" # Or "64" etc.
             print(f"  Applying smaller batch size ({override_batch_size}) for potentially large model.")
             command.extend(["--batch_size", override_batch_size])
        else:
             # Use default from config for others
             command.extend(["--batch_size", str(config.BATCH_SIZE)])

        print(f"Executing command: {' '.join(command)}")
        start_model_time = time.time()
        process_return_code = -1 # Default to error state

        try:
            # --- Use Popen for real-time output ---
            # stdout=subprocess.PIPE: Capture standard output
            # stderr=subprocess.STDOUT: Redirect standard error to standard output
            # text=True: Decode output as text (UTF-8 by default)
            # bufsize=1: Use line buffering for more immediate output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True # Often used with text=True
            )

            # Read and print output line by line
            print(f"\n--- Live Output for {model_desc} ---")
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    # Process has finished and no more output
                    break
                if output_line:
                    # Print the line from the subprocess immediately
                    print(output_line.strip())

            # Get the final return code
            process_return_code = process.poll()
            print(f"--- End Live Output for {model_desc} ---")

            end_model_time = time.time()

            if process_return_code == 0:
                print(f"Script for '{model_desc}' finished successfully in {end_model_time - start_model_time:.2f}s.")
            else:
                # The error message should have been printed live via stderr redirection
                print(f"ERROR: Script for '{model_desc}' failed with exit code {process_return_code}.")
                # Note: The results json saved by run_single_model should reflect the failure

        except FileNotFoundError:
             print(f"ERROR: Could not find 'run_single_model.py'. Make sure it's in the same directory.")
             # Optionally save a failed status here if needed, though the loop will stop
             break
        except Exception as e:
            # Catch other potential errors during subprocess handling
            print(f"\nUNEXPECTED ERROR running subprocess for '{model_desc}': {e}")
            traceback.print_exc()
            # Consider marking this specific run as failed in a way, maybe by not having a "Completed" status in results json

        # Optional delay between models
        # print("Waiting 5s before next model...")
        # time.sleep(5)

    end_controller_time = time.time()
    print("\n" + "="*20 + " Controller Finished " + "="*20)
    print(f"Total controller runtime: {end_controller_time - start_controller_time:.2f}s ({ (end_controller_time - start_controller_time)/60:.2f} min)")
    print("Run 'python compare_results.py' to generate the final comparison.")

if __name__ == "__main__":
    main()