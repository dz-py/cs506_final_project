import os
import pickle
import matplotlib.pyplot as plt
import numpy as np # Import numpy for advanced tick handling if needed
import logging

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO)

# =========================
# Configuration
# =========================
# Number of epochs used in the initial training phase (before fine-tuning).
# Adjust this to match your training script!
INITIAL_TRAINING_EPOCHS = 10

# =========================
# Directory and Path Setup
# =========================
try:
    # If running as a script, get script directory (assumes 'scripts' folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    # Fallback for interactive environments (e.g., Jupyter)
    project_root = os.getcwd()
    logging.warning(f"Could not determine script directory, using current working directory: {project_root}")

model_dir = os.path.join(project_root, "models")
history_file_path = os.path.join(model_dir, 'training_history.pkl')
plot_save_path = os.path.join(model_dir, 'training_history.png')

logging.info(f"Looking for history file at: {history_file_path}")

# =========================
# Load Training History
# =========================
if not os.path.exists(history_file_path):
    logging.error(f"History file not found: {history_file_path}")
    logging.error("Please ensure train_model.py ran successfully and saved the history.")
    exit() # Exit if the file doesn't exist

with open(history_file_path, 'rb') as f:
    history = pickle.load(f)
logging.info("Training history loaded successfully.")

# =========================
# Validate History Content
# =========================
# Ensure history has the expected structure and keys
if not history or 'accuracy' not in history or not history['accuracy']:
    logging.error("Loaded history is empty or missing 'accuracy' key.")
    exit()

total_epochs_ran = len(history['accuracy'])
epochs_range = range(1, total_epochs_ran + 1)
logging.info(f"Total epochs recorded in history: {total_epochs_ran}")

# =========================
# Plotting Function
# =========================
def plot_training_history(history_dict, epoch_axis, fine_tune_start_epoch=None, save_path=None):
    """
    Plots training and validation accuracy and loss curves.
    Optionally marks the fine-tuning phase start epoch.

    Args:
        history_dict (dict): Contains 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
        epoch_axis (range): Epoch numbers for the x-axis.
        fine_tune_start_epoch (int, optional): Epoch after which fine-tuning started.
        save_path (str, optional): If provided, saves the plot image to this path.
    """
    # Ensure required keys are present
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    if not all(key in history_dict for key in required_keys):
        logging.error(f"History dictionary is missing one or more required keys: {required_keys}")
        return

    plt.figure(figsize=(14, 6)) # Set figure size for clarity

    # --- Accuracy subplot ---
    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, history_dict['accuracy'], marker='.', linestyle='-', label='Train Accuracy')
    plt.plot(epoch_axis, history_dict['val_accuracy'], marker='.', linestyle='--', label='Val Accuracy')
    # Optionally, mark the fine-tuning start epoch with a vertical line
    ft_line = None
    if fine_tune_start_epoch is not None and fine_tune_start_epoch < len(epoch_axis):
        ft_line = plt.axvline(x=fine_tune_start_epoch + 0.5, color='grey', linestyle=':', linewidth=2, label='Fine-Tune Start')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # Ensure the fine-tune line label appears only once
    handles, labels = plt.gca().get_legend_handles_labels()
    if ft_line and "Fine-Tune Start" not in labels:
        handles.append(ft_line)
        labels.append("Fine-Tune Start")
    plt.legend(handles=handles, labels=labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Accuracy')
    # Set x-ticks for readability (show ~10 ticks)
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis)+1, step=max(1, len(epoch_axis)//10)))
    plt.tight_layout()

    # --- Loss subplot ---
    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, history_dict['loss'], marker='.', linestyle='-', label='Train Loss')
    plt.plot(epoch_axis, history_dict['val_loss'], marker='.', linestyle='--', label='Val Loss')
    # Optionally, mark the fine-tuning start epoch with a vertical line
    ft_line = None
    if fine_tune_start_epoch is not None and fine_tune_start_epoch < len(epoch_axis):
        ft_line = plt.axvline(x=fine_tune_start_epoch + 0.5, color='grey', linestyle=':', linewidth=2, label='Fine-Tune Start')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    handles, labels = plt.gca().get_legend_handles_labels()
    if ft_line and "Fine-Tune Start" not in labels:
        handles.append(ft_line)
        labels.append("Fine-Tune Start")
    plt.legend(handles=handles, labels=labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Loss')
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis)+1, step=max(1, len(epoch_axis)//10)))
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        try:
            plt.savefig(save_path)
            logging.info(f"Plot saved successfully to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot to {save_path}: {e}")

    plt.show() # Display the plot

# =========================
# Plot the Training History
# =========================
# Pass the intended end epoch of the initial phase to mark fine-tuning start
plot_training_history(history, epochs_range, fine_tune_start_epoch=INITIAL_TRAINING_EPOCHS, save_path=plot_save_path)
