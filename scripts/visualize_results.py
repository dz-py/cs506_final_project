import pickle
import matplotlib.pyplot as plt

# Load history from file
with open('models/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Ensure the directory for saving the plot exists
save_path = 'training_history.png'

print(history.keys())  # Ensure it has ['accuracy', 'val_accuracy', 'loss', 'val_loss']
print(history['accuracy'])
print(len(history['accuracy']), len(history['val_accuracy']))
print(len(history['loss']), len(history['val_loss']))


epochs_range = range(1, len(history['accuracy']) + 1)  # Start from 1

def plot_training_history(history):
    epochs_range = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['accuracy'], marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], marker='s', linestyle='--', label='Val Accuracy', color='red')
    plt.xticks(epochs_range)  # Force x-axis to show all epochs
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['loss'], marker='o', linestyle='-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], marker='s', linestyle='--', label='Val Loss', color='red')
    plt.xticks(epochs_range)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Loss')


    plt.savefig(save_path)

plot_training_history(history)