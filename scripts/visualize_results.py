import pickle
import matplotlib.pyplot as plt

# Load history from file
with open('models/training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Ensure the directory for saving the plot exists
save_path = 'training_history.png'

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Save the plot as an image
    plt.savefig(save_path)  # Save to file

plot_training_history(history)
