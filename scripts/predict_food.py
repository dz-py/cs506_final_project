import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # Use keras preprocessing
from tensorflow.keras.applications.resnet import preprocess_input # Use same preprocessing
import numpy as np
import pickle
import logging
import random # Added for random selection

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Path Setup
# =========================
# Determine paths relative to this script's location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
    logging.warning(f"Could not determine script directory, using current working directory: {script_dir}")

project_root = os.path.dirname(script_dir) # Assumes script is in 'scripts' folder
model_dir = os.path.join(project_root, "models")
original_images_dir = os.path.join(project_root, "food-101", "images") # Path to original images
processed_images_dir = os.path.join(project_root, "food-101", "images_processed") # Used to find trained classes

# --- Define paths to the necessary files ---
MODEL_PATH = os.path.join(model_dir, "food_classifier_final.keras")
CLASS_INDICES_PATH = os.path.join(model_dir, "class_indices.pkl")

# --- Define image size (must match training) ---
IMG_SIZE = (224, 224)

# =========================
# Load Model and Class Indices
# =========================

# --- Load the trained model ---
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at: {MODEL_PATH}")
    exit()
try:
    logging.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

# --- Load the class indices ---
if not os.path.exists(CLASS_INDICES_PATH):
    logging.error(f"Class indices file not found at: {CLASS_INDICES_PATH}")
    exit()
try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    # Create inverse mapping (index -> class name)
    class_names_map = {v: k for k, v in class_indices.items()}
    trained_class_names = list(class_names_map.values()) # Get list of names model was trained on
    logging.info(f"Loaded class indices. Model trained on {len(trained_class_names)} classes: {trained_class_names}")
except Exception as e:
    logging.error(f"Error loading class indices: {e}")
    exit()

# =========================
# Find Trained vs Untrained Classes
# =========================

if not os.path.isdir(original_images_dir):
    logging.error(f"Original images directory not found at: {original_images_dir}")
    exit()

# Get all classes from the original directory
try:
    all_original_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])
    if not all_original_classes:
         logging.error(f"No class directories found in {original_images_dir}")
         exit()
except OSError as e:
     logging.error(f"Error reading original images directory {original_images_dir}: {e}")
     exit()


# Determine untrained classes
untrained_class_names = sorted(list(set(all_original_classes) - set(trained_class_names)))

if not untrained_class_names:
    logging.warning("Could not find any classes in the original dataset that the model wasn't trained on.")
    # Decide how to handle this - exit or only test trained? We'll exit for now.
    exit()

logging.info(f"Found {len(untrained_class_names)} classes the model was not trained on (example: {untrained_class_names[0]}).")


# =========================
# Select Random Images
# =========================

def get_random_image_path(base_dir, selected_classes):
    """Selects a random class from the list and a random image from that class."""
    if not selected_classes:
        return None, None
    chosen_class = random.choice(selected_classes)
    class_path = os.path.join(base_dir, chosen_class)
    try:
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            logging.warning(f"No image files found in directory: {class_path}")
            return None, None
        chosen_image = random.choice(images)
        return os.path.join(class_path, chosen_image), chosen_class
    except OSError as e:
        logging.error(f"Error accessing directory {class_path}: {e}")
        return None, None

# --- Select one image from a trained class ---
trained_image_path, trained_actual_class = get_random_image_path(original_images_dir, trained_class_names)

# --- Select one image from an untrained class ---
untrained_image_path, untrained_actual_class = get_random_image_path(original_images_dir, untrained_class_names)

if not trained_image_path:
    logging.error("Failed to select a random image from trained classes.")
    exit()
if not untrained_image_path:
    logging.error("Failed to select a random image from untrained classes.")
    exit()


# =========================
# Image Preprocessing Function (same as before)
# =========================
def preprocess_single_image(img_path, target_size):
    """Loads and preprocesses a single image for model prediction."""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        return img_preprocessed
    except FileNotFoundError:
        logging.error(f"Image file not found at: {img_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

# =========================
# Prediction Function
# =========================
def predict_and_print(image_path, actual_class_name, image_type_label):
    """Preprocesses, predicts, and prints results for a single image."""
    logging.info(f"--- Predicting {image_type_label} Image ---")
    logging.info(f"Selected Image: {os.path.basename(image_path)} (Actual Class: {actual_class_name})")
    preprocessed_image = preprocess_single_image(image_path, IMG_SIZE)

    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        predicted_index = np.argmax(predictions[0])
        predicted_class_name = class_names_map.get(predicted_index, "Unknown Class")
        confidence_score = predictions[0][predicted_index]

        print(f"\nResults for {image_type_label} Image:")
        print(f"  File: {os.path.basename(image_path)}")
        print(f"  Actual Class: {actual_class_name}")
        print(f"  Predicted Class: {predicted_class_name}")
        print(f"  Confidence: {confidence_score:.4f}")
        if image_type_label == "Untrained Class":
             print("  (Note: Model was not trained on this class, prediction indicates the closest known class)")
    else:
        print(f"\nCould not process {image_type_label} image: {image_path}")
    print("-" * (len(image_type_label) + 24)) # Separator

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Predict the image from a trained class
    predict_and_print(trained_image_path, trained_actual_class, "Trained Class")

    # Predict the image from an untrained class
    predict_and_print(untrained_image_path, untrained_actual_class, "Untrained Class")

    logging.info("Prediction script finished.")