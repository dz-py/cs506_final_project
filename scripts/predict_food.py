import os
import tensorflow as tf
# Suppress TensorFlow INFO/WARNING messages (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARNING')

from tensorflow.keras.preprocessing import image # Use keras preprocessing
from tensorflow.keras.applications.resnet import preprocess_input # Use same preprocessing
import numpy as np
import pickle
import logging
import random

# =========================
# Configuration
# =========================
# Set how many random images to test from each category type
NUM_SAMPLES_PER_CLASS_TO_TEST = 3
# Set the confidence threshold below which prediction is considered "Other"
CONFIDENCE_THRESHOLD = 0.60 # e.g., 60% confidence required
# Name of the sample dataset directory containing unseen/unknown samples
UNSEEN_SAMPLE_DIR_NAME = "food-101-unseen-trained-plus-unknown-samples"
# Name of the 'unknown' class directory within the sample dataset
UNKNOWN_CLASS_DIR_NAME = "unknown"
# Name of the directory containing the actual train/val splits used for training
TRAIN_VAL_SPLIT_DIR_NAME = "dataset_split"
# Base directory where the filtered dataset (containing the split) resides
FILTERED_DATA_BASE_DIR_NAME = "food-101-filtered-5main-unknown"
# =========================

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Path Setup (MODIFIED)
# =========================
try:
    # Assumes the script is in a 'scripts' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively)
    script_dir = os.getcwd()
    if os.path.basename(script_dir).lower() == 'scripts':
         project_root = os.path.dirname(script_dir)
    else:
         project_root = script_dir
    logging.warning(f"Could not determine script directory reliably, assuming project root is: {project_root}")

# Model directory
model_dir = os.path.join(project_root, "models")
# Directory containing the unseen/unknown samples for testing
unseen_sample_dataset_dir = os.path.join(project_root, UNSEEN_SAMPLE_DIR_NAME)
# Directory containing the actual train/val splits used for training
split_base_dir = os.path.join(project_root, FILTERED_DATA_BASE_DIR_NAME, TRAIN_VAL_SPLIT_DIR_NAME)
# Specific path to the training data directory
train_dir_path = os.path.join(split_base_dir, "train")


# --- Define paths to the necessary files ---
# Use the best model saved by checkpoint
MODEL_PATH = os.path.join(model_dir, "best_food_classifier.keras")
CLASS_INDICES_PATH = os.path.join(model_dir, "class_indices.pkl") # Needed for name mapping

# --- Define image size (must match training) ---
IMG_SIZE = (224, 224)

logging.info(f"Model directory: {model_dir}")
logging.info(f"Unseen/Unknown sample dataset directory: {unseen_sample_dataset_dir}")
logging.info(f"Training data directory: {train_dir_path}")
logging.info(f"Class indices path: {CLASS_INDICES_PATH}")
logging.info(f"Model path to load: {MODEL_PATH}")

# =========================
# Load Model and Class Indices
# =========================
# --- Load the trained model ---
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at: {MODEL_PATH}")
    logging.error("Ensure train_model.py ran and saved the best model.")
    exit(1)
try:
    logging.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Use compile=False
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# --- Load the class indices ---
if not os.path.exists(CLASS_INDICES_PATH):
    logging.error(f"Class indices file not found at: {CLASS_INDICES_PATH}")
    logging.error("Ensure train_model.py ran and saved the class indices.")
    exit(1)
try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    # class_indices will map name -> index (e.g., {'beef_carpaccio': 0, ..., 'unknown': 5})
    # Create inverse mapping (index -> class name)
    class_names_map = {v: k for k, v in class_indices.items()}
    # Get the names of the classes the loaded model expects (should be 6)
    model_trained_classes = sorted(list(class_names_map.values()))
    # Specifically identify the main classes (excluding unknown)
    main_trained_classes = sorted([name for name in model_trained_classes if name != UNKNOWN_CLASS_DIR_NAME])
    logging.info(f"Loaded class indices. Model expects {len(model_trained_classes)} classes: {model_trained_classes}")
    logging.info(f"Identified {len(main_trained_classes)} main trained classes: {main_trained_classes}")
except Exception as e:
    logging.error(f"Error loading class indices: {e}")
    exit(1)

# =========================
# Find Classes in Sample Dataset (for unseen/unknown tests)
# =========================
classes_in_unseen_sample_dir = []
if not os.path.isdir(unseen_sample_dataset_dir):
    logging.warning(f"Unseen sample dataset directory not found: {unseen_sample_dataset_dir}")
    logging.warning("Skipping tests on unseen/unknown samples.")
else:
    try:
        # List all subdirectories (classes) within the unseen sample dataset directory
        classes_in_unseen_sample_dir = sorted([d for d in os.listdir(unseen_sample_dataset_dir)
                                       if os.path.isdir(os.path.join(unseen_sample_dataset_dir, d))])
        if not classes_in_unseen_sample_dir:
             logging.warning(f"No class subdirectories found in the unseen sample dataset directory: {unseen_sample_dataset_dir}")
        else:
             logging.info(f"Found classes in unseen sample dataset to test: {classes_in_unseen_sample_dir}")
    except OSError as e:
         logging.error(f"Error reading unseen sample dataset directory {unseen_sample_dataset_dir}: {e}")


# =========================
# Select Random Images Function
# =========================
def get_random_images_from_class(class_dir_path, num_to_select):
    """Selects a specified number of random image paths from a class directory."""
    selected_paths = []
    if not os.path.isdir(class_dir_path): # Add check if class dir exists
        logging.warning(f"Directory not found during image selection: {class_dir_path}")
        return []
    try:
        images = [f for f in os.listdir(class_dir_path)
                  if os.path.isfile(os.path.join(class_dir_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            logging.warning(f"No image files found in directory: {class_dir_path}")
            return []

        num_available = len(images)
        actual_num_to_select = min(num_available, num_to_select)

        if num_available < num_to_select:
            logging.warning(f"Directory '{os.path.basename(class_dir_path)}' only has {num_available} images, requested {num_to_select}. Selecting all available.")

        selected_filenames = random.sample(images, actual_num_to_select)
        selected_paths = [os.path.join(class_dir_path, fname) for fname in selected_filenames]

    except OSError as e:
        logging.warning(f"Error accessing directory {class_dir_path} during image selection: {e}")

    return selected_paths


# =========================
# Image Preprocessing Function
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
def predict_and_print(image_path, actual_class_name, image_type_label, threshold):
    """Preprocesses, predicts, and prints results for a single image, using a confidence threshold."""
    logging.info(f"--- Predicting {image_type_label} Image ---")
    logging.info(f"Image: {os.path.basename(image_path)} (Actual Class: {actual_class_name})")
    preprocessed_image = preprocess_single_image(image_path, IMG_SIZE)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image, verbose=0)
        predicted_index = np.argmax(predictions[0])
        # Use the loaded map corresponding to the *trained model*
        top_predicted_class_name = class_names_map.get(predicted_index, "ModelOutputIndexError")
        confidence_score = predictions[0][predicted_index]

        print(f"\nResults for {image_type_label} Image:")
        print(f"  File: {os.path.basename(image_path)}")
        print(f"  Actual Class: {actual_class_name}") # This is the folder name from the sample dataset

        if confidence_score >= threshold:
            print(f"  Predicted Class: {top_predicted_class_name}")
            print(f"  Confidence: {confidence_score:.2%}")
        else:
            print(f"  Predicted Class: Other (Below Threshold)")
            print(f"  Confidence: {confidence_score:.2%} (Threshold: {threshold:.0%})")
            print(f"  (Top prediction was: {top_predicted_class_name})")

        # Add specific notes based on the image_type_label
        if "Unknown Class" in image_type_label:
             print("  (Note: This image is from the 'unknown' sample category)")
        elif "Unseen Trained" in image_type_label:
             print("  (Note: Image from a trained class, but sample was unseen during training/validation)")
        elif "Seen Training" in image_type_label:
            print("  (Note: Image from the actual training set)")
        else:
             # Fallback for unexpected labels
             print(f"  (Note: Actual class '{actual_class_name}' tested)")


    else:
        print(f"\nCould not process {image_type_label} image: {image_path}")
    print("-" * 40) # Separator

# =========================
# Main Execution
# =========================
if __name__ == "__main__":

    print("\n" + "="*10 + f" TESTING IMAGES " + "="*10)
    print(f"Using model: {MODEL_PATH}")
    print(f"Confidence Threshold for 'Other': {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Testing {NUM_SAMPLES_PER_CLASS_TO_TEST} images per class found...")

    # --- 1. Test images the model was TRAINED on ---
    print("\n" + "="*10 + " CATEGORY 1: SEEN TRAINING SAMPLES " + "="*10)
    if not os.path.isdir(train_dir_path):
        logging.warning(f"Training directory not found ({train_dir_path}). Skipping tests on seen training samples.")
    else:
        # Iterate through the main classes the model was trained on (excluding unknown)
        for class_name in main_trained_classes:
            class_path = os.path.join(train_dir_path, class_name)
            logging.info(f"\nSelecting training images for class: {class_name}")
            images_to_test = get_random_images_from_class(class_path, NUM_SAMPLES_PER_CLASS_TO_TEST)
            if not images_to_test:
                logging.warning(f"No training images selected for class '{class_name}'. Skipping.")
                continue
            # Predict each selected image
            for img_path in images_to_test:
                predict_and_print(img_path, class_name, "Seen Training Sample", CONFIDENCE_THRESHOLD)

    # --- 2. Test UNSEEN images from TRAINED classes (using the separate sample dataset) ---
    print("\n" + "="*10 + " CATEGORY 2: UNSEEN TRAINED CLASS SAMPLES " + "="*10)
    if not classes_in_unseen_sample_dir:
         logging.warning("No classes found in unseen sample dataset directory. Skipping these tests.")
    else:
        # Iterate through the classes found in the unseen sample directory
        for class_name in classes_in_unseen_sample_dir:
            # Skip the 'unknown' class here, test it separately
            if class_name == UNKNOWN_CLASS_DIR_NAME:
                continue
            # Check if this class was actually one the model was trained on
            if class_name not in model_trained_classes:
                 logging.warning(f"Class '{class_name}' found in unseen sample dir, but not in model's trained classes. Skipping.")
                 continue

            class_path = os.path.join(unseen_sample_dataset_dir, class_name)
            logging.info(f"\nSelecting unseen images for trained class: {class_name}")
            images_to_test = get_random_images_from_class(class_path, NUM_SAMPLES_PER_CLASS_TO_TEST)
            if not images_to_test:
                logging.warning(f"No images selected for class '{class_name}' from unseen sample dir. Skipping.")
                continue
            # Predict each selected image
            for img_path in images_to_test:
                predict_and_print(img_path, class_name, "Unseen Trained Class Sample", CONFIDENCE_THRESHOLD)


    # --- 3. Test images from the UNKNOWN class sample ---
    print("\n" + "="*10 + " CATEGORY 3: UNKNOWN CLASS SAMPLES " + "="*10)
    unknown_class_path_in_sample = os.path.join(unseen_sample_dataset_dir, UNKNOWN_CLASS_DIR_NAME)
    if UNKNOWN_CLASS_DIR_NAME not in classes_in_unseen_sample_dir:
         logging.warning(f"Directory for '{UNKNOWN_CLASS_DIR_NAME}' not found in unseen sample dataset. Skipping these tests.")
    else:
        logging.info(f"\nSelecting images from the '{UNKNOWN_CLASS_DIR_NAME}' sample class...")
        images_to_test = get_random_images_from_class(unknown_class_path_in_sample, NUM_SAMPLES_PER_CLASS_TO_TEST)
        if not images_to_test:
            logging.warning(f"No images selected for class '{UNKNOWN_CLASS_DIR_NAME}' from unseen sample dir. Skipping.")
        else:
            # Predict each selected image
            for img_path in images_to_test:
                # The 'actual class' is 'unknown' for these samples
                predict_and_print(img_path, UNKNOWN_CLASS_DIR_NAME, "Unknown Class Sample", CONFIDENCE_THRESHOLD)


    logging.info("Prediction script finished.")

