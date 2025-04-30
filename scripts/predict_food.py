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
import random # Added for random selection
# Removed time import if it was only for seeding

# =========================
# Configuration
# =========================
# Set how many random images to test for each category type
NUM_SAMPLES_PER_TYPE = 3
# Set the confidence threshold below which prediction is considered "Other"
CONFIDENCE_THRESHOLD = 0.60 # e.g., 60% confidence required
# --- Removed fixed RANDOM_SEED ---
# =========================

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
split_base_dir = os.path.join(project_root, "food-101", "dataset_split") # Path to train/val splits

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
    exit(1)
try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    class_names_map = {v: k for k, v in class_indices.items()}
    trained_class_names = sorted(list(class_names_map.values()))
    logging.info(f"Loaded class indices. Model trained on {len(trained_class_names)} classes: {trained_class_names}")
except Exception as e:
    logging.error(f"Error loading class indices: {e}")
    exit(1)

# =========================
# Find Trained, Untrained, and Unseen Images
# =========================
# (Logic remains the same as previous version)
if not os.path.isdir(original_images_dir):
    logging.error(f"Original images directory not found at: {original_images_dir}")
    exit(1)
if not os.path.isdir(split_base_dir):
     logging.error(f"Split dataset directory not found at: {split_base_dir}")
     exit(1)
try:
    all_original_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])
    if not all_original_classes:
         logging.error(f"No class directories found in {original_images_dir}")
         exit(1)
except OSError as e:
     logging.error(f"Error reading original images directory {original_images_dir}: {e}")
     exit(1)
untrained_class_names = sorted(list(set(all_original_classes) - set(trained_class_names)))
logging.info(f"Found {len(untrained_class_names)} classes the model was not trained on (example: {untrained_class_names[0] if untrained_class_names else 'N/A'}).")
unseen_images_by_class = {}
logging.info("Identifying unseen images within trained classes...")
for class_name in trained_class_names:
    original_class_path = os.path.join(original_images_dir, class_name)
    train_class_path = os.path.join(split_base_dir, "train", class_name)
    val_class_path = os.path.join(split_base_dir, "val", class_name)
    try:
        original_files = set(f for f in os.listdir(original_class_path) if os.path.isfile(os.path.join(original_class_path, f)))
        seen_files = set()
        if os.path.isdir(train_class_path):
            seen_files.update(f for f in os.listdir(train_class_path) if os.path.isfile(os.path.join(train_class_path, f)))
        if os.path.isdir(val_class_path):
            seen_files.update(f for f in os.listdir(val_class_path) if os.path.isfile(os.path.join(val_class_path, f)))
        unseen_files = list(original_files - seen_files)
        if unseen_files:
            unseen_images_by_class[class_name] = unseen_files
        else:
             logging.warning(f"No unseen images found for trained class: {class_name}")
    except FileNotFoundError:
        logging.warning(f"Could not find original or split directory for class: {class_name}. Skipping unseen check for this class.")
    except OSError as e:
        logging.error(f"Error processing files for class {class_name}: {e}")
if not unseen_images_by_class:
    logging.warning("Could not identify any unseen images for any trained class.")

# =========================
# Select Multiple Random Images for Testing
# =========================
# (Helper functions get_random_image_path and get_random_unseen_image_path remain the same)
def get_random_image_path(base_dir, selected_classes):
    if not selected_classes:
        logging.warning(f"Cannot select image, no classes provided in the list: {selected_classes}")
        return None, None
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        chosen_class = random.choice(selected_classes)
        class_path = os.path.join(base_dir, chosen_class)
        try:
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                chosen_image = random.choice(images)
                return os.path.join(class_path, chosen_image), chosen_class
            else:
                logging.warning(f"No image files found in directory: {class_path}. Trying another class.")
        except FileNotFoundError:
             logging.warning(f"Class directory not found: {class_path}. Trying another class.")
        except OSError as e:
            logging.warning(f"Error accessing directory {class_path}: {e}. Trying another class.")
        attempts += 1
    logging.error(f"Failed to find a valid image after {max_attempts} attempts for classes: {selected_classes}")
    return None, None

def get_random_unseen_image_path(base_dir, unseen_map):
    if not unseen_map:
        return None, None
    eligible_classes = list(unseen_map.keys())
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts and eligible_classes:
        chosen_class = random.choice(eligible_classes)
        unseen_files = unseen_map.get(chosen_class, []) # Use .get for safety
        if unseen_files:
            chosen_image_name = random.choice(unseen_files)
            return os.path.join(base_dir, chosen_class, chosen_image_name), chosen_class
        else:
            logging.warning(f"Class {chosen_class} selected, but its unseen image list is empty. Removing from choices.")
            eligible_classes.remove(chosen_class)
        attempts += 1
    logging.error(f"Failed to find a valid unseen image after {max_attempts} attempts.")
    return None, None

# (Image selection logic remains the same, using the helper functions)
trained_seen_images_to_test = []
logging.info(f"\nSelecting {NUM_SAMPLES_PER_TYPE} random images from TRAINED classes (could be seen or unseen)...")
for i in range(NUM_SAMPLES_PER_TYPE):
    path, actual_class = get_random_image_path(original_images_dir, trained_class_names)
    if path and actual_class:
        trained_seen_images_to_test.append({"path": path, "actual_class": actual_class})
    else:
        logging.warning(f"Could not get sample {i+1} for trained classes.")
trained_unseen_images_to_test = []
logging.info(f"\nSelecting {NUM_SAMPLES_PER_TYPE} random UNSEEN images from TRAINED classes...")
if not unseen_images_by_class:
    logging.warning("Skipping selection as no unseen images were identified.")
else:
    for i in range(NUM_SAMPLES_PER_TYPE):
        path, actual_class = get_random_unseen_image_path(original_images_dir, unseen_images_by_class)
        if path and actual_class:
            trained_unseen_images_to_test.append({"path": path, "actual_class": actual_class})
        else:
            logging.warning(f"Could not get sample {i+1} for unseen trained images.")
untrained_images_to_test = []
logging.info(f"\nSelecting {NUM_SAMPLES_PER_TYPE} random images from UNTRAINED classes...")
if not untrained_class_names:
     logging.warning("Skipping selection of untrained images as no untrained classes were found.")
else:
    for i in range(NUM_SAMPLES_PER_TYPE):
        path, actual_class = get_random_image_path(original_images_dir, untrained_class_names)
        if path and actual_class:
            untrained_images_to_test.append({"path": path, "actual_class": actual_class})
        else:
             logging.warning(f"Could not get sample {i+1} for untrained classes.")


# =========================
# Image Preprocessing Function (same as before)
# =========================
def preprocess_single_image(img_path, target_size):
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
# Prediction Function (same as before)
# =========================
def predict_and_print(image_path, actual_class_name, image_type_label, threshold):
    logging.info(f"--- Predicting {image_type_label} Image ---")
    logging.info(f"Selected Image: {os.path.basename(image_path)} (Actual Class: {actual_class_name})")
    preprocessed_image = preprocess_single_image(image_path, IMG_SIZE)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image, verbose=0)
        predicted_index = np.argmax(predictions[0])
        top_predicted_class_name = class_names_map.get(predicted_index, "Unknown Class")
        confidence_score = predictions[0][predicted_index]
        print(f"\nResults for {image_type_label} Image:")
        print(f"  File: {os.path.basename(image_path)}")
        print(f"  Actual Class: {actual_class_name}")
        if confidence_score >= threshold:
            print(f"  Predicted Class: {top_predicted_class_name}")
            print(f"  Confidence: {confidence_score:.2%}")
        else:
            print(f"  Predicted Class: Other")
            print(f"  Confidence: {confidence_score:.2%} (Below threshold {threshold:.0%})")
            print(f"  (Top prediction was: {top_predicted_class_name})")
        if "Untrained Class" in image_type_label:
             print("  (Note: Model was not trained on this actual class)")
        elif "Unseen Sample" in image_type_label:
             print("  (Note: Image from a trained class, but not used during training/validation)")
    else:
        print(f"\nCould not process {image_type_label} image: {image_path}")
    print("-" * 40) # Separator

# =========================
# Main Execution (same as before)
# =========================
if __name__ == "__main__":

    if not trained_seen_images_to_test and not trained_unseen_images_to_test and not untrained_images_to_test:
         logging.error("No images were selected for testing across all categories. Exiting.")
         exit()

    if trained_seen_images_to_test:
        print("\n" + "="*10 + " PREDICTIONS ON TRAINED CLASSES (RANDOM ORIGINAL SAMPLE) " + "="*10)
        for img_data in trained_seen_images_to_test:
            predict_and_print(img_data["path"], img_data["actual_class"], "Trained Class (Random Original)", CONFIDENCE_THRESHOLD)
    else:
        logging.info("No images selected from random original trained classes to predict.")

    if trained_unseen_images_to_test:
        print("\n" + "="*10 + " PREDICTIONS ON TRAINED CLASSES (UNSEEN SAMPLES) " + "="*10)
        for img_data in trained_unseen_images_to_test:
            predict_and_print(img_data["path"], img_data["actual_class"], "Trained Class (Unseen Sample)", CONFIDENCE_THRESHOLD)
    else:
        logging.info("No images selected from unseen samples of trained classes to predict.")

    if untrained_images_to_test:
        print("\n" + "="*10 + " PREDICTIONS ON UNTRAINED CLASSES " + "="*10)
        for img_data in untrained_images_to_test:
            predict_and_print(img_data["path"], img_data["actual_class"], "Untrained Class", CONFIDENCE_THRESHOLD)
    else:
        logging.info("No images selected from untrained classes to predict.")

    logging.info("Prediction script finished.")