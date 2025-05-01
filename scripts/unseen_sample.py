import os
import shutil
import random
import logging
import pickle

# =========================
# Configuration
# =========================
NUM_IMAGES_PER_MAIN_CLASS = 10 # How many *unseen* images per trained class to include
NUM_IMAGES_PER_UNKNOWN_CLASS = 3 # How many images per *original remaining* class to put into 'unknown'
RANDOM_SEED = 42         # Seed for reproducible random selection
OUTPUT_DIR_NAME = "food-101-unseen-trained-plus-unknown-samples" # Updated Name for the output directory
UNKNOWN_CLASS_DIR_NAME = "unknown" # Name for the directory holding images from other classes
# =========================

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Path Setup
# =========================
try:
    # Assumes the script is in a 'scripts' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively)
    script_dir = os.getcwd()
    # If running from project root, project_root is cwd. If running from scripts, need parent.
    if os.path.basename(script_dir).lower() == 'scripts':
         project_root = os.path.dirname(script_dir)
    else:
         project_root = script_dir # Assume running from project root
    logging.warning(f"Could not determine script directory reliably, assuming project root is: {project_root}")

# Define the path to the original full images directory
original_images_dir = os.path.join(project_root, "food-101", "images")
# Define the path to the directory containing the train/val splits (used to identify unseen)
split_base_dir = os.path.join(project_root, "food-101-filtered-5main-unknown", "dataset_split")
# Define the path where the sample dataset will be created
output_sample_dir = os.path.join(project_root, OUTPUT_DIR_NAME)
# Path to load class indices (to know which classes were trained)
model_dir = os.path.join(project_root, "models")
CLASS_INDICES_PATH = os.path.join(model_dir, "class_indices.pkl")


logging.info(f"Original images directory: {original_images_dir}")
logging.info(f"Split dataset directory: {split_base_dir}")
logging.info(f"Output sample directory: {output_sample_dir}")
logging.info(f"Class indices path: {CLASS_INDICES_PATH}")

# =========================
# Load Trained Class Names
# =========================
if not os.path.exists(CLASS_INDICES_PATH):
    logging.error(f"Class indices file not found at: {CLASS_INDICES_PATH}")
    logging.error("Please ensure train_model.py ran successfully and saved the class indices.")
    exit(1)
try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    # Get only the main trained classes (exclude 'unknown' if present)
    trained_main_classes = sorted([name for name, index in class_indices.items() if name != UNKNOWN_CLASS_DIR_NAME])
    logging.info(f"Loaded class indices. Identifying unseen images for {len(trained_main_classes)} main trained classes: {trained_main_classes}")
    if not trained_main_classes:
         logging.error("No main trained classes found in the class indices file (excluding 'unknown').")
         exit(1)
except Exception as e:
    logging.error(f"Error loading or processing class indices: {e}")
    exit(1)

# =========================
# Identify Remaining Classes for Unknown
# =========================
logging.info("Identifying remaining classes for the 'unknown' category...")
remaining_classes = []
try:
    # List all potential class directories in the original dataset
    all_original_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])
    if not all_original_classes:
        logging.error(f"No class subdirectories found in {original_images_dir}")
        exit(1)
    # Find classes that are NOT in the main trained list
    remaining_classes = sorted(list(set(all_original_classes) - set(trained_main_classes)))
    logging.info(f"Identified {len(remaining_classes)} classes for the '{UNKNOWN_CLASS_DIR_NAME}' category.")
    if not remaining_classes:
        logging.warning("No remaining classes found to populate the 'unknown' category.")

except OSError as e:
    logging.error(f"Error listing classes in {original_images_dir}: {e}")
    exit(1)


# =========================
# Create Output Directory (Clean Slate)
# =========================
if os.path.exists(output_sample_dir):
    logging.warning(f"Removing existing sample directory: {output_sample_dir}")
    try:
        shutil.rmtree(output_sample_dir)
    except OSError as e:
        logging.error(f"Error removing directory {output_sample_dir}: {e}")
        exit(1)
try:
    os.makedirs(output_sample_dir, exist_ok=True)
    logging.info(f"Created empty sample directory: {output_sample_dir}")
except OSError as e:
     logging.error(f"Error creating directory {output_sample_dir}: {e}")
     exit(1)

# =========================
# Identify and Copy Unseen Images from TRAINED classes
# =========================
# **FIXED:** Changed log message to use correct variable name
logging.info(f"Identifying and copying {NUM_IMAGES_PER_MAIN_CLASS} unseen images for each main trained class...")
total_copied_main_unseen = 0
total_skipped_due_to_shortage_main = 0

# Set the random seed for reproducibility
random.seed(RANDOM_SEED)

for class_name in trained_main_classes:
    original_class_path = os.path.join(original_images_dir, class_name)
    train_class_path = os.path.join(split_base_dir, "train", class_name)
    val_class_path = os.path.join(split_base_dir, "val", class_name)
    dst_class_dir = os.path.join(output_sample_dir, class_name) # Destination within the new sample dir

    unseen_files = []
    try:
        # Get all original images for this class
        if not os.path.isdir(original_class_path):
            logging.warning(f"Original directory not found for class '{class_name}' at {original_class_path}. Skipping.")
            continue
        original_files = set(f for f in os.listdir(original_class_path)
                             if os.path.isfile(os.path.join(original_class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')))

        # Get seen images (train + val)
        seen_files = set()
        if os.path.isdir(train_class_path):
            seen_files.update(f for f in os.listdir(train_class_path)
                              if os.path.isfile(os.path.join(train_class_path, f)))
        if os.path.isdir(val_class_path):
            seen_files.update(f for f in os.listdir(val_class_path)
                              if os.path.isfile(os.path.join(val_class_path, f)))

        # Find the difference
        unseen_files = sorted(list(original_files - seen_files))

        if not unseen_files:
            logging.warning(f"No unseen images found for trained class: {class_name}. Skipping copy for this class.")
            continue # Skip to next class if no unseen images found

        logging.info(f"  Found {len(unseen_files)} unseen images for class '{class_name}'.")

        # Select images to copy
        num_available = len(unseen_files)
        # **FIXED:** Use NUM_IMAGES_PER_MAIN_CLASS here
        num_to_select = min(num_available, NUM_IMAGES_PER_MAIN_CLASS)

        # **FIXED:** Use NUM_IMAGES_PER_MAIN_CLASS here
        if num_available < NUM_IMAGES_PER_MAIN_CLASS:
            # **FIXED:** Use NUM_IMAGES_PER_MAIN_CLASS here
            logging.warning(f"  Class '{class_name}' only has {num_available} unseen images, requested {NUM_IMAGES_PER_MAIN_CLASS}. Selecting all available.")
            # **FIXED:** Use NUM_IMAGES_PER_MAIN_CLASS here
            total_skipped_due_to_shortage_main += (NUM_IMAGES_PER_MAIN_CLASS - num_available)

        selected_image_filenames = random.sample(unseen_files, num_to_select)

        # Create destination directory and copy files
        os.makedirs(dst_class_dir, exist_ok=True)
        count_per_class = 0
        for img_filename in selected_image_filenames:
            src_path = os.path.join(original_class_path, img_filename)
            dst_path = os.path.join(dst_class_dir, img_filename)
            try:
                shutil.copy2(src_path, dst_path) # copy2 preserves metadata
                count_per_class += 1
            except Exception as e:
                logging.error(f"  Failed to copy {src_path} to {dst_path}: {e}")
        logging.info(f"  Copied {count_per_class} unseen images for class '{class_name}'.")
        total_copied_main_unseen += count_per_class

    except OSError as e:
        logging.error(f"Error processing files for class {class_name}: {e}")
        continue # Skip to next class on error

logging.info(f"Finished copying unseen images for main classes. Total copied: {total_copied_main_unseen}")

# =========================
# Copy Images for "Unknown" Class from REMAINING classes
# =========================
# (This section correctly uses NUM_IMAGES_PER_UNKNOWN_CLASS - no changes needed here)
logging.info(f"\nCopying {NUM_IMAGES_PER_UNKNOWN_CLASS} images from each of the {len(remaining_classes)} remaining classes into '{UNKNOWN_CLASS_DIR_NAME}'...")

# Create the destination directory for the unknown class
unknown_dst_dir = os.path.join(output_sample_dir, UNKNOWN_CLASS_DIR_NAME)
os.makedirs(unknown_dst_dir, exist_ok=True)

total_unknown_copied = 0
total_skipped_due_to_shortage_unknown = 0
# Loop through the classes designated for 'unknown'
for class_name in remaining_classes:
    src_class_dir = os.path.join(original_images_dir, class_name)

    try:
        # List valid image files
        images_in_class = [
            f for f in os.listdir(src_class_dir)
            if os.path.isfile(os.path.join(src_class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not images_in_class:
            logging.warning(f"No images found in source directory for class '{class_name}'. Skipping for unknown.")
            continue

        num_available = len(images_in_class)
        num_to_select = min(num_available, NUM_IMAGES_PER_UNKNOWN_CLASS)

        if num_available < NUM_IMAGES_PER_UNKNOWN_CLASS:
            logging.warning(f"Class '{class_name}' only has {num_available} images, requested {NUM_IMAGES_PER_UNKNOWN_CLASS} for unknown. Selecting all available.")
            total_skipped_due_to_shortage_unknown += (NUM_IMAGES_PER_UNKNOWN_CLASS - num_available)


        # Randomly select images
        selected_images = random.sample(images_in_class, num_to_select)

        # Copy selected images into the single 'unknown' directory
        count_per_class = 0
        for img_filename in selected_images:
            src_path = os.path.join(src_class_dir, img_filename)
            # Prepend original class name to filename to avoid collisions
            dst_filename = f"{class_name}_{img_filename}"
            dst_path = os.path.join(unknown_dst_dir, dst_filename)
            try:
                shutil.copy2(src_path, dst_path)
                count_per_class += 1
            except Exception as e:
                logging.error(f"Failed to copy {src_path} to {dst_path} (as {dst_filename}): {e}")
        # logging.debug(f"  Copied {count_per_class} images from class '{class_name}' to '{UNKNOWN_CLASS_DIR_NAME}'.")
        total_unknown_copied += count_per_class

    except OSError as e:
        logging.warning(f"Could not process directory {src_class_dir} for unknown class: {e}. Skipping this class.")
        continue

logging.info(f"Finished copying images for '{UNKNOWN_CLASS_DIR_NAME}'. Total copied: {total_unknown_copied}")

# =========================
# Final Summary
# =========================
logging.info(f"\nSample dataset creation complete.")
logging.info(f"  Total main classes included: {len(trained_main_classes)}")
logging.info(f"  Total unseen images copied for main classes: {total_copied_main_unseen}")
# **FIXED:** Use NUM_IMAGES_PER_MAIN_CLASS here
if total_skipped_due_to_shortage_main > 0:
    logging.warning(f"    Note: Could not find the requested {NUM_IMAGES_PER_MAIN_CLASS} unseen images for some main classes. Short by {total_skipped_due_to_shortage_main} images.")
logging.info(f"  Total images copied for '{UNKNOWN_CLASS_DIR_NAME}' category: {total_unknown_copied} (from {len(remaining_classes)} source classes)")
if total_skipped_due_to_shortage_unknown > 0:
     logging.warning(f"    Note: Could not find the requested {NUM_IMAGES_PER_UNKNOWN_CLASS} images for some remaining classes. Short by {total_skipped_due_to_shortage_unknown} images.")
logging.info(f"Sample dataset created at: {output_sample_dir}")
