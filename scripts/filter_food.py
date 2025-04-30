import os
import shutil
import random
import logging

# =========================
# Configuration
# =========================
NUM_MAIN_CLASSES = 10 # Number of main food classes to select
NUM_IMAGES_PER_MAIN_CLASS = 100 # Number of images per main class
NUM_IMAGES_PER_UNKNOWN_CLASS = 10 # Number of images per *original* class to put into 'unknown'
RANDOM_SEED = 42 # Set a seed for reproducible random selection
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
    project_root = os.getcwd() # Assume running from project root if script dir unknown
    logging.warning(f"Could not determine script directory, assuming project root is: {project_root}")


# Define the path to the original images directory
original_images_dir = os.path.join(project_root, "food-101", "images")

# Define the path where the filtered images will be saved
# Changed directory name to reflect the new structure
filtered_images_dir = os.path.join(project_root, "food-101", "images_filtered_10_main_1_unknown")

logging.info(f"Original images directory: {original_images_dir}")
logging.info(f"Filtered images directory: {filtered_images_dir}")

# Create the filtered images directory if it doesn't already exist
# Also remove it first if it exists, to ensure a clean slate
if os.path.exists(filtered_images_dir):
    logging.warning(f"Removing existing filtered directory: {filtered_images_dir}")
    shutil.rmtree(filtered_images_dir)
os.makedirs(filtered_images_dir, exist_ok=True)
logging.info(f"Created empty filtered directory: {filtered_images_dir}")

# =========================
# Class Selection
# =========================
logging.info("Scanning original images directory to identify classes...")

if not os.path.isdir(original_images_dir):
    logging.error(f"Original images directory not found: {original_images_dir}")
    exit(1)

try:
    # List all potential class directories
    all_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])
    if not all_classes:
        logging.error(f"No class subdirectories found in {original_images_dir}")
        exit(1)
    logging.info(f"Found {len(all_classes)} total classes.")

    if len(all_classes) < NUM_MAIN_CLASSES:
        logging.error(f"Requested {NUM_MAIN_CLASSES} main classes, but only found {len(all_classes)} total classes.")
        exit(1)

except OSError as e:
    logging.error(f"Error listing classes in {original_images_dir}: {e}")
    exit(1)

# Set the random seed for reproducibility
random.seed(RANDOM_SEED)

# Randomly select the main classes
main_classes = sorted(random.sample(all_classes, NUM_MAIN_CLASSES))
logging.info(f"Selected {len(main_classes)} main classes: {main_classes}")

# Determine the classes to be grouped into 'unknown'
unselected_classes = sorted(list(set(all_classes) - set(main_classes)))
logging.info(f"Identified {len(unselected_classes)} classes for the '{UNKNOWN_CLASS_DIR_NAME}' category.")

# =========================
# Copy Images for Main Classes
# =========================
logging.info(f"Copying {NUM_IMAGES_PER_MAIN_CLASS} images for each of the {len(main_classes)} main classes...")
total_main_copied = 0
for class_name in main_classes:
    src_class_dir = os.path.join(original_images_dir, class_name)
    dst_class_dir = os.path.join(filtered_images_dir, class_name)
    os.makedirs(dst_class_dir, exist_ok=True) # Create destination directory

    try:
        # List valid image files in the source class directory
        images_in_class = [
            f for f in os.listdir(src_class_dir)
            if os.path.isfile(os.path.join(src_class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(images_in_class) < NUM_IMAGES_PER_MAIN_CLASS:
            logging.warning(f"Class '{class_name}' only has {len(images_in_class)} images, requested {NUM_IMAGES_PER_MAIN_CLASS}. Copying all available.")
            selected_images = images_in_class # Select all available
        else:
            # Randomly select the desired number of images
            selected_images = random.sample(images_in_class, NUM_IMAGES_PER_MAIN_CLASS)

        # Copy selected images
        count_per_class = 0
        for img_filename in selected_images:
            src_path = os.path.join(src_class_dir, img_filename)
            dst_path = os.path.join(dst_class_dir, img_filename)
            try:
                shutil.copy2(src_path, dst_path)
                count_per_class += 1
            except Exception as e:
                logging.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        logging.info(f"  Copied {count_per_class} images for class '{class_name}'.")
        total_main_copied += count_per_class

    except OSError as e:
        logging.warning(f"Could not process directory {src_class_dir}: {e}. Skipping this class.")
        continue

logging.info(f"Finished copying main class images. Total copied: {total_main_copied}")

# =========================
# Copy Images for "Unknown" Class
# =========================
logging.info(f"Copying {NUM_IMAGES_PER_UNKNOWN_CLASS} images from each of the {len(unselected_classes)} remaining classes into '{UNKNOWN_CLASS_DIR_NAME}'...")

# Create the destination directory for the unknown class
unknown_dst_dir = os.path.join(filtered_images_dir, UNKNOWN_CLASS_DIR_NAME)
os.makedirs(unknown_dst_dir, exist_ok=True)

total_unknown_copied = 0
# Loop through the classes designated for 'unknown'
for class_name in unselected_classes:
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

        if len(images_in_class) < NUM_IMAGES_PER_UNKNOWN_CLASS:
            logging.warning(f"Class '{class_name}' only has {len(images_in_class)} images, requested {NUM_IMAGES_PER_UNKNOWN_CLASS} for unknown. Selecting all available.")
            selected_images = images_in_class
        else:
            # Randomly select images
            selected_images = random.sample(images_in_class, NUM_IMAGES_PER_UNKNOWN_CLASS)

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
        # Log progress per original class contributing to unknown
        # logging.debug(f"  Copied {count_per_class} images from class '{class_name}' to '{UNKNOWN_CLASS_DIR_NAME}'.")
        total_unknown_copied += count_per_class

    except OSError as e:
        logging.warning(f"Could not process directory {src_class_dir} for unknown class: {e}. Skipping this class.")
        continue

logging.info(f"Finished copying images for '{UNKNOWN_CLASS_DIR_NAME}'. Total copied: {total_unknown_copied}")
logging.info("Script finished.")