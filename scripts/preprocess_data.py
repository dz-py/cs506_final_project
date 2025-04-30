import os
import cv2
import logging

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

# Define the input directory - **MODIFIED** to read from the correct filtered directory
input_dir = os.path.join(project_root, "food-101", "images_filtered_10_main_1_unknown")

# Define the output directory where the processed images will be saved
output_dir = os.path.join(project_root, "food-101", "images_processed")

logging.info(f"Input directory: {input_dir}")
logging.info(f"Output directory: {output_dir}")

# =========================
# Preprocessing Function
# =========================
def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    """
    Preprocesses images by resizing them to a specified size and saving them
    to a new directory structure, maintaining class folders.

    Args:
        input_dir (str): Path to the input directory containing class folders with images.
        output_dir (str): Path to the output directory where processed images will be saved.
        img_size (tuple): Desired image size (width, height) for resizing.
    """
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    # Create the output directory if it doesn't exist
    # Also remove it first if it exists, to ensure a clean slate
    if os.path.exists(output_dir):
        logging.warning(f"Removing existing processed directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            logging.error(f"Error removing directory {output_dir}: {e}")
            return # Stop if we can't clear the old directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created empty processed directory: {output_dir}")
    except OSError as e:
        logging.error(f"Error creating directory {output_dir}: {e}")
        return # Stop if we can't create the output directory


    processed_count_total = 0
    skipped_count_total = 0

    # Iterate over each category (class folder) in the input directory
    try:
        categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if not categories:
            logging.warning(f"No category subdirectories found in {input_dir}")
            return
    except OSError as e:
        logging.error(f"Could not list directories in {input_dir}: {e}")
        return

    logging.info(f"Found categories: {categories}")

    for category in categories:
        category_path = os.path.join(input_dir, category)  # Path to the current input class folder
        output_category_path = os.path.join(output_dir, category)  # Path to save processed images for this class

        # Create the output directory for the current class
        try:
            os.makedirs(output_category_path, exist_ok=True)
        except OSError as e:
            logging.error(f"Could not create output directory {output_category_path}: {e}. Skipping category '{category}'.")
            continue # Skip this category

        logging.info(f"Processing category: {category}...")
        processed_count_category = 0
        skipped_count_category = 0

        # Iterate over each image file in the current class folder
        try:
            image_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        except OSError as e:
             logging.error(f"Could not list files in {category_path}: {e}. Skipping category '{category}'.")
             continue # Skip this category

        for img_file in image_files:
            # Check for valid image extensions (optional but good practice)
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.warning(f"Skipping non-image file: {img_file} in {category}")
                skipped_count_category += 1
                continue

            img_path = os.path.join(category_path, img_file)  # Full path to the input image
            output_img_path = os.path.join(output_category_path, img_file) # Full path for output image

            try:
                # Read the image using OpenCV
                img = cv2.imread(img_path)

                # If the image could not be read (e.g., corrupted file), skip it
                if img is None:
                    logging.warning(f"Could not read image file (possibly corrupt): {img_path}. Skipping.")
                    skipped_count_category += 1
                    continue

                # Resize the image to the specified size (default: 224x224)
                img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA) # Specify interpolation

                # Save the resized image to the corresponding output directory
                success = cv2.imwrite(output_img_path, img_resized)
                if success:
                    processed_count_category += 1
                else:
                    logging.warning(f"Failed to write processed image: {output_img_path}. Skipping.")
                    skipped_count_category += 1

            except Exception as e:
                 logging.error(f"Error processing image {img_path}: {e}. Skipping.")
                 skipped_count_category += 1
                 continue # Skip to next image on error

        logging.info(f"  Finished category '{category}'. Processed: {processed_count_category}, Skipped: {skipped_count_category}")
        processed_count_total += processed_count_category
        skipped_count_total += skipped_count_category

    logging.info(f"\nPreprocessing complete. Total processed: {processed_count_total}, Total skipped: {skipped_count_total}")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    preprocess_images(input_dir, output_dir)
    logging.info("Preprocessing script finished.")
