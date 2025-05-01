import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
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
    # If running from project root, project_root is cwd. If running from scripts, need parent.
    if os.path.basename(script_dir).lower() == 'scripts':
         project_root = os.path.dirname(script_dir)
    else:
         project_root = script_dir # Assume running from project root
    logging.warning(f"Could not determine script directory reliably, assuming project root is: {project_root}")

# Define the base directory for the sample dataset
sample_base_dir = os.path.join(project_root, "food-101-filtered-5main-unknown")

# Define the input directory containing the processed images (organized by class)
# Reads from the 'images_processed' folder *inside* the sample directory
input_dir = os.path.join(sample_base_dir, "images_processed")

# Define the output directory where the 'dataset_split' folder (containing train/val) will be created
# **MODIFIED:** Point to a 'dataset_split' subdirectory *inside* the sample directory
output_dir = os.path.join(sample_base_dir, "dataset_split")

logging.info(f"Input directory (Processed Images): {input_dir}")
logging.info(f"Output directory (Split Dataset Root): {output_dir}")
logging.warning(f"Train/Val splits will be created inside: {output_dir}")


# =========================
# Splitting Function
# =========================
def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Splits the dataset from input_dir into training and validation sets
    inside output_dir, preserving class distribution, and copies the images.
    The train/val folders will be created inside the specified output_dir.

    Args:
        input_dir (str): Path to the directory with processed images organized by class.
        output_dir (str): Path where the 'train' and 'val' split folders will be created.
        train_ratio (float): Proportion of data to use for training (rest is for validation).
        seed (int): Random seed for reproducibility.
    """
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        logging.error("Please ensure preprocess_data.py ran successfully and created this directory.")
        return

    # Define specific train/val paths *inside* the output directory
    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")

    # Clean slate: Remove existing output_dir (which contains train/val) if it exists
    if os.path.exists(output_dir):
        logging.warning(f"Removing existing split directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            logging.error(f"Error removing directory {output_dir}: {e}")
            return # Stop if we can't clear the old directory
    # Create the main output directory (e.g., 'dataset_split')
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created empty split directory: {output_dir}")
        # Also create train/val immediately inside it for clarity, although makedirs later would also work
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directory structure within {output_dir}: {e}")
        return

    # Get a list of all category (class) directories in the input directory
    try:
        categories = [cat for cat in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cat))]
        if not categories:
            logging.warning(f"No category subdirectories found in {input_dir}")
            return
        logging.info(f"Found categories to split: {categories}")
    except OSError as e:
        logging.error(f"Could not list directories in {input_dir}: {e}")
        return

    # Prepare a list to hold image file information (category and filename)
    data = []
    logging.info("Scanning processed images to build file list...")
    for cat in categories:
        category_path = os.path.join(input_dir, cat)
        try:
            # Iterate through all files in each category directory
            for img in os.listdir(category_path):
                # Only consider image files with common extensions
                if img.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(category_path, img)):
                    data.append({'category': cat, 'img': img})
        except OSError as e:
            logging.warning(f"Could not list files in {category_path}: {e}. Skipping category '{cat}'.")
            continue # Skip this category if error

    if not data:
        logging.error("No image files found in any category directory.")
        return

    # Convert the data list into a pandas DataFrame for easy manipulation
    df = pd.DataFrame(data)
    logging.info(f"Total images found across all categories: {len(df)}")

    # Check if stratification is possible
    min_samples_per_class = df['category'].value_counts().min()
    if min_samples_per_class < 2:
         logging.warning(f"Some classes have fewer than 2 samples ({min_samples_per_class}). Stratified split might behave unexpectedly or fail. Consider adjusting data or using non-stratified split.")

    # Split the DataFrame into training and validation sets
    logging.info(f"Splitting dataset with train ratio {train_ratio}...")
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=1-train_ratio,
            stratify=df['category'], # Ensures class distribution is preserved
            random_state=seed
        )
        logging.info(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    except ValueError as e:
         logging.error(f"Error during train/test split (often due to too few samples in a class for stratification): {e}")
         return
    except Exception as e:
        logging.error(f"Unexpected error during train/test split: {e}")
        return


    # For both splits (train and val), copy images to their respective directories
    logging.info("Copying files to train and validation directories...")
    total_copied_train = 0
    total_copied_val = 0
    # Use the specific train/val output directories
    for split, split_df, split_target_dir in [('train', train_df, train_output_dir), ('val', val_df, val_output_dir)]:

        # Iterate through each row in the split DataFrame
        for _, row in split_df.iterrows():
            category_name = row['category']
            img_name = row['img']
            # Source file path (from processed images)
            src = os.path.join(input_dir, category_name, img_name)
            # Destination directory for the current category *within* the specific split dir
            dst_dir = os.path.join(split_target_dir, category_name)
            try:
                os.makedirs(dst_dir, exist_ok=True) # Create category directory if it doesn't exist
            except OSError as e:
                 logging.error(f"Could not create destination directory {dst_dir}: {e}. Skipping file {img_name}.")
                 continue # Skip this file if dir creation fails

            # Destination file path
            dst = os.path.join(dst_dir, img_name)

            # Copy the image file if it exists at source and not at destination
            if os.path.exists(src):
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst) # copy2 preserves metadata
                        if split == 'train':
                            total_copied_train += 1
                        else:
                            total_copied_val += 1
                    except Exception as e:
                        logging.error(f"Failed to copy {src} to {dst}: {e}")
                # else: # Optional: log if file already exists
                #     logging.debug(f"Destination file already exists, skipping copy: {dst}")
            else:
                 logging.warning(f"Source file not found during copy operation: {src}. Skipping.")

    logging.info(f"File copying complete. Train files copied: {total_copied_train}, Val files copied: {total_copied_val}")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    split_dataset(input_dir, output_dir)
    logging.info("Dataset splitting script finished.")
