import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the input directory containing the processed images (organized by class)
input_dir = os.path.join(os.getcwd(), "food-101", "images_processed")

# Define the output directory where the split dataset (train/val) will be stored
output_dir = os.path.join(os.getcwd(), "food-101", "dataset_split")

def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Splits the dataset into training and validation sets, preserving class distribution,
    and copies the images into corresponding directories.

    Args:
        input_dir (str): Path to the directory with processed images organized by class.
        output_dir (str): Path where the split dataset will be stored.
        train_ratio (float): Proportion of data to use for training (rest is for validation).
        seed (int): Random seed for reproducibility.
    """
    # Get a list of all category (class) directories in the input directory
    categories = [cat for cat in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cat))]
    
    # Prepare a list to hold image file information (category and filename)
    data = []
    for cat in categories:
        # Iterate through all files in each category directory
        for img in os.listdir(os.path.join(input_dir, cat)):
            # Only consider image files with common extensions
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({'category': cat, 'img': img})
    
    # Convert the data list into a pandas DataFrame for easy manipulation
    df = pd.DataFrame(data)
    
    # Split the DataFrame into training and validation sets
    # Stratify ensures class distribution is preserved in both splits
    train_df, val_df = train_test_split(
        df,
        test_size=1-train_ratio,
        stratify=df['category'],
        random_state=seed
    )

    # For both splits (train and val), copy images to their respective directories
    for split, split_df in [('train', train_df), ('val', val_df)]:
        split_dir = os.path.join(output_dir, split)  # Directory for current split (train/val)
        os.makedirs(split_dir, exist_ok=True)        # Create the split directory if it doesn't exist
        
        # Iterate through each row in the split DataFrame
        for _, row in split_df.iterrows():
            # Source file path (original image location)
            src = os.path.join(input_dir, row['category'], row['img'])
            # Destination directory for the current category in the split
            dst_dir = os.path.join(split_dir, row['category'])
            os.makedirs(dst_dir, exist_ok=True)      # Create category directory if it doesn't exist
            # Destination file path
            dst = os.path.join(dst_dir, row['img'])
            # Copy the image file if it doesn't already exist at the destination
            if not os.path.exists(dst):
                shutil.copy(src, dst)

# Run the split_dataset function if this script is executed as the main program
if __name__ == "__main__":
    split_dataset(input_dir, output_dir)
