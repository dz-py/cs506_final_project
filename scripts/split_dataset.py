import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    Split a multi-class dataset into training and validation sets.
    Each category will have its images split proportionally.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        images = [img for img in os.listdir(category_path) if img.endswith((".jpg", ".png"))]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        train_images = images[:train_split]
        val_images = images[train_split:]

        # Create category directories in train/val folders
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)

        # Copy images to respective directories
        for img in train_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(train_category_dir, img)
            if src_path != dst_path:  # Avoid SameFileError
                shutil.copy(src_path, dst_path)
        
        for img in val_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(val_category_dir, img)
            if src_path != dst_path:  # Avoid SameFileError
                shutil.copy(src_path, dst_path)

# Example usage:
input_dir = os.path.join(os.getcwd(), "dataset", "train")  # Path to original dataset
output_dir = os.path.join(os.getcwd(), "dataset_split")   # Separate directory for split data
split_dataset(input_dir, output_dir)

