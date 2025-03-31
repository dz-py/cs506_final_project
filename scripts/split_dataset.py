import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    Split a single-category dataset into training and validation sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train", "rice")
    val_dir = os.path.join(output_dir, "val", "rice")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    images = [img for img in os.listdir(input_dir) if img.endswith(".jpg")]
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    train_images = images[:train_split]
    val_images = images[train_split:]

    for img in train_images:
        shutil.copy(os.path.join(input_dir, img), os.path.join(train_dir, img))
    
    for img in val_images:
        shutil.copy(os.path.join(input_dir, img), os.path.join(val_dir, img))

# Example usage:
input_dir = os.path.join(os.getcwd(), "dataset", "train", "train1")
output_dir = os.path.join(os.getcwd(), "dataset")
split_dataset(input_dir, output_dir)

