import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    categories = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    for category in categories:
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = [img for img in os.listdir(category_path) if img.endswith(".jpg")]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for img_set, folder in zip([train_images, val_images, test_images], ["train", "val", "test"]):
            output_category_path = os.path.join(output_dir, folder, category)
            os.makedirs(output_category_path, exist_ok=True)

            for img in img_set:
                shutil.copy(os.path.join(category_path, img), os.path.join(output_category_path, img))

# Example usage:
split_dataset("data/UECFOOD256", "data", train_ratio=0.7, val_ratio=0.15)
