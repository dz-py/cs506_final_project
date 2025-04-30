import os
import shutil
import random

# Define the path to the original images directory
original_images_dir = os.path.join(os.getcwd(), "food-101", "images")

# Define the path where the filtered images will be saved
filtered_images_dir = os.path.join(os.getcwd(), "food-101", "images_filtered")

# Create the filtered images directory if it doesn't already exist
os.makedirs(filtered_images_dir, exist_ok=True)

# List all class directories (i.e., food categories) in the original images directory
# Only include directories (exclude files)
all_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])

# Set the random seed for reproducibility
random.seed(42)

# Randomly select 6 classes from the list of all classes
selected_classes = random.sample(all_classes, 6)
print("Selected classes:", selected_classes)

# For each selected class, copy 100 random images to the filtered directory
for class_name in selected_classes:
    # Source directory for the current class
    src = os.path.join(original_images_dir, class_name)
    # Destination directory for the current class in the filtered directory
    dst = os.path.join(filtered_images_dir, class_name)
    # Create the destination directory if it doesn't exist
    os.makedirs(dst, exist_ok=True)
    # List all image files in the source class directory
    images = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    # Randomly select 100 images from the list
    selected_images = random.sample(images, 100)
    # Copy each selected image from the source to the destination directory
    for img in selected_images:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))
