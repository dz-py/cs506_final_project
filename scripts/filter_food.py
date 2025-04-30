import os
import shutil
import random

original_images_dir = os.path.join(os.getcwd(), "food-101", "images")
filtered_images_dir = os.path.join(os.getcwd(), "food-101", "images_filtered")

os.makedirs(filtered_images_dir, exist_ok=True)

all_classes = sorted([d for d in os.listdir(original_images_dir) if os.path.isdir(os.path.join(original_images_dir, d))])
random.seed(42)
selected_classes = random.sample(all_classes, 6)
print("Selected classes:", selected_classes)

for class_name in selected_classes:
    src = os.path.join(original_images_dir, class_name)
    dst = os.path.join(filtered_images_dir, class_name)
    os.makedirs(dst, exist_ok=True)
    images = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    selected_images = random.sample(images, 100)
    for img in selected_images:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))
