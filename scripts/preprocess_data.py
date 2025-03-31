import os
import cv2

def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_resized = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(output_category_path, img_file), img_resized)
