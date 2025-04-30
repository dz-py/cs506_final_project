import os
import cv2

# Define the input directory containing the filtered images (organized by class)
input_dir = os.path.join(os.getcwd(), "food-101", "images_filtered")

# Define the output directory where the processed images will be saved
output_dir = os.path.join(os.getcwd(), "food-101", "images_processed")

def preprocess_images(input_dir, output_dir, img_size=(224, 224)):
    """
    Preprocesses images by resizing them to a specified size and saving them to a new directory structure.

    Args:
        input_dir (str): Path to the input directory containing class folders with images.
        output_dir (str): Path to the output directory where processed images will be saved.
        img_size (tuple): Desired image size (width, height) for resizing.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each category (class) in the input directory
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)  # Path to the current class folder
        output_category_path = os.path.join(output_dir, category)  # Path to save processed images for this class
        
        # Create the output directory for the current class if it doesn't exist
        os.makedirs(output_category_path, exist_ok=True)
        
        # Iterate over each image file in the current class folder
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)  # Full path to the input image
            
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            
            # If the image could not be read (e.g., corrupted file), skip it
            if img is None:
                continue
            
            # Resize the image to the specified size (default: 224x224)
            img_resized = cv2.resize(img, img_size)
            
            # Save the resized image to the corresponding output directory
            cv2.imwrite(os.path.join(output_category_path, img_file), img_resized)

# Run the preprocessing function if this script is executed as the main program
if __name__ == "__main__":
    preprocess_images(input_dir, output_dir)
