import os
import tensorflow as tf
# Suppress TensorFlow INFO/WARNING messages (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARNING')

from tensorflow.keras.preprocessing import image # Use keras preprocessing
from tensorflow.keras.applications.resnet import preprocess_input # Use same preprocessing
import numpy as np
import pickle
import logging
import random
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont # <-- Added for image manipulation
import textwrap # <-- Added for text wrapping

load_dotenv()

# =========================
# USDA API Integration
# =========================
def get_usda_nutrition(food_name):
    """Get nutritional data from USDA FoodData Central API"""
    api_key = os.getenv('USDA_API_KEY')
    if not api_key:
        logging.error("USDA_API_KEY environment variable not set")
        return None

    url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
    headers = {'Content-Type': 'application/json'}

    processed_name = food_name.replace('_', ' ').title()

    params = {
        'query': processed_name,
        'dataType': ["Foundation", "Survey (FNDDS)", "Branded"],
        'pageSize': 5,
        'sortBy': 'dataType.keyword',
        'queryOperator': 'AND' # Changed from 'OR' to 'AND' for potentially better relevance
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            params={'api_key': api_key},
            json=params,
            timeout=10 # Increased timeout slightly
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return find_best_match(processed_name, response.json())

    except requests.exceptions.RequestException as e:
        logging.error(f"USDA API Error: {e}")
        return None
    except Exception as e: # Catch potential JSON decoding errors or others
        logging.error(f"Error processing USDA response for '{processed_name}': {e}")
        return None


def find_best_match(query, api_response):
    """Finds the closest matching food item"""
    # Use difflib for slightly fuzzy matching (optional but can help)
    from difflib import get_close_matches

    if not api_response or not api_response.get('foods'):
        logging.warning(f"No foods found in USDA response for query: {query}")
        return None

    # Prioritize exact or very close matches first
    descriptions = [f['description'].lower() for f in api_response['foods']]
    matches = get_close_matches(query.lower(), descriptions, n=1, cutoff=0.8) # Stricter cutoff

    if matches:
        best_match_desc = matches[0]
        # Find the full entry for the best match description
        best_match_entry = next((f for f in api_response['foods']
                                 if f['description'].lower() == best_match_desc), None)
        if best_match_entry:
             logging.info(f"Found close match for '{query}': '{best_match_entry['description']}'")
             return extract_nutrients(best_match_entry)

    # Fallback: Check if query is a substring of any description (less precise)
    logging.info(f"No close match found for '{query}'. Trying partial matching...")
    for food in api_response['foods']:
        if query.lower() in food['description'].lower():
            logging.info(f"Found partial match for '{query}': '{food['description']}'")
            return extract_nutrients(food)

    # If still no match, return None
    logging.warning(f"Could not find a suitable match for '{query}' in USDA results.")
    return None


def extract_nutrients(food_entry):
    """Extracts key nutrients with enhanced validation"""
    if not food_entry:
        return None

    nutrients = {
        'description': food_entry.get('description', 'Unknown Food'),
        'fdcId': food_entry.get('fdcId')
    }

    # Map nutrient IDs to desired keys (adjust IDs if needed based on USDA documentation)
    # Common IDs: Calories (1008), Protein (1003), Total lipid (fat) (1004), Carbohydrate, by difference (1005)
    # Check specific nutrients if needed (e.g., Sugars (2000), Fiber (1079), Sodium (1093))
    nutrient_map = {
        '1008': 'calories', # Energy in kcal
        '1003': 'protein',  # Protein in g
        '1004': 'fat',      # Total lipid (fat) in g
        '1005': 'carbs'     # Carbohydrate, by difference in g
    }

    found_nutrients = set()
    for nutrient in food_entry.get('foodNutrients', []):
        nutrient_id = str(nutrient.get('nutrientId')) # Ensure ID is string for mapping
        nutrient_name = nutrient.get('nutrientName', '')

        # Check by ID first
        if nutrient_id in nutrient_map:
            key = nutrient_map[nutrient_id]
            # Ensure value exists and is numeric, default to 0 if missing/invalid
            value = nutrient.get('value', 0)
            nutrients[key] = float(value) if isinstance(value, (int, float)) else 0
            found_nutrients.add(key)
            continue # Go to next nutrient once mapped by ID

        # Optional: Fallback check by name (less reliable) - uncomment if needed
        # for map_id, map_key in nutrient_map.items():
        #     if map_key not in found_nutrients and map_key in nutrient_name.lower():
        #          value = nutrient.get('value', 0)
        #          nutrients[map_key] = float(value) if isinstance(value, (int, float)) else 0
        #          found_nutrients.add(map_key)
        #          break # Stop checking names for this nutrient

    # Validate that we have at least calories or some other key nutrient
    required_present = ['calories'] # Or maybe ['calories', 'protein', 'fat', 'carbs']
    if not any(k in found_nutrients for k in required_present):
        logging.warning(f"Required nutrient(s) ({required_present}) not found for FDC ID {nutrients.get('fdcId')}, Description: {nutrients['description']}")
        # Decide if you want to return None or partial data
        # return None # Option 1: Reject if key nutrients missing
        pass # Option 2: Allow partial data if description is present

    # Ensure all expected keys exist, setting default 'N/A' if missing after processing
    for key in nutrient_map.values():
        if key not in nutrients:
            nutrients[key] = 'N/A'

    return nutrients

# =========================
# Configuration
# =========================
NUM_SAMPLES_PER_CLASS_TO_TEST = 3
CONFIDENCE_THRESHOLD = 0.60
UNSEEN_SAMPLE_DIR_NAME = "food-101-unseen-trained-plus-unknown-samples"
UNKNOWN_CLASS_DIR_NAME = "unknown"
TRAIN_VAL_SPLIT_DIR_NAME = "dataset_split"
FILTERED_DATA_BASE_DIR_NAME = "food-101-filtered-5main-unknown"
PREDICTION_VIS_DIR_NAME = "prediction_visualization" # <-- New directory name

# =========================
# Logging Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Path Setup (MODIFIED)
# =========================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    script_dir = os.getcwd()
    if os.path.basename(script_dir).lower() == 'scripts':
         project_root = os.path.dirname(script_dir)
    else:
         project_root = script_dir
    logging.warning(f"Could not determine script directory reliably, assuming project root is: {project_root}")

# Model directory
model_dir = os.path.join(project_root, "models")
# Directory containing the unseen/unknown samples for testing
unseen_sample_dataset_dir = os.path.join(project_root, UNSEEN_SAMPLE_DIR_NAME)
# Directory containing the actual train/val splits used for training
split_base_dir = os.path.join(project_root, FILTERED_DATA_BASE_DIR_NAME, TRAIN_VAL_SPLIT_DIR_NAME)
# Specific path to the training data directory
train_dir_path = os.path.join(split_base_dir, "train")
# --- NEW: Path for prediction visualization output ---
prediction_vis_dir = os.path.join(project_root, PREDICTION_VIS_DIR_NAME)


# --- Define paths to the necessary files ---
MODEL_PATH = os.path.join(model_dir, "best_food_classifier.h5")
CLASS_INDICES_PATH = os.path.join(model_dir, "class_indices.pkl")

# --- Define image size (must match training) ---
IMG_SIZE = (224, 224) # Note: Pillow uses (width, height), TF uses (height, width) sometimes. Be careful.

logging.info(f"Model directory: {model_dir}")
logging.info(f"Unseen/Unknown sample dataset directory: {unseen_sample_dataset_dir}")
logging.info(f"Training data directory: {train_dir_path}")
logging.info(f"Class indices path: {CLASS_INDICES_PATH}")
logging.info(f"Model path to load: {MODEL_PATH}")
logging.info(f"Prediction visualization output directory: {prediction_vis_dir}") # <-- Log new path

# --- Create the output directory for visualizations ---
try:
    os.makedirs(prediction_vis_dir, exist_ok=True)
    logging.info(f"Ensured prediction visualization directory exists: {prediction_vis_dir}")
except OSError as e:
    logging.error(f"Could not create directory for prediction visualizations: {prediction_vis_dir} - {e}")
    # Decide if you want to exit or just disable visualization
    # exit(1) # Option: Exit if directory cannot be created
    prediction_vis_dir = None # Option: Disable visualization if dir fails
    logging.warning("Prediction visualization disabled.")


# =========================
# Load Model and Class Indices
# =========================
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at: {MODEL_PATH}")
    exit(1)
try:
    logging.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

if not os.path.exists(CLASS_INDICES_PATH):
    logging.error(f"Class indices file not found at: {CLASS_INDICES_PATH}")
    exit(1)
try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    class_names_map = {v: k for k, v in class_indices.items()}
    model_trained_classes = sorted(list(class_names_map.values()))
    main_trained_classes = sorted([name for name in model_trained_classes if name != UNKNOWN_CLASS_DIR_NAME])
    logging.info(f"Loaded class indices. Model expects {len(model_trained_classes)} classes: {model_trained_classes}")
    logging.info(f"Identified {len(main_trained_classes)} main trained classes: {main_trained_classes}")
except Exception as e:
    logging.error(f"Error loading class indices: {e}")
    exit(1)

# =========================
# Find Classes in Sample Dataset
# =========================
classes_in_unseen_sample_dir = []
if not os.path.isdir(unseen_sample_dataset_dir):
    logging.warning(f"Unseen sample dataset directory not found: {unseen_sample_dataset_dir}")
else:
    try:
        classes_in_unseen_sample_dir = sorted([d for d in os.listdir(unseen_sample_dataset_dir)
                                       if os.path.isdir(os.path.join(unseen_sample_dataset_dir, d))])
        if not classes_in_unseen_sample_dir:
             logging.warning(f"No class subdirectories found in {unseen_sample_dataset_dir}")
        else:
             logging.info(f"Found classes in unseen sample dataset to test: {classes_in_unseen_sample_dir}")
    except OSError as e:
         logging.error(f"Error reading {unseen_sample_dataset_dir}: {e}")

# =========================
# Select Random Images Function
# =========================
def get_random_images_from_class(class_dir_path, num_to_select):
    selected_paths = []
    if not os.path.isdir(class_dir_path):
        logging.warning(f"Directory not found during image selection: {class_dir_path}")
        return []
    try:
        images = [f for f in os.listdir(class_dir_path)
                  if os.path.isfile(os.path.join(class_dir_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            logging.warning(f"No image files found in directory: {class_dir_path}")
            return []
        num_available = len(images)
        actual_num_to_select = min(num_available, num_to_select)
        if num_available < num_to_select:
            logging.warning(f"Directory '{os.path.basename(class_dir_path)}' only has {num_available} images, requested {num_to_select}. Selecting all available.")
        selected_filenames = random.sample(images, actual_num_to_select)
        selected_paths = [os.path.join(class_dir_path, fname) for fname in selected_filenames]
    except OSError as e:
        logging.warning(f"Error accessing directory {class_dir_path} during image selection: {e}")
    return selected_paths

# =========================
# Image Preprocessing Function
# =========================
def preprocess_single_image(img_path, target_size):
    """Loads and preprocesses a single image for model prediction."""
    try:
        # Keras load_img uses (height, width) but target_size here is (width, height) for Pillow later
        # Ensure target_size for load_img is correct: (height, width)
        img = image.load_img(img_path, target_size=(target_size[1], target_size[0]))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        return img_preprocessed
    except FileNotFoundError:
        logging.error(f"Image file not found at: {img_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

# =========================
# Visualization Function  <-- NEW FUNCTION
# =========================
def create_prediction_visualization(original_image_path, text_lines, output_path):
    """Creates an image combining the original photo and prediction text."""
    try:
        # Load the original image
        img = Image.open(original_image_path)
        img_width, img_height = img.size

        # --- Font and Text Setup ---
        padding = 10
        line_spacing = 5
        font_size = 15
        try:
            # Try loading a common sans-serif font
            # Adjust path if needed for your system (e.g., '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' on Linux)
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            logging.warning("Arial font not found, using default PIL font.")
            try:
                font = ImageFont.load_default() # Fallback to default, size control is limited
                # Adjust effective line spacing/padding if needed for default font
            except Exception as e:
                 logging.error(f"Could not load default font: {e}")
                 return # Cannot proceed without a font

        # --- Calculate Text Block Size ---
        draw = ImageDraw.Draw(img) # Dummy draw object to measure text
        max_text_width = 0
        total_text_height = 0
        wrapped_lines = []

        # Estimate wrap width (slightly less than image width)
        # Adjust this character count based on font/image size if needed
        wrap_width_chars = int((img_width - 2 * padding) / (font_size * 0.6)) # Rough estimate

        for line in text_lines:
            # Wrap long lines
            wrapped = textwrap.fill(line, width=max(10, wrap_width_chars)) # Ensure width is positive
            for sub_line in wrapped.split('\n'):
                # Use getbbox for more accurate size in newer Pillow versions
                try:
                    line_bbox = draw.textbbox((0, 0), sub_line, font=font)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_height = line_bbox[3] - line_bbox[1]
                except AttributeError: # Fallback for older Pillow versions
                    line_width, line_height = draw.textsize(sub_line, font=font)

                max_text_width = max(max_text_width, line_width)
                total_text_height += line_height + line_spacing
                wrapped_lines.append(sub_line) # Store the processed lines

        total_text_height -= line_spacing # Remove last spacing
        text_block_height = total_text_height + 2 * padding
        text_block_width = max_text_width + 2 * padding

        # --- Create Canvas ---
        # Make canvas wide enough for image OR text, whichever is wider
        canvas_width = max(img_width, text_block_width)
        canvas_height = img_height + text_block_height

        # Create a new white canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Paste the original image at the top-left
        canvas.paste(img, (0, 0))

        # --- Draw Text ---
        draw = ImageDraw.Draw(canvas)
        current_y = img_height + padding
        for sub_line in wrapped_lines:
             # Use textbbox again to get height for positioning this specific line
            try:
                line_bbox = draw.textbbox((0, 0), sub_line, font=font)
                line_height = line_bbox[3] - line_bbox[1]
            except AttributeError:
                _, line_height = draw.textsize(sub_line, font=font)

            draw.text((padding, current_y), sub_line, fill='black', font=font)
            current_y += line_height + line_spacing

        # --- Save the image ---
        canvas.save(output_path)
        logging.info(f"Saved prediction visualization to: {output_path}")

    except FileNotFoundError:
        logging.error(f"Original image not found during visualization: {original_image_path}")
    except Exception as e:
        logging.error(f"Failed to create prediction visualization for {original_image_path}: {e}")


# =========================
# Prediction Function (MODIFIED)
# =========================
# Added output_vis_dir parameter
def predict_and_print(image_path, actual_class_name, image_type_label, threshold, output_vis_dir):
    """Preprocesses, predicts, prints results, and saves visualization."""
    logging.info(f"--- Predicting {image_type_label} Image ---")
    logging.info(f"Image: {os.path.basename(image_path)} (Actual Class: {actual_class_name})")

    # Use IMG_SIZE defined globally (width, height) for Pillow, but TF needs (height, width)
    tf_target_size = (IMG_SIZE[1], IMG_SIZE[0])
    preprocessed_image = preprocess_single_image(image_path, tf_target_size)

    # Initialize variables
    predicted_class_name_for_vis = "Error"
    confidence_score_for_vis = 0.0
    nutrition_data = None
    prediction_status = "Error processing image" # Default status text

    if preprocessed_image is not None:
        try:
            predictions = model.predict(preprocessed_image, verbose=0)
            predicted_index = np.argmax(predictions[0])
            top_predicted_class_name = class_names_map.get(predicted_index, "ModelOutputIndexError")
            confidence_score = predictions[0][predicted_index]

            # Store for visualization
            confidence_score_for_vis = confidence_score

            # Determine final predicted class based on threshold
            if confidence_score >= threshold:
                predicted_class_name_for_vis = top_predicted_class_name
                prediction_status = f"Predicted: {top_predicted_class_name} ({confidence_score:.1%})"
                # Get nutritional data only if prediction is confident
                nutrition_data = get_usda_nutrition(top_predicted_class_name)
            else:
                predicted_class_name_for_vis = "Other" # Use "Other" for filename/vis text
                prediction_status = f"Predicted: Other (Conf: {confidence_score:.1%}, Threshold: {threshold:.0%})"
                # Optionally include the top prediction before thresholding
                prediction_status += f"\n(Top guess: {top_predicted_class_name})"


        except Exception as e:
             logging.error(f"Error during prediction for {image_path}: {e}")
             prediction_status = "Error during prediction"
             predicted_class_name_for_vis = "PredictionError"


    # --- Print to Console ---
    print(f"\nResults for {image_type_label} Image:")
    print(f"  File: {os.path.basename(image_path)}")
    print(f"  Actual Class: {actual_class_name}")

    if preprocessed_image is None:
         print(f"  Status: {prediction_status}") # Error processing image
    elif predicted_class_name_for_vis == "PredictionError":
        print(f"  Status: {prediction_status}") # Error during prediction
    elif predicted_class_name_for_vis == "Other":
         print(f"  Predicted Class: Other (Below Threshold)")
         print(f"  Confidence: {confidence_score_for_vis:.2%} (Threshold: {threshold:.0%})")
         # Find the actual top prediction index/name again if needed for console output
         top_pred_idx_console = np.argmax(predictions[0])
         top_pred_name_console = class_names_map.get(top_pred_idx_console, "N/A")
         print(f"  (Top prediction was: {top_pred_name_console})")
    else: # Confident prediction
        print(f"  Predicted Class: {predicted_class_name_for_vis}")
        print(f"  Confidence: {confidence_score_for_vis:.2%}")


    # Add nutritional information to console output
    if nutrition_data:
        print("\n  Nutritional Information (Per 100g Estimate):")
        print(f"  Food Match: {nutrition_data.get('description', 'N/A')}")
        print(f"  Calories: {nutrition_data.get('calories', 'N/A')} kcal")
        print(f"  Protein: {nutrition_data.get('protein', 'N/A')} g")
        print(f"  Fat: {nutrition_data.get('fat', 'N/A')} g")
        print(f"  Carbohydrates: {nutrition_data.get('carbs', 'N/A')} g")
    elif predicted_class_name_for_vis not in ["Error", "PredictionError", "Other"]:
        print("\n  Nutritional data unavailable or lookup failed for this prediction.")

    # Add notes to console output
    note = ""
    if "Unknown Class" in image_type_label:
         note = "Actual image is from 'unknown' sample category"
    elif "Unseen Trained" in image_type_label:
         note = "Image from trained class, unseen during training"
    elif "Seen Training" in image_type_label:
        note = "Image from the actual training set"
    else:
         note = f"Actual class '{actual_class_name}' tested"
    print(f"  (Note: {note})")


    # --- Generate and Save Visualization ---
    # Only proceed if visualization directory is set and image was processed
    if output_vis_dir and preprocessed_image is not None:
        # Prepare text lines for the visualization image
        vis_text_lines = [
            f"File: {os.path.basename(image_path)}",
            f"Actual Class: {actual_class_name}",
            f"Type: {image_type_label}",
            "-"*20, # Separator
            prediction_status # Contains predicted class and confidence
        ]

        # Add nutrition details if available
        if nutrition_data:
            vis_text_lines.append("\nNutrition (Per 100g Estimate):")
            vis_text_lines.append(f" Food: {nutrition_data.get('description', 'N/A')}")
            vis_text_lines.append(f" Calories: {nutrition_data.get('calories', 'N/A')} kcal")
            vis_text_lines.append(f" Protein: {nutrition_data.get('protein', 'N/A')} g")
            vis_text_lines.append(f" Fat: {nutrition_data.get('fat', 'N/A')} g")
            vis_text_lines.append(f" Carbs: {nutrition_data.get('carbs', 'N/A')} g")
        elif predicted_class_name_for_vis not in ["Error", "PredictionError", "Other"]:
             vis_text_lines.append("\nNutrition: Data unavailable.")

        vis_text_lines.append(f"\nNote: {note}") # Add the same note

        # Construct a meaningful output filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        # Use the predicted class name (or 'Other'/'Error') in the filename
        output_filename = f"{base_filename}_PRED_{predicted_class_name_for_vis.replace(' ', '_')}.png" # Save as PNG
        output_image_path = os.path.join(output_vis_dir, output_filename)

        # Call the visualization function
        create_prediction_visualization(image_path, vis_text_lines, output_image_path)

    elif not output_vis_dir:
        logging.warning("Visualization disabled because output directory is not set.")

    print("-" * 40) # Separator for console output

# =========================
# Main Execution
# =========================
if __name__ == "__main__":

    print("\n" + "="*10 + f" TESTING IMAGES " + "="*10)
    print(f"Using model: {MODEL_PATH}")
    print(f"Confidence Threshold for 'Other': {CONFIDENCE_THRESHOLD:.0%}")
    print(f"Testing {NUM_SAMPLES_PER_CLASS_TO_TEST} images per class found...")
    if prediction_vis_dir: # Check if visualization is enabled
        print(f"Saving visualization images to: {prediction_vis_dir}")
    else:
        print("Visualization image saving is disabled (output directory issue).")

    # --- 1. Test images the model was TRAINED on ---
    print("\n" + "="*10 + " CATEGORY 1: SEEN TRAINING SAMPLES " + "="*10)
    if not os.path.isdir(train_dir_path):
        logging.warning(f"Training directory not found ({train_dir_path}). Skipping tests on seen training samples.")
    else:
        for class_name in main_trained_classes:
            class_path = os.path.join(train_dir_path, class_name)
            logging.info(f"\nSelecting training images for class: {class_name}")
            images_to_test = get_random_images_from_class(class_path, NUM_SAMPLES_PER_CLASS_TO_TEST)
            if not images_to_test:
                logging.warning(f"No training images selected for class '{class_name}'. Skipping.")
                continue
            for img_path in images_to_test:
                # Pass the visualization output directory
                predict_and_print(img_path, class_name, "Seen Training Sample", CONFIDENCE_THRESHOLD, prediction_vis_dir)

    # --- 2. Test UNSEEN images from TRAINED classes ---
    print("\n" + "="*10 + " CATEGORY 2: UNSEEN TRAINED CLASS SAMPLES " + "="*10)
    if not classes_in_unseen_sample_dir:
         logging.warning("No classes found in unseen sample dataset directory. Skipping these tests.")
    else:
        for class_name in classes_in_unseen_sample_dir:
            if class_name == UNKNOWN_CLASS_DIR_NAME:
                continue
            if class_name not in model_trained_classes:
                 logging.warning(f"Class '{class_name}' found in sample dir, but not in model's trained classes. Skipping.")
                 continue

            class_path = os.path.join(unseen_sample_dataset_dir, class_name)
            logging.info(f"\nSelecting unseen images for trained class: {class_name}")
            images_to_test = get_random_images_from_class(class_path, NUM_SAMPLES_PER_CLASS_TO_TEST)
            if not images_to_test:
                logging.warning(f"No images selected for class '{class_name}' from unseen sample dir. Skipping.")
                continue
            for img_path in images_to_test:
                 # Pass the visualization output directory
                predict_and_print(img_path, class_name, "Unseen Trained Class Sample", CONFIDENCE_THRESHOLD, prediction_vis_dir)


    # --- 3. Test images from the UNKNOWN class sample ---
    print("\n" + "="*10 + " CATEGORY 3: UNKNOWN CLASS SAMPLES " + "="*10)
    unknown_class_path_in_sample = os.path.join(unseen_sample_dataset_dir, UNKNOWN_CLASS_DIR_NAME)
    if UNKNOWN_CLASS_DIR_NAME not in classes_in_unseen_sample_dir:
         logging.warning(f"Directory for '{UNKNOWN_CLASS_DIR_NAME}' not found in unseen sample dataset. Skipping these tests.")
    else:
        logging.info(f"\nSelecting images from the '{UNKNOWN_CLASS_DIR_NAME}' sample class...")
        images_to_test = get_random_images_from_class(unknown_class_path_in_sample, NUM_SAMPLES_PER_CLASS_TO_TEST)
        if not images_to_test:
            logging.warning(f"No images selected for class '{UNKNOWN_CLASS_DIR_NAME}' from unseen sample dir. Skipping.")
        else:
            for img_path in images_to_test:
                 # Pass the visualization output directory
                predict_and_print(img_path, UNKNOWN_CLASS_DIR_NAME, "Unknown Class Sample", CONFIDENCE_THRESHOLD, prediction_vis_dir)


    logging.info("Prediction script finished.")