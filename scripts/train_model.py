import os
import tensorflow as tf
# Suppress TensorFlow INFO/WARNING messages (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARNING')

from tensorflow.keras.applications import ResNet152 # Keep the import for transfer learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Layer # Import Layer for type checking
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import mixed_precision
import pickle
import logging # Optional: for better messages

# =========================
# Logging setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# Mixed Precision Setup
# =========================
# Enable mixed precision for faster training and reduced memory usage on supported hardware
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
logging.info(f"Using mixed precision policy: {policy.name}")

# =========================
# Directory Setup (MODIFIED)
# =========================
# Determine paths for data and model saving based on script location
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

# **MODIFIED:** Point data_dir to the 'dataset_split' folder *inside* the sample directory
data_dir = os.path.join(project_root, "food-101-filtered-5main-unknown", "dataset_split")
train_dir = os.path.join(data_dir, "train") # Train dir is inside dataset_split
val_dir = os.path.join(data_dir, "val")     # Val dir is inside dataset_split

# Model directory remains the same
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)

logging.info(f"Train directory: {train_dir}")
logging.info(f"Validation directory: {val_dir}")
logging.info(f"Model directory: {model_dir}")

# =========================
# Data Augmentation & Generators
# =========================
# (Data Augmentation settings remain the same)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input, # Preprocessing for ResNet
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0
)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

# Create generators that yield batches of images and labels from directories
# Check if train/val directories exist before creating generators
if not os.path.isdir(train_dir):
     logging.error(f"Training directory not found: {train_dir}")
     logging.error("Please ensure split_dataset.py ran successfully and created the train/val folders inside the sample dataset's dataset_split directory.")
     exit(1)
if not os.path.isdir(val_dir):
    logging.error(f"Validation directory not found: {val_dir}")
    logging.error("Please ensure split_dataset.py ran successfully and created the train/val folders inside the sample dataset's dataset_split directory.")
    exit(1)

try:
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
except Exception as e:
     logging.error(f"Error creating data generators: {e}")
     exit(1)


num_classes = train_gen.num_classes
logging.info(f"Found {num_classes} classes.") # Should now report 6 (5 main + 1 unknown)

# =========================
# Model Building Function
# =========================
# Default value for num_classes is changed for clarity, but it's overridden anyway
def build_food_classifier(input_shape=(224, 224, 3), num_classes=6):
    """
    Build a Sequential Keras model for food classification using ResNet152 as the base.
    The base is frozen initially for feature extraction.
    """
    # Load ResNet152 without its top classifier layers, using pretrained ImageNet weights
    resnet_base = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape, name="resnet152_base")
    resnet_base.trainable = False # Freeze base model for initial training

    # Build the full model
    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        resnet_base,                         # Base feature extractor
        Flatten(name="flatten_layer"),       # Flatten output for dense layers
        Dropout(0.5, name="dropout_layer"),  # Dropout for regularization
        Dense(512, activation='relu', name="dense_layer_1"), # Dense layer for learning
        # Output layer: use float32 for numerical stability with mixed precision
        # This will use the dynamically detected num_classes (e.g., 6)
        Dense(num_classes, activation='softmax', dtype='float32', name="output_layer")
    ], name="FoodClassifier")
    return model

# =========================
# Model Compilation & Initial Training
# =========================
# Build and compile the model for initial feature extraction training
model = build_food_classifier(num_classes=num_classes) # Pass the detected number of classes
optimizer_initial = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_initial, loss='categorical_crossentropy', metrics=['accuracy'])
logging.info("Model compiled for initial training.")
model.summary(show_trainable=True) # Print model summary with trainable status

# =========================
# Callbacks Setup
# =========================
# (Callbacks remain the same)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint_path = os.path.join(model_dir, 'best_food_classifier.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger_path = os.path.join(model_dir, 'training_log.csv')
csv_logger = CSVLogger(csv_logger_path) # Overwrites by default

# =========================
# Initial Training (Feature Extraction)
# =========================
logging.info("Starting initial training (feature extraction)...")
# Check generator lengths
if len(train_gen) == 0 or len(val_gen) == 0:
     logging.error("Train or validation generator is empty. Cannot train.")
     exit(1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10, # Number of epochs for initial training
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[early_stopping, checkpoint, csv_logger]
)
logging.info("Initial training finished.")

# =========================
# Fine-tuning Preparation
# =========================
# (Fine-tuning logic remains the same)
logging.info("Setting up for fine-tuning...")
base_model_layer = None
target_base_model_name = "resnet152_base"
for layer in model.layers:
    if layer.name == target_base_model_name:
        if isinstance(layer, Layer):
            base_model_layer = layer
            logging.info(f"Identified base model layer by name: '{layer.name}'")
            break
        else:
            logging.warning(f"Found layer with name '{layer.name}' but it's not a Keras Layer instance? Type: {type(layer)}")
if base_model_layer is None:
    layer_names = [l.name for l in model.layers]
    raise ValueError(f"Could not find the base model layer named '{target_base_model_name}'. Found layers: {layer_names}. Check model definition and naming.")

# =========================
# Fine-tuning: Unfreeze Base Model
# =========================
# (Unfreezing logic remains the same)
base_model_layer.trainable = True
logging.info(f"Base model layer '{base_model_layer.name}' set to trainable.")
if not hasattr(base_model_layer, 'layers') or not base_model_layer.layers:
    logging.warning(f"Identified base model layer '{base_model_layer.name}' does not seem to have internal layers. Skipping internal layer freezing.")
else:
    num_base_layers = len(base_model_layer.layers)
    layers_to_freeze = num_base_layers - 30
    if layers_to_freeze > 0:
        logging.info(f"Freezing the first {layers_to_freeze} layers of the base model '{base_model_layer.name}'...")
        for i, internal_layer in enumerate(base_model_layer.layers):
            if i < layers_to_freeze:
                internal_layer.trainable = False
            else:
                internal_layer.trainable = True
    else:
        logging.warning(f"Base model '{base_model_layer.name}' has {num_base_layers} layers or fewer. Cannot freeze all but last 30. All internal layers remain trainable.")

# =========================
# Fine-tuning Compilation
# =========================
# (Compilation logic remains the same)
optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer_finetune, loss='categorical_crossentropy', metrics=['accuracy'])
logging.info("Model re-compiled for fine-tuning with lower learning rate.")
model.summary(show_trainable=True)

# =========================
# Fine-tuning Training
# =========================
# (Fine-tuning fit logic remains the same)
logging.info("Starting fine-tuning...")
csv_logger_finetune = CSVLogger(csv_logger_path, append=True)
if history.epoch:
    initial_fine_tune_epoch = history.epoch[-1] + 1
else:
    initial_fine_tune_epoch = 0
    logging.warning("Initial training history is empty. Starting fine-tuning from epoch 0.")
fine_tune_epochs = 10
finetune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_fine_tune_epoch + fine_tune_epochs,
    initial_epoch=initial_fine_tune_epoch,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[early_stopping, checkpoint, csv_logger_finetune]
)
logging.info("Fine-tuning finished.")

# =========================
# Combine Training Histories
# =========================
# (Combining logic remains the same)
combined_history = history.history.copy()
if hasattr(finetune_history, 'history'):
    for key in finetune_history.history:
        if key in combined_history:
            combined_history[key].extend(finetune_history.history[key])
        else:
            combined_history[key] = [None] * len(history.epoch)
            combined_history[key].extend(finetune_history.history[key])
else:
    logging.warning("Fine-tuning history object not found or empty. Combined history may be incomplete.")

# =========================
# Save Final Model, History, Indices
# =========================
# (Saving logic remains the same)
final_model_path = os.path.join(model_dir, "food_classifier_final.keras")
try:
    model.save(final_model_path)
    logging.info(f"Final fine-tuned model saved successfully to {final_model_path}")
except Exception as e:
    logging.error(f"Error saving final model to {final_model_path}: {e}")
history_path = os.path.join(model_dir, 'training_history.pkl')
try:
    with open(history_path, 'wb') as f:
        pickle.dump(combined_history, f)
    logging.info(f"Combined training history saved successfully to {history_path}")
except Exception as e:
    logging.error(f"Error saving training history to {history_path}: {e}")
class_indices_path = os.path.join(model_dir, 'class_indices.pkl')
try:
    with open(class_indices_path, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)
    logging.info(f"Class indices saved successfully to {class_indices_path}")
except Exception as e:
    logging.error(f"Error saving class indices to {class_indices_path}: {e}")

logging.info("Script completed.")
