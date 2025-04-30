import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet152 # Keep the import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Layer # Import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import mixed_precision
import pickle
import logging # Optional: for better messages

# Setup logging (Optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
logging.info(f"Using mixed precision policy: {policy.name}")

# Define directories (Ensure these paths are correct for your environment)
# Using absolute path based on script location for potentially better robustness
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively)
    script_dir = os.getcwd()
    logging.warning(f"__file__ not found, using current working directory: {script_dir}")

project_root = os.path.dirname(script_dir) # Assumes script is in a 'scripts' folder, adjust if needed
data_dir = os.path.join(project_root, "food-101", "dataset_split")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)
logging.info(f"Train directory: {train_dir}")
logging.info(f"Validation directory: {val_dir}")
logging.info(f"Model directory: {model_dir}")


# Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
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

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

num_classes = train_gen.num_classes
logging.info(f"Found {num_classes} classes.")

# Model Building Function
def build_food_classifier(input_shape=(224, 224, 3), num_classes=6):
    # Define the base model using the imported ResNet152 function
    # *** Assign an explicit name here ***
    resnet_base = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape, name="resnet152_base")
    resnet_base.trainable = False # Start with base frozen
    # Build the Sequential model
    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        resnet_base,               # Add the base model instance here (this should be layer 1)
        Flatten(name="flatten_layer"),
        Dropout(0.5, name="dropout_layer"),
        Dense(512, activation='relu', name="dense_layer_1"),
        # Use float32 for output layer with mixed precision for stability
        Dense(num_classes, activation='softmax', dtype='float32', name="output_layer")
    ], name="FoodClassifier")
    return model

# Build and compile the model for initial training
model = build_food_classifier(num_classes=num_classes)
optimizer_initial = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_initial, loss='categorical_crossentropy', metrics=['accuracy'])
logging.info("Model compiled for initial training.")
model.summary(show_trainable=True) # Show trainable status in summary

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint_path = os.path.join(model_dir, 'best_food_classifier.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger_path = os.path.join(model_dir, 'training_log.csv')
csv_logger = CSVLogger(csv_logger_path) # Overwrites by default

# Initial Training (Feature Extraction)
logging.info("Starting initial training (feature extraction)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10, # Adjust as needed
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[early_stopping, checkpoint, csv_logger]
)
logging.info("Initial training finished.")


# === Fine-tuning Section ===
logging.info("Setting up for fine-tuning...")

# --- FIX: Find base model layer reliably by NAME ---
base_model_layer = None
target_base_model_name = "resnet152_base" # The name assigned in build_food_classifier

for layer in model.layers:
    # Check if the layer name matches the target name
    if layer.name == target_base_model_name:
        # Found the layer, double check it's a Keras Layer/Model instance
        if isinstance(layer, Layer): # Check against the base Layer class
             base_model_layer = layer
             logging.info(f"Identified base model layer by name: '{layer.name}'")
             break # Exit loop once found
        else:
             logging.warning(f"Found layer with name '{layer.name}' but it's not a Keras Layer instance? Type: {type(layer)}")
             # Continue searching just in case, but this is unexpected

# Check if the base model layer was found
if base_model_layer is None:
    layer_names = [l.name for l in model.layers]
    raise ValueError(f"Could not find the base model layer named '{target_base_model_name}'. Found layers: {layer_names}. Check model definition and naming.")
# --- End FIX ---


# Set the identified base model layer to trainable
base_model_layer.trainable = True
logging.info(f"Base model layer '{base_model_layer.name}' set to trainable.")

# Freeze all layers *within* the base model except the last 30
# Check if base_model_layer actually has internal layers (it should as it's a functional model)
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
                internal_layer.trainable = True # Ensure last 30 are trainable
    else:
        logging.warning(f"Base model '{base_model_layer.name}' has {num_base_layers} layers or fewer. Cannot freeze all but last 30. All internal layers remain trainable.")

# Re-compile the model with a lower learning rate for fine-tuning
optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=1e-5) # Use a smaller LR
model.compile(optimizer=optimizer_finetune, loss='categorical_crossentropy', metrics=['accuracy'])
logging.info("Model re-compiled for fine-tuning with lower learning rate.")
model.summary(show_trainable=True) # Optional: Check trainable params again

# Continue training (Fine-tuning)
logging.info("Starting fine-tuning...")
# Append to the existing log file
csv_logger_finetune = CSVLogger(csv_logger_path, append=True)

# Determine starting epoch for fine-tuning phase
# Use history.epoch which contains the actual epochs executed (accounts for early stopping)
if history.epoch:
    initial_fine_tune_epoch = history.epoch[-1] + 1
else:
    initial_fine_tune_epoch = 0 # Should not happen if initial training ran
    logging.warning("Initial training history is empty. Starting fine-tuning from epoch 0.")

fine_tune_epochs = 10 # Number of epochs for fine-tuning stage

finetune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_fine_tune_epoch + fine_tune_epochs, # Total epochs needed relative to start of script
    initial_epoch=initial_fine_tune_epoch, # Start counting from here for this phase
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    # Reuse early stopping and checkpoint. Use the logger that appends.
    callbacks=[early_stopping, checkpoint, csv_logger_finetune]
)
logging.info("Fine-tuning finished.")

# Combine history data (important for correct plotting later)
combined_history = history.history.copy() # Start with initial history
# Make sure fine-tuning actually ran and produced history
if hasattr(finetune_history, 'history'):
    for key in finetune_history.history:
        if key in combined_history:
            combined_history[key].extend(finetune_history.history[key])
        else:
            # Handle keys that might only appear in fine-tuning (less common)
            combined_history[key] = [None] * len(history.epoch) # Pad initial phase
            combined_history[key].extend(finetune_history.history[key])
else:
    logging.warning("Fine-tuning history object not found or empty. Combined history may be incomplete.")


# Save the *final* model state (after fine-tuning)
final_model_path = os.path.join(model_dir, "food_classifier_final.keras")
try:
    model.save(final_model_path)
    logging.info(f"Final fine-tuned model saved successfully to {final_model_path}")
except Exception as e:
    logging.error(f"Error saving final model to {final_model_path}: {e}")


# Save the *combined* training history
history_path = os.path.join(model_dir, 'training_history.pkl')
try:
    with open(history_path, 'wb') as f:
        pickle.dump(combined_history, f) # Save the combined dict
    logging.info(f"Combined training history saved successfully to {history_path}")
except Exception as e:
    logging.error(f"Error saving training history to {history_path}: {e}")


# Save class indices (no change needed here)
class_indices_path = os.path.join(model_dir, 'class_indices.pkl')
try:
    with open(class_indices_path, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)
    logging.info(f"Class indices saved successfully to {class_indices_path}")
except Exception as e:
    logging.error(f"Error saving class indices to {class_indices_path}: {e}")


logging.info("Script completed.")