import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Step 1: Load Data
def load_data(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary'
    )

    return train_gen, val_gen

train_dir = os.path.join(os.getcwd(), "dataset", "train")
val_dir = os.path.join(os.getcwd(), "dataset", "val")
train_gen, val_gen = load_data(train_dir, val_dir)

# Step 2: Build Model
def build_food_classifier(input_shape=(224, 224, 3), num_classes=1):  # Single neuron for binary classification
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='sigmoid')  # Single neuron for binary classification
    ])
    return model

model = build_food_classifier()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Step 3: Train Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen)
)

# Step 4: Save Model
os.makedirs("models", exist_ok=True)  # Ensure models directory exists
model.save(os.path.join("models", "rice_classifier.h5"))

# Save history to a file
with open('models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)