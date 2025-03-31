import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle
import matplotlib.pyplot as plt

# Step 1: Load Data with Augmentation
def load_data(train_dir, val_dir):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    return train_gen, val_gen

train_dir = os.path.join(os.getcwd(), "dataset_split", "train")
val_dir = os.path.join(os.getcwd(), "dataset_split", "val")
train_gen, val_gen = load_data(train_dir, val_dir)

# Step 2: Build Model with Transfer Learning
def build_food_classifier(input_shape=(224, 224, 3), num_classes=4):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers initially

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')  # Use 'softmax' for multi-class classification
    ])
    
    return model

model = build_food_classifier()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for Optimization
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)

# Step 3: Train Model with Callbacks
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[lr_schedule, early_stopping]
)


# check epochs run:
print(len(history.history['accuracy']))

save_path = 'training_history.png'

history = history.history

print(history.keys())  # Ensure it has ['accuracy', 'val_accuracy', 'loss', 'val_loss']
print(history['accuracy'])
print(len(history['accuracy']), len(history['val_accuracy']))
print(len(history['loss']), len(history['val_loss']))



def plot_training_history(history):
    epochs_range = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['accuracy'], marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], marker='s', linestyle='--', label='Val Accuracy', color='red')
    plt.xticks(epochs_range)  # Force x-axis to show all epochs
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['loss'], marker='o', linestyle='-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], marker='s', linestyle='--', label='Val Loss', color='red')
    plt.xticks(epochs_range)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Training & Validation Loss')


    plt.savefig(save_path)

plot_training_history(history)

# Step 4: Save Model and History
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "food_classifier_256.keras"))

with open('models/training_history_256.pkl', 'wb') as f:
    pickle.dump(history.history, f)