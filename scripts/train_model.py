from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.food_classifier import build_food_classifier, compile_model

def train_food_classifier(train_dir, val_dir):
    datagen = ImageDataGenerator(rescale=1./255)

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

    model = build_food_classifier(num_classes=len(train_gen.class_indices))
    model = compile_model(model)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen)
    )

    model.save('models/food_classifier.h5')
