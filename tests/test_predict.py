import os
import pickle
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def test_model_basic_functionality():
    """Verify model can load and make valid predictions"""
    # Paths relative to project root
    MODEL_PATH = os.path.join("models", "best_food_classifier.h5")
    CLASS_INDICES_PATH = os.path.join("models", "class_indices.pkl")
    TEST_IMAGE_PATH = os.path.join("tests", "test_image.jpg")
    
    # 1. Verify required files exist
    assert os.path.exists(MODEL_PATH), "Model file missing"
    assert os.path.exists(CLASS_INDICES_PATH), "Class indices file missing"
    assert os.path.exists(TEST_IMAGE_PATH), "Test image missing"
    
    # 2. Load model and class mapping
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_INDICES_PATH, "rb") as f:
        class_indices = pickle.load(f)
    
    # 3. Verify model structure
    assert model is not None, "Model failed to load"
    assert len(model.layers) > 0, "Model has no layers"
    assert model.input_shape[1:] == (224, 224, 3), "Model input shape mismatch"
    
    # 4. Verify predictions on test image
    img = tf.keras.preprocessing.image.load_img(TEST_IMAGE_PATH, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    assert predictions.shape[1] == len(class_indices), "Prediction shape mismatch with number of classes"
    assert np.all(predictions >= 0) and np.all(predictions <= 1), "Predictions not in [0,1] range"
    assert np.isclose(np.sum(predictions[0]), 1.0, rtol=1e-5), "Predictions don't sum to 1"
