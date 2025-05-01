import os
import pickle
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def test_model_basic_functionality():
    """Verify model can load and make valid predictions"""
    # Paths relative to project root
    MODEL_PATH = os.path.join("models", "best_food_classifier.keras")
    CLASS_INDICES_PATH = os.path.join("models", "class_indices.pkl")
    TEST_IMAGE_PATH = os.path.join("tests", "test_image.jpg")
    
    # 1. Verify required files exist
    assert os.path.exists(MODEL_PATH), "Model file missing"
    assert os.path.exists(CLASS_INDICES_PATH), "Class indices missing"
    assert os.path.exists(TEST_IMAGE_PATH), "Test image missing"
    
    # 2. Load model and class mapping
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_INDICES_PATH, "rb") as f:
        class_indices = pickle.load(f)
    
    # 3. Verify prediction on test image
    img = image.load_img(TEST_IMAGE_PATH, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    
    # 4. Validate prediction format
    assert preds.shape == (1, len(class_indices)), "Invalid output shape"
    assert np.all(preds >= 0), "Negative predictions found"
    assert np.all(preds <= 1), "Predictions exceed 1.0"
