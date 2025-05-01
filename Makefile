.PHONY: install test clean preprocess split train visualize predict all

# Variables
CONDA_ENV_NAME = food_recognition
PYTHON = conda run -n $(CONDA_ENV_NAME) python
PIP = conda run -n $(CONDA_ENV_NAME) pip

# Default target
all: install preprocess split train visualize

# Installation
install:
	conda create -n $(CONDA_ENV_NAME) python=3.8 -y
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

# Data preprocessing
preprocess:
	$(PYTHON) scripts/preprocess_data.py

# Dataset splitting
split:
	$(PYTHON) scripts/split_dataset.py

# Model training
train:
	$(PYTHON) scripts/train_model.py

# Visualization
visualize:
	$(PYTHON) scripts/visualize_results.py

# Prediction
predict:
	$(PYTHON) scripts/predict_food.py

# Clean up
clean:
	conda env remove -n $(CONDA_ENV_NAME)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 