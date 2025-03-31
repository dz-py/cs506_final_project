# Food Recognition and Macro Estimation using Trained Model

## Project Description
This project aims to create a model that analyzes food images to estimate calories and macronutrients (protein, carbs, fats). Users can take a picture of their meal, and the model will provide corresponding nutritional information.

## Project Goals
- Identify food items from images.
- Estimate macronutrients and calorie content.
- Build a machine learning model for food recognition.

## Data Collection
We will gather images and nutritional data from:
- Open source food datasets (e.g., Food-101, UEC FOOD 256).
- Nutrition databases like USDA FoodData Central.

## Model Training
- Use deep learning (e.g., CNNs like ResNet) for food classification.
- Implement weight estimation techniques using image processing and reference data.
- Retrieve nutritional data via an external API.
- Implement and test using TensorFlow/PyTorch/Sckit-Learn.

## Data Visualization
- Display food recognition results with estimated weight, calorie, and macronutrient breakdown.
- Create pie charts to show the proportion of macronutrients (protein, carbs, fats) in each identified food item.
- Implement bar graphs to compare calorie content across different food items in a meal.
- Utilize residual plots to measure accuracy of the prediction to test the model
- Isolate parameters to examine the importance of each parameter on accuracy
- Design a simple, color-coded nutritional label for each recognized food item, similar to standard food packaging labels.

## Test Plan
- Use 80% of data for training and 20% for testing.
- Evaluate accuracy using real-world images.
- Compare model estimates with actual nutrition data.
