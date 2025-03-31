from scripts.fetch_nutrition_api import get_food_nutrition

def predict_calories_and_macros(food_name: str, weight_grams: float) -> dict:
    """
    Predict calories and macronutrients (protein, carbs, fats) for a given food item and weight.

    Parameters:
        food_name (str): The name of the food item (e.g., "apple").
        weight_grams (float): The weight of the food item in grams.

    Returns:
        dict: A dictionary containing calculated calories, protein, carbs, and fats.
    """
    # Fetch nutritional information from the USDA API
    nutrition_data = get_food_nutrition(food_name)

    if "error" in nutrition_data:
        return {"error": nutrition_data["error"]}

    try:
        # Extract per-100g nutritional values
        calories_per_100g = float(nutrition_data.get("calories", 0))
        protein_per_100g = float(nutrition_data.get("protein", 0))
        carbs_per_100g = float(nutrition_data.get("carbs", 0))
        fats_per_100g = float(nutrition_data.get("fats", 0))

        # Scale values based on the provided weight
        scale_factor = weight_grams / 100
        return {
            "calories": round(calories_per_100g * scale_factor, 2),
            "protein": round(protein_per_100g * scale_factor, 2),
            "carbs": round(carbs_per_100g * scale_factor, 2),
            "fats": round(fats_per_100g * scale_factor, 2),
        }
    except ValueError:
        return {"error": "Invalid data received from API"}

# Example usage:
# result = predict_calories_and_macros("apple", 150)
# print(result)
