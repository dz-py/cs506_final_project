from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
USDA_API_KEY = os.getenv("USDA_API_KEY")

def get_food_nutrition(food_name):
    """
    Fetch nutritional information for a given food name using the USDA FoodData Central API.
    """
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={USDA_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'foods' in data and len(data['foods']) > 0:
            food_item = data['foods'][0]
            nutrients = {nutrient['nutrientName']: nutrient['value'] for nutrient in food_item.get('foodNutrients', [])}
            return {
                "description": food_item.get("description"),
                "calories": nutrients.get("Energy", "N/A"),
                "protein": nutrients.get("Protein", "N/A"),
                "carbs": nutrients.get("Carbohydrate, by difference", "N/A"),
                "fats": nutrients.get("Total lipid (fat)", "N/A"),
            }
        else:
            return {"error": "Food not found"}
    else:
        return {"error": f"API request failed with status code {response.status_code}"}

# Example usage:
# nutrition_info = get_food_nutrition("apple")
# print(nutrition_info)
