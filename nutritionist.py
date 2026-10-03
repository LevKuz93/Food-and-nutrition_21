#!/usr/bin/env python3
# nutritionist.py
import sys
from recipes import RecipeNutritionist

def main():
    if len(sys.argv) < 2:
        print("Usage: ./nutritionist.py <ingredient1, ingredient2, ...>")
        sys.exit(1)


    full_input = " ".join(sys.argv[1:])
        
    try:
        nutritionist = RecipeNutritionist()
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        sys.exit(1)

    
    # БОНУСНАЯ ЧАСТЬ: Обработка команды генерации меню
    
    if full_input.strip().lower() == 'menu':
        from recipes import RecipeMenuGenerator
        print("--- GENERATING DAILY MENU (BONUS ACTIVE) ---")
        generator = RecipeMenuGenerator(
            feature_names=nutritionist.feature_names, 
            df_nutrition=nutritionist.df_nutrition
        )
        menu_text = generator.generate_daily_menu()
        print(menu_text)
        sys.exit(0)

    
    raw_ingredients = [ing.strip().lower() for ing in full_input.split(',') if ing.strip()]
    
        
    try:
        valid_ingredients = nutritionist.validate_ingredients_list(raw_ingredients)
    except ValueError as e:
        print(e)
        sys.exit(0)
        
    print("I. OUR FORECAST")
    _, forecast_text = nutritionist.predict_rating_class(valid_ingredients)
    print(forecast_text)
    print()
    
    print("II. NUTRITION FACTS")
    nutrition_text = nutritionist.get_nutrition_facts(valid_ingredients)
    print(nutrition_text)
    print()
    
    print("III. TOP-3 SIMILAR RECIPES:")
    similar_recipes_text = nutritionist.find_top_similar_recipes(valid_ingredients)
    print(similar_recipes_text)


if __name__ == '__main__':
    main()
