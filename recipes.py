# recipes.py
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecipeNutritionist:
    
    
    def __init__(self, model_path='best_recipe_model.pkl', 
                 nutrition_path='ingredients_nutrition_dv.csv', 
                 urls_path='recipes_with_urls.csv',
                 raw_dataset_path='../materials/epi_r.csv'):
        
        
        artifacts = joblib.load(model_path)
        self.pipeline = artifacts['model']
        self.feature_names = artifacts['features']
        
    
        self.df_nutrition = pd.read_csv(nutrition_path)
        
        
        self.df_urls = pd.read_csv(urls_path)
        if 'recipe_id' in self.df_urls.columns:
            self.df_urls = self.df_urls.set_index('recipe_id')
            
        
        df_raw = pd.read_csv(raw_dataset_path)
        
        rename_dict = {'milk/cream': 'milk', 'jam or jelly': 'jam'}
        df_raw = df_raw.rename(columns=rename_dict)
        self.X_matrix = df_raw.loc[self.df_urls.index, self.feature_names].copy().fillna(0)
        self.y_ratings = df_raw.loc[self.df_urls.index, 'rating'].values

    def validate_ingredients_list(self, ingredients_list):
        
        missing = []
        valid = []
        
        for ing in ingredients_list:
            if ing in self.feature_names:
                valid.append(ing)
            else:
                missing.append(ing)
                
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"the following ingredients are missing in our database: {missing_str}")
            
        return valid

    def predict_rating_class(self, valid_ingredients):
        
        input_vector = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        for ing in valid_ingredients:
            input_vector.loc[0, ing] = 1
            

        pred_class = self.pipeline.predict(input_vector)[0]
        
        if pred_class == 0:
            return "bad", "You might find it tasty, but in our opinion, it is a bad idea to have a dish with that list of ingredients."
        elif pred_class == 1:
            return "so-so", "In our opinion, it is a so-so idea. It could be acceptable, but not perfect."
        else:
            return "great", "Wow! In our opinion, this is an excellent and great idea to have a dish with that list of ingredients!"
        

    def get_nutrition_facts(self, valid_ingredients):
        
        nutrition_output = []
       
        pretty_names = {
            'protein_DV': 'Protein', 'fat_DV': 'Total Fat', 'carbs_DV': 'Total Carbohydrate',
            'fiber_DV': 'Fiber', 'calcium_DV': 'Calcium', 'iron_DV': 'Iron',
            'potassium_DV': 'Potassium', 'sodium_DV': 'Sodium', 'vit_A_DV': 'Vitamin A',
            'vit_C_DV': 'Vitamin C', 'vit_D_DV': 'Vitamin D', 'vit_E_DV': 'Vitamin E',
            'vit_B12_DV': 'Vitamin B12', 'saturated_fat_DV': 'Saturated Fat'
        }
        
        for ing in valid_ingredients:
            ing_row = self.df_nutrition[self.df_nutrition['ingredient'] == ing]
            if not ing_row.empty:
                ing_output = [f"{ing.capitalize()}"]
                for col_name, display_name in pretty_names.items():
                    if col_name in ing_row.columns:
                        dv_val = ing_row[col_name].values[0]
                        ing_output.append(f"  {display_name} - {int(round(dv_val))}% of Daily Value")
                nutrition_output.append("\n".join(ing_output))
                
        return "\n\n".join(nutrition_output)

    def find_top_similar_recipes(self, valid_ingredients):
        
        user_vector = np.zeros((1, len(self.feature_names)))
        for ing in valid_ingredients:
            idx = self.feature_names.index(ing)
            user_vector[0, idx] = 1
            
        similarities = cosine_similarity(user_vector, self.X_matrix.values)
        
        if np.max(similarities) < 0.05:
            return "похожих рецептов нет"
            
        
        top_indices = np.argsort(similarities[0])[::-1][:3]
        
        recipes_output = []
        for idx in top_indices:
            recipe_info = self.df_urls.loc[self.X_matrix.index[idx]]
            title = recipe_info['title']
            
            
            slug = title.lower().strip()
            slug = re.sub(r'[^a-z0-9\s-]', '', slug)
            slug = re.sub(r'\s+', '-', slug)
            slug = re.sub(r'-+', '-', slug).strip('-')
            
            
            correct_url = f"https://epicurious.com/{slug}"
            rating = self.y_ratings[idx]
            
            recipes_output.append(f"- {title}, rating: {rating:.1f}, URL: {correct_url}")
            
        return "\n".join(recipes_output)



class RecipeMenuGenerator:
    """Бонусный класс для генерации оптимального дневного меню (Завтрак, Обед, Ужин)"""
    
    def __init__(self, feature_names, df_nutrition, urls_path='recipes_with_urls.csv', raw_dataset_path='../materials/epi_r.csv'):
        self.feature_names = feature_names
        self.df_nutrition = df_nutrition.set_index('ingredient')
        
        
        self.df_urls = pd.read_csv(urls_path).set_index('recipe_id')
        df_raw = pd.read_csv(raw_dataset_path)
        
        
        rename_dict = {'milk/cream': 'milk', 'jam or jelly': 'jam'}
        df_raw = df_raw.rename(columns=rename_dict)
        
        
        self.X_matrix = df_raw.loc[self.df_urls.index, self.feature_names].copy().fillna(0)
        self.ratings = df_raw.loc[self.df_urls.index, 'rating'].values
        self.titles = self.df_urls['title'].values
        self.urls = self.df_urls['url'].values
        
        
        self.is_breakfast = df_raw.loc[self.df_urls.index, 'breakfast'].values if 'breakfast' in df_raw.columns else np.zeros(len(df_raw))
        self.is_lunch = df_raw.loc[self.df_urls.index, 'lunch'].values if 'lunch' in df_raw.columns else np.zeros(len(df_raw))
        self.is_dinner = df_raw.loc[self.df_urls.index, 'dinner'].values if 'dinner' in df_raw.columns else np.zeros(len(df_raw))
        
        
        if self.is_lunch.sum() == 0: self.is_lunch = df_raw.loc[self.df_urls.index, 'snack'].values if 'snack' in df_raw.columns else np.ones(len(df_raw))
        if self.is_dinner.sum() == 0: self.is_dinner = df_raw.loc[self.df_urls.index, 'dinner'].values if 'dinner' in df_raw.columns else np.ones(len(df_raw))

        
        self.nutrient_cols = ['protein_DV', 'fat_DV', 'carbs_DV', 'sodium_DV']
        self.pretty_nutrients = {'protein_DV': 'protein', 'fat_DV': 'fat', 'carbs_DV': 'carbohydrate', 'sodium_DV': 'sodium'}

        
        self.recipe_nutrients = self._precompute_recipe_nutrition()

    def _precompute_recipe_nutrition(self):
        
        ing_nutr_matrix = pd.DataFrame(0.0, index=self.feature_names, columns=self.nutrient_cols)
        for ing in self.feature_names:
            if ing in self.df_nutrition.index:
                ing_nutr_matrix.loc[ing] = self.df_nutrition.loc[ing, self.nutrient_cols].values
                
        
        calculated = np.dot(self.X_matrix.values, ing_nutr_matrix.values) / 3.0
        return np.clip(calculated, 0, 100) # Ограничиваем сверху 100% на один рецепт

    def generate_daily_menu(self):
        
        idx_b = np.where(self.is_breakfast == 1)[0]
        idx_l = np.where(self.is_lunch == 1)[0]
        idx_d = np.where(self.is_dinner == 1)[0]
        
        
        if len(idx_b) == 0: idx_b = np.arange(len(self.titles))
        if len(idx_l) == 0: idx_l = np.arange(len(self.titles))
        if len(idx_d) == 0: idx_d = np.arange(len(self.titles))

        best_score = -1
        best_menu = None
        
        
        # np.random.seed(42)
        for _ in range(1000):
            b = np.random.choice(idx_b)
            l = np.random.choice(idx_l)
            d = np.random.choice(idx_d)
            
            
            total_nutrients = self.recipe_nutrients[b] + self.recipe_nutrients[l] + self.recipe_nutrients[d]
            
            
            if np.all(total_nutrients <= 100.0) and np.mean(total_nutrients) >= 35.0:
                
                current_score = self.ratings[b] + self.ratings[l] + self.ratings[d]
                
                if current_score > best_score:
                    best_score = current_score
                    best_menu = (b, l, d)
                    
        
        if best_menu == None:
            b = idx_b[np.argmax(self.ratings[idx_b])]
            l = idx_l[np.argmax(self.ratings[idx_l])]
            d = idx_d[np.argmax(self.ratings[idx_d])]
            best_menu = (b, l, d)

        
        stages = ['BREAKFAST', 'LUNCH', 'DINNER']
        output = []
        
        for stage, r_idx in zip(stages, best_menu):
            
            row_ingredients = self.X_matrix.iloc[r_idx]
            active_ings = [self.feature_names[i] for i, val in enumerate(row_ingredients) if val == 1]
            ing_lines = "\n".join([f"- {ing}" for ing in active_ings])
            
            
            nutr_lines = []
            for n_idx, col in enumerate(self.nutrient_cols):
                val = int(round(self.recipe_nutrients[r_idx][n_idx]))
                nutr_lines.append(f"  - {self.pretty_nutrients[col]}: {val}%")
            nutr_str = "\n".join(nutr_lines)
            
            
            clean_slug = self.titles[r_idx].lower().strip()
            clean_slug = re.sub(r'[^a-z0-9\s-]', '', clean_slug)
            clean_slug = re.sub(r'\s+', '-', clean_slug)
            clean_slug = re.sub(r'-+', '-', clean_slug).strip('-')
            correct_url = f"https://epicurious.com/{clean_slug}"

            
            stage_block = (
                f"{stage}\n"
                f"---------------------\n"
                f"{self.titles[r_idx]} (rating: {self.ratings[r_idx]:.3f})\n"
                f"Ingredients:\n{ing_lines}\n"
                f"Nutrients:\n{nutr_str}\n"
                f"URL: {correct_url}"
            )
            output.append(stage_block)
            
        return "\n\n".join(output)


if __name__ == '__main__':
    print("--- ТЕСТИРОВАНИЕ МОДУЛЯ RECIPES.PY ---")
    try:
        
        print("\n[Тест 1] Инициализация базового класса RecipeNutritionist...")
        tester = RecipeNutritionist()
        print("-> Базовые файлы и артефакты моделей успешно обнаружены.")
        
        
        test_ingredients = ["milk", "honey", "jam"]
        valid = tester.validate_ingredients_list(test_ingredients)
        print(f"-> Тест валидации пройден. Распознано ингредиентов: {len(valid)}")
        
        
        print("\n[Тест 2] Инициализация бонусного класса RecipeMenuGenerator...")
        menu_tester = RecipeMenuGenerator(
            feature_names=tester.feature_names, 
            df_nutrition=tester.df_nutrition
        )
        print("-> Предрасчет матрицы КБЖУ для рецептов выполнен успешно.")
        
        
        print("-> Проверка работы алгоритма подбора меню...")
        sample_menu = menu_tester.generate_daily_menu()
        print("-> Алгоритм успешно сгенерировал тестовое меню.")
        
        print("СТАТУС МОДУЛЯ: ВСЕ КЛАССЫ ИСПРАВНЫ И ГОТОВЫ К РАБОТЕ.")
        
        
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] При самотестировании модуля: {e}")

