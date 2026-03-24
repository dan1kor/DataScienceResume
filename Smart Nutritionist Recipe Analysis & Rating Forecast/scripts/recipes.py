import numpy as np
import pandas as pd

import joblib

class Recipe:
    def __init__(self, ingredients):
        cols = pd.read_csv('../data/processed/epi_r.csv').columns[1:]
        self.ing_to_idx = {j: i for (i, j) in enumerate(cols)}
        self.idx_to_ing = {j: i for (i, j) in self.ing_to_idx.items()} 
        def to_numpy_array(ingredients):
            ings_arr = np.zeros(len(cols))
            for i in ingredients:
                if self.ing_to_idx.get(i, -1) == -1:
                    raise Exception(f'Незнакомый ингредиент: {i}')

                ings_arr[self.ing_to_idx[i]] = 1
            
            return ings_arr.reshape(1, -1)

        self.ingredients = to_numpy_array(ingredients)

    def rate_ingredients(self):
        model = joblib.load('../models/final_model.pkl')['SVC']

        print('I. OUR FORECAST')
        exp_rating = model.predict(self.ingredients)
        if exp_rating == 0:
            print('Может это и вкусно, но готовить блюдо из этих рецептов - плохая идея')
        elif exp_rating == 1:
            print('В целом неплохо, но могло быть и лучше')
        else:
            print('Шедеврально! Премию мишлен этому шефу!')

    def get_nutrition_facts(self):
        nutritions = pd.read_csv('../data/processed/nutritions.csv')
        print('II. NUTRITION FACTS')

        for i in np.where(self.ingredients)[1]:
            cur_ing = self.idx_to_ing[i]
            print(f'{cur_ing.capitalize()}:')
            cur_data = nutritions[nutritions['Ingredient'] == cur_ing]
            for col in cur_data.columns[1:]:
                print(f'{col} - {cur_data[col].values[0]}% of Daily Value')
            print('\n')

    def get_recommendations(self):
        rec_model = joblib.load('../models/recommendation_model.pkl')
        dishes = pd.read_csv('../data/raw/epi_r.csv')
        print('III. TOP-3 SIMILAR RECIPES:')
        distances, indices = rec_model.kneighbors(self.ingredients)
        for i in indices[0]:
            cur_dish = dishes.iloc[i]
            print(f'— {cur_dish['title']}, rating: {cur_dish['rating']}')