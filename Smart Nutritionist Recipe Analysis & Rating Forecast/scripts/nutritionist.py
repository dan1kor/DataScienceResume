import sys
import numpy as np
import warnings

from recipes import Recipe

def main():
    warnings.filterwarnings('ignore')

    if len(sys.argv) != 2:
        raise Exception('Неверное кол-во аргументов')

    ingredients = sys.argv[1].split(',')

    recipe = Recipe(ingredients)
    recipe.rate_ingredients()
    recipe.get_nutrition_facts()
    recipe.get_recommendations()
    

if __name__ == '__main__':
    main()
