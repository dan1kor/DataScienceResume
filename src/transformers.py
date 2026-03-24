import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class HeroesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, player_df=None):
        self.hero_to_idx = {}
        self.n_heroes_ = 0
        self.player_df = player_df

    def fit(self, X, y=None):
        if self.player_df is None:
            self.player_df = X.copy()
        unique_heroes = sorted(self.player_df['hero_id'].unique())
        self.hero_to_idx = {hero_id: i for i, hero_id in enumerate(unique_heroes)}
        self.n = len(unique_heroes)
        return self

    def transform(self, X):
        mask = self.player_df['match_id'].isin(X['match_id'])
        current_players = self.player_df[mask].copy()
        
        match_to_idx_map = pd.Series(np.arange(len(X)), index=X['match_id'])
        
        current_players['row_idx'] = current_players['match_id'].map(match_to_idx_map)
        
        current_players = current_players[current_players['hero_id'].isin(self.hero_to_idx)]
        current_players['col_idx'] = current_players['hero_id'].map(self.hero_to_idx)
        
        current_players['val'] = np.where(current_players['player_slot'] < 128, 1, -1)
        
        heroes_sparse = coo_matrix(
            (current_players['val'], (current_players['row_idx'], current_players['col_idx'])),
            shape=(len(X), self.n)
        )
    
        return heroes_sparse.tocsr()
    
class AdvantageEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dota_adv=None):
        self.dota_adv = dota_adv
        self.dota_adv_indexed = None
        self.n_features = 8

    def fit(self, X, y=None):
        if self.dota_adv is not None:
            self.dota_adv_indexed = self.dota_adv.set_index('match_id')
        return self

    def transform(self, X):
        match_ids = pd.Series(X['match_id'])
        result = np.zeros((len(match_ids), self.n_features))

        if self.dota_adv_indexed is not None:
            valid_mask = match_ids.isin(self.dota_adv_indexed.index)
            
            if valid_mask.any():
                ids_to_find = match_ids[valid_mask]
                result[valid_mask] = self.dota_adv_indexed.loc[ids_to_find].values
                
        return result

class TrendTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, gold_data: np.array, exp_data: np.array, id_order):
        self.gold_data = gold_data
        self.exp_data = exp_data
        self.id_order = id_order
        self.match_to_idx = None

    def fit(self, X=None, y=None):
        self.match_to_idx = pd.Series(np.arange(len(self.id_order)), index=self.id_order)
        return self

    def transform(self, X):
        match_ids = X['match_id'].values
        idxs = self.match_to_idx.reindex(match_ids).values

        valid_mask = ~np.isnan(idxs)
        safe_idxs = np.nan_to_num(idxs, nan=0).astype(int)

        current_gold = self.gold_data[safe_idxs]
        current_exp = self.exp_data[safe_idxs]

        current_gold[~valid_mask] = 0
        current_exp[~valid_mask] = 0

        n = current_gold.shape[1]
        x = np.arange(n, dtype=np.float64)

        def get_params(Y):
            x_mean = np.mean(x)
            y_mean = np.mean(Y, axis=1, keepdims=True)
            
            numerator = np.sum((x - x_mean) * (Y - y_mean), axis=1)
            denominator = np.sum((x - x_mean)**2)
            
            slope = numerator / denominator
            intercept = y_mean.flatten() - slope * x_mean
            
            y_pred = slope.reshape(-1, 1) * x + intercept.reshape(-1, 1)
            ss_res = np.sum((Y - y_pred)**2, axis=1)
            ss_tot = np.sum((Y - y_mean)**2, axis=1)
            r2 = np.where(ss_tot > 1e-10, 1 - (ss_res / ss_tot), 0.0)
            
            return slope, intercept, r2

        g_slope, g_int, g_r2 = get_params(current_gold)
        e_slope, e_int, e_r2 = get_params(current_exp)
        
        result = np.column_stack([g_slope, g_int, g_r2, e_slope, e_int, e_r2])
        result = np.nan_to_num(result)
        return result