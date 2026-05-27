import numpy as np
from scipy import stats

def apply_mode_filter(predictions, window_size=5):
    """
    Applique un filtre mode glissant sur un vecteur de prédictions.
    Utilise une taille de fenêtre impaire (ex: 5 pour lisser ~7.68 secondes).
    Idéal pour des données catégorielles (contrairement au filtre médian pur).
    """
    if window_size % 2 == 0:
        raise ValueError("La taille de la fenêtre (window_size) doit être impaire.")
        
    pad_size = window_size // 2
    
    # Rembourrage (padding) aux extrémités pour garder la même taille de vecteur
    padded_preds = np.pad(predictions, (pad_size, pad_size), mode='edge')
    smoothed_preds = np.zeros_like(predictions)
    
    for i in range(len(predictions)):
        # Extrait la fenêtre locale
        window = padded_preds[i : i + window_size]
        # Trouve la classe majoritaire (mode) dans cette fenêtre
        mode_val, _ = stats.mode(window, keepdims=False)
        smoothed_preds[i] = mode_val
        
    return smoothed_preds