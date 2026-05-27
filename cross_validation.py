import numpy as np
from sklearn.model_selection import GroupKFold

def get_subject_folds(X, y, subjects, n_splits=7):
    """
    Génère les indices d'entraînement et de validation pour un K-Fold
    basé strictement sur les sujets. 
    Par exemple, sur 21 sujets de train, n_splits=7 créera 7 configurations
    où 18 sujets entraînent et 3 valident.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    # GroupKFold garantit que les mêmes sujets ne se retrouvent jamais
    # à la fois dans le train et le val.
    folds = []
    for train_idx, val_idx in gkf.split(X, y, groups=subjects):
        folds.append((train_idx, val_idx))
        
    return folds

def run_subject_kfold_classic(model, X, y, subjects, n_splits=7):
    """
    Exécute la validation croisée pour les modèles classiques (SVM, RF, XGBoost)
    et retourne les prédictions (out-of-fold) pour tout le dataset.
    """
    folds = get_subject_folds(X, y, subjects, n_splits=n_splits)
    
    oof_predictions = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"--- Entraînement Fold {fold + 1}/{n_splits} ---")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédiction sur le set de validation
        preds = model.predict(X_val)
        oof_predictions[val_idx] = preds
        
    return oof_predictions