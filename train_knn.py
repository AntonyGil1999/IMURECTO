import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

# Importation des modules communs personnalisés (Architecture standardisée)
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Grille de recherche pour le KNN
PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Configuration du lissage temporel (Filtre mode de 5 trames ≈ 6.4s)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# ==============================================================================
# 1. CHARGEMENT DES 561 FEATURES CHRONOLOGIQUES
# ==============================================================================
print("Chargement des features expertes (561 variables) avec tri chronologique...")
X_train_full, y_train_full, subjects_train_full = load_features(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_features(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets: {np.unique(subjects_test)})")

# ==============================================================================
# 2. GRID SEARCH CV AVEC ISOLATION STRICTE DES SUJETS
# ==============================================================================
print("\nLancement du GridSearch avec GroupKFold (Recherche des meilleurs paramètres)...")

# Validateur croisé basé sur les groupes (sujets)
gkf = GroupKFold(n_splits=K_FOLDS)

# L'algorithme KNN est généralement plus rapide à entraîner que le SVM
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, # Utilisation de tous les cœurs du processeur
    verbose=2
)

# Application stricte du zéro fuite de données via groups=subjects_train_full
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleurs paramètres trouvés par le GridSearch : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Récupération du modèle optimal
best_knn_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST ET LISSAGE BIOLOGIQUE
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction brute tout en conservant l'ordre temporel des fenêtres
y_pred_raw = best_knn_model.predict(X_test)

# Application du filtre mode glissant pour corriger les erreurs isolées
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION ET AFFICHAGE DES MATRICES COMPARATIVES
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)