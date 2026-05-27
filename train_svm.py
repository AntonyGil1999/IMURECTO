import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold

# Importation de tes modules communs personnalisés
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Grille de recherche pour le SVM (à ajuster selon ta puissance de calcul)
# Le kernel 'rbf' est le standard pour ce dataset.
PARAM_GRID = {
    'C': [10, 100, 1000],
    'gamma': [0.01, 0.001, 'scale'],
    'kernel': ['rbf']
}

# Configuration du lissage temporel
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# ==============================================================================
# 1. CHARGEMENT DES 561 FEATURES CHRONOLOGIQUES
# ==============================================================================
print("Chargement des features expertes (561 variables) avec tri chronologique...")
# On utilise load_features() et non load_raw_signals() pour le SVM
X_train_full, y_train_full, subjects_train_full = load_features(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_features(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets: {np.unique(subjects_test)})")

# ==============================================================================
# 2. GRID SEARCH CV AVEC ISOLATION STRICTE DES SUJETS
# ==============================================================================
print("\nLancement du GridSearch avec GroupKFold (Recherche des meilleurs paramètres)...")
print("Cela peut prendre quelques minutes selon ton processeur (n_jobs=-1 utilise tous les cœurs).")

# Initialisation du validateur croisé basé sur les sujets
gkf = GroupKFold(n_splits=K_FOLDS)

# Initialisation du GridSearch
# refit=True entraîne automatiquement le meilleur modèle trouvé sur tout le X_train_full à la fin
grid_search = GridSearchCV(
    estimator=SVC(probability=False), # False accélère énormément le calcul, on veut juste la classe
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, 
    verbose=2
)

# L'argument groups=subjects_train_full est LA clé qui garantit le zéro fuite de données
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleurs paramètres trouvés par le GridSearch : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Extraction du modèle optimal entraîné
best_svm_model = grid_search.best_estimator_

# ==============================================================================
# 3. ÉVALUATION FINALE ET LISSAGE BIOLOGIQUE SUR LE JEU DE TEST
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction brute avec le meilleur modèle
# L'ordre temporel est vital ici pour que le lissage fonctionne ensuite
y_pred_raw = best_svm_model.predict(X_test)

# Application du filtre mode glissant (correction des erreurs Assis/Debout isolées)
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)