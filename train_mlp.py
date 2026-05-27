import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

# Importation de l'architecture modulaire du Projet IMU
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Grille de recherche pour le Multi-Layer Perceptron
PARAM_GRID = {
    # Architectures (Ex: (64,) = 1 couche de 64. (128, 64) = 2 couches)
    'hidden_layer_sizes': [(64,), (128, 64), (64, 32)], 
    'alpha': [0.0001, 0.01, 0.1],          # Force de la régularisation L2 (Anti-overfitting)
    'activation': ['relu'],                # Fonction d'activation standard
    'max_iter': [500]                      # Nombre d'époques maximum pour garantir la convergence
}

# Configuration du lissage temporel (Fenêtre de 5 trames ≈ 6.4 secondes)
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
# 2. GRID SEARCH CV : RECHERCHE DU MLP OPTIMAL (ZÉRO FUITE)
# ==============================================================================
print("\nCréation et optimisation du réseau MLP via GroupKFold...")
print("Entraînement des réseaux de neurones en cours...")

# Le Validateur Étanché
gkf = GroupKFold(n_splits=K_FOLDS)

grid_search = GridSearchCV(
    estimator=MLPClassifier(random_state=42, early_stopping=True), # early_stopping interne pour gagner du temps
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, # Parallélisation maximale sur ton processeur
    verbose=1
)

# Entraînement avec la contrainte absolue des groupes (subjects)
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleure architecture MLP trouvée : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Récupération de l'ensemble optimal
best_mlp_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST ET LISSAGE
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction respectant la chronologie
y_pred_raw = best_mlp_model.predict(X_test)

# Application du lissage biologique (essentiel pour stabiliser les hésitations du réseau)
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION MACRO ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)