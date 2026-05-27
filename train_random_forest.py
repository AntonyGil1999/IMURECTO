import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Grille de recherche pour la Forêt Aléatoire
PARAM_GRID = {
    'n_estimators': [50, 100, 200],         # Nombre d'arbres dans la forêt
    'max_depth': [10, 20, None],            # Profondeur des arbres
    'min_samples_leaf': [1, 5, 10]          # Contrainte anti-bruit : minimum d'échantillons par décision finale
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
# 2. GRID SEARCH CV : RECHERCHE DE LA FORÊT OPTIMALE (ZÉRO FUITE)
# ==============================================================================
print("\nCréation et optimisation de la Forêt Aléatoire via GroupKFold...")
print("Le calcul peut prendre quelques minutes (plusieurs centaines d'arbres à entraîner)...")

# Le Validateur Étanché
gkf = GroupKFold(n_splits=K_FOLDS)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1), # n_jobs=-1 utilise tous les cœurs pour chaque forêt
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, # Utilise aussi tous les cœurs pour paralléliser le GridSearch
    verbose=2
)

# Entraînement avec la contrainte absolue des groupes (subjects)
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleurs hyperparamètres de la Forêt : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Récupération de l'ensemble optimal
best_rf_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST (SUJETS INCONNUS)
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction respectant la chronologie
y_pred_raw = best_rf_model.predict(X_test)

# Application du lissage biologique par le filtre mode
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION MACRO ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)