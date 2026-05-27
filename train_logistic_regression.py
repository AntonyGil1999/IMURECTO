import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

# Importation de notre architecture modulaire
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Grille de recherche pour la Régression Logistique
PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'newton-cg'],
    'max_iter': [1000] # Crucial pour éviter le ConvergenceWarning avec 561 features
}

# Configuration du lissage temporel (Fenêtre de 5 trames)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# ==============================================================================
# 1. CHARGEMENT DES 561 FEATURES CHRONOLOGIQUES
# ==============================================================================
print("Chargement des features expertes (561 variables) avec tri chronologique strict...")
X_train_full, y_train_full, subjects_train_full = load_features(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_features(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets: {np.unique(subjects_test)})")

# ==============================================================================
# 2. GRID SEARCH CV AVEC ISOLATION STRICTE DES SUJETS
# ==============================================================================
print("\nLancement du GridSearch avec GroupKFold (Optimisation du modèle linéaire)...")

# Le "Mur" étanche de validation
gkf = GroupKFold(n_splits=K_FOLDS)

# Configuration de la recherche
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, # Accélération multicoeur
    verbose=2
)

# Application stricte de la contrainte 'groups' pour éviter les fuites de données
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleurs paramètres trouvés : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Extraction du modèle linéaire optimal
best_lr_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST ET LISSAGE BIOLOGIQUE
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction respectant la chronologie
y_pred_raw = best_lr_model.predict(X_test)

# Application du filtre mode pour stabiliser les transitions et hésitations
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION ET AFFICHAGE BI-MATRICE
# ==============================================================================
# Bilan complet avec macro-moyenne
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)