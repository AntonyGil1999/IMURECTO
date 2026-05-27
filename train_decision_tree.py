import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

# Grille de recherche rigoureuse pour empêcher l'arbre de faire du "par cœur"
PARAM_GRID = {
    'criterion': ['gini', 'entropy'],      # Façon de calculer la pureté d'une séparation
    'max_depth': [10, 15, 20, None],       # Limite la profondeur (None = profondeur infinie = danger d'overfitting)
    'min_samples_split': [2, 10, 20],      # Nombre minimum de fenêtres pour créer une nouvelle branche
    'min_samples_leaf': [1, 5, 10]         # Nombre minimum de fenêtres requises pour valider une feuille finale
}

# Configuration du lissage temporel (Fenêtre de 5 trames ≈ 6.4 secondes)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# ==============================================================================
# 1. CHARGEMENT CHRONOLOGIQUE DES DONNÉES
# ==============================================================================
print("Chargement des features expertes (561 variables) avec tri chronologique...")
X_train_full, y_train_full, subjects_train_full = load_features(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_features(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets: {np.unique(subjects_test)})")

# ==============================================================================
# 2. GRID SEARCH CV : RECHERCHE DE L'ARBRE OPTIMAL (SANS FUITE)
# ==============================================================================
print("\nÉlagage et optimisation de l'Arbre via GroupKFold...")

# Le Validateur Étanché
gkf = GroupKFold(n_splits=K_FOLDS)

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42), # random_state pour figer le choix si égalité mathématique
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Entraînement avec la contrainte absolue des groupes (subjects)
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleure architecture d'Arbre trouvée : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Récupération du modèle optimal
best_tree_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST (SUJETS INCONNUS)
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")

# Prédiction respectant la chronologie
y_pred_raw = best_tree_model.predict(X_test)

# Application du lissage biologique (VITAL pour les Arbres de décision)
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION MACRO ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)