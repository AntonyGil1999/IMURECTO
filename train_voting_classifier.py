import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold

# Importation de notre architecture modulaire standardisée
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Configuration des 3 estimateurs de base avec leurs hyperparamètres optimaux
svm_base = SVC(C=100, gamma=0.01, kernel='rbf', probability=True, random_state=42)
lr_base = LogisticRegression(C=10, solver='lbfgs', max_iter=1000, random_state=42)
mlp_base = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42)

# Liste des modèles qui composent le Voting
ESTIMATORS = [
    ('svm', svm_base),
    ('logistic', lr_base),
    ('mlp', mlp_base)
]

# Grille de recherche pour le Voting Classifier
PARAM_GRID = {
    'voting': ['hard', 'soft'] # Teste le vote majoritaire brut vs le vote par probabilités
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
# 2. GRID SEARCH CV : RECHERCHE DU VOTE OPTIMAL (SANS FUITE)
# ==============================================================================
print("\nInitialisation de l'Ensemble et optimisation du mode de vote via GroupKFold...")
print("Entraînement simultané des 3 modèles sur chaque pli...")

# Le Validateur Étanché par sujet
gkf = GroupKFold(n_splits=K_FOLDS)

# Instanciation du méta-classifieur
grid_search = GridSearchCV(
    estimator=VotingClassifier(estimators=ESTIMATORS),
    param_grid=PARAM_GRID,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1, # Utilisation maximale des cœurs du processeur
    verbose=2
)

# Entraînement robuste avec la contrainte absolue des groupes (subjects)
grid_search.fit(X_train_full, y_train_full, groups=subjects_train_full)

print(f"\nMeilleure stratégie de vote trouvée : {grid_search.best_params_}")
print(f"Meilleur score de validation interne : {grid_search.best_score_ * 100:.2f}%")

# Récupération du Voting Classifier optimal réentraîné sur tout le Train Set
best_voting_model = grid_search.best_estimator_

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST ET LISSAGE BIOLOGIQUE
# ==============================================================================
print("\nInférence de l'ensemble sur le jeu de test (9 sujets vierges)...")

# Prédiction brute respectant strictement l'ordre chronologique des fenêtres
y_pred_raw = best_voting_model.predict(X_test)

# Application du lissage comportemental par notre filtre mode
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION MACRO ET AFFICHAGE BI-MATRICE
# ==============================================================================
# Bilan complet et génération des graphiques comparatifs
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)