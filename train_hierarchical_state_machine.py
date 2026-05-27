import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Importation de notre architecture modulaire
from data_loader import load_features
from cross_validation import get_subject_folds
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET FILTRAGE DES VARIABLES (52 FEATURES UNIQUES)
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"

LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

DYNAMIC_CLASSES = [0, 1, 2] # Marche, Monter, Descendre
STATIC_CLASSES = [3, 4, 5]  # Assis, Debout, Allongé

# Notre Top 10 par activité (60 caractéristiques brutes)
TOP_FEATURES = [
    199, 38, 118, 160, 105, 200, 451, 103, 119, 157, # MARCHE
    56, 460, 63, 459, 58, 23, 135, 193, 191, 560,    # MONTER
    90, 93, 25, 10, 29, 505, 198, 503, 508, 377,     # DESCENDRE
    187, 42, 51, 188, 54, 184, 105, 146, 247, 186,   # ASSIS
    183, 57, 560, 446, 157, 122, 106, 110, 288, 148, # DEBOUT
    559, 58, 59, 42, 51, 54, 159, 456, 38, 458       # ALLONGÉ
]

# Suppression des 8 doublons et passage en indexation Python (0-based) -> 52 caractéristiques uniques
selected_indices = np.unique(TOP_FEATURES) - 1

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES (CHRONOLOGIE ET SUJETS PRÉSERVÉS)
# ==============================================================================
print("1. Chargement des données d'origine...")
X_train_full, y_train_full, subjects_train = load_features(BASE_DIR, dataset_type='train')
X_test_full, y_test, subjects_test = load_features(BASE_DIR, dataset_type='test')

print(f"-> Application du masque de sélection : réduction de 561 à {len(selected_indices)} variables uniques.")
X_train_light = X_train_full[:, selected_indices]
X_test_light = X_test_full[:, selected_indices]

# Génération des plis de validation croisée par sujet (7 Folds étanches)
cv_folds = get_subject_folds(X_train_light, y_train_full, subjects_train, n_splits=7)

# ==============================================================================
# 2. PRÉPARATION DES LABELS HIÉRARCHIQUES
# ==============================================================================
y_train_binary = np.isin(y_train_full, DYNAMIC_CLASSES).astype(int)
y_test_binary = np.isin(y_test, DYNAMIC_CLASSES).astype(int)

# Masques d'isolation pour l'entraînement des Experts
mask_train_dyn = np.isin(y_train_full, DYNAMIC_CLASSES)
mask_train_stat = np.isin(y_train_full, STATIC_CLASSES)

X_train_dynamic = X_train_light[mask_train_dyn]
y_train_dynamic = y_train_full[mask_train_dyn]
subjects_dynamic = subjects_train[mask_train_dyn]

X_train_static = X_train_light[mask_train_stat]
y_train_static = y_train_full[mask_train_stat]
subjects_static = subjects_train[mask_train_stat]

# Generate specialized folds for the experts to ensure clean GridSearch
cv_folds_dyn = get_subject_folds(X_train_dynamic, y_train_dynamic, subjects_dynamic, n_splits=7)
cv_folds_stat = get_subject_folds(X_train_static, y_train_static, subjects_static, n_splits=7)

# ==============================================================================
# 3. OPTIMISATION ET ENTRAÎNEMENT PAR RECHERCHE SUR GRILLE (GRIDSEARCHCV)
# ==============================================================================
print("\n2. Optimisation des cerveaux de l'architecture...")

print("[Phase 1/3] GridSearch du Gatekeeper (Statique vs Dynamique)...")
param_grid_gate = {'C': [0.1, 1.0, 10.0]}
gatekeeper_gs = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid_gate, cv=cv_folds, n_jobs=-1)
gatekeeper_gs.fit(X_train_light, y_train_binary)
gatekeeper = gatekeeper_gs.best_estimator_
print(f"-> Meilleur paramètre Gatekeeper : {gatekeeper_gs.best_params_}")

print("[Phase 2/3] GridSearch de l'Expert Dynamique (SVM)...")
param_grid_dyn = {'C': [10, 100], 'gamma': [0.01, 'scale']}
expert_dyn_gs = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid_dyn, cv=cv_folds_dyn, n_jobs=-1)
expert_dyn_gs.fit(X_train_dynamic, y_train_dynamic)
expert_dynamic = expert_dyn_gs.best_estimator_
print(f"-> Meilleur paramètre Expert Dynamique : {expert_dyn_gs.best_params_}")

print("[Phase 3/3] GridSearch de l'Expert Statique (Régression Logistique)...")
param_grid_stat = {'C': [1, 10, 100]}
expert_stat_gs = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42), param_grid_stat, cv=cv_folds_stat, n_jobs=-1)
expert_stat_gs.fit(X_train_static, y_train_static)
expert_static = expert_stat_gs.best_params_
expert_static = expert_stat_gs.best_estimator_
print(f"-> Meilleur paramètre Expert Statique : {expert_stat_gs.best_params_}")

# ==============================================================================
# 4. INFERENCE HIÉRARCHIQUE SUR LE TEST SET
# ==============================================================================
print("\n3. Inférence hiérarchique en cours sur le jeu de test...")
y_pred_raw = np.zeros_like(y_test)

# Étape 1 : Le Gatekeeper décide pour chaque fenêtre (0 = statique, 1 = dynamique)
predictions_binary = gatekeeper.predict(X_test_light)

# Étape 2 : Routage étanche vers les experts concernés
for i in range(len(X_test_light)):
    sample = X_test_light[i].reshape(1, -1)
    
    if predictions_binary[i] == 1:
        y_pred_raw[i] = expert_dynamic.predict(sample)[0]
    else:
        y_pred_raw[i] = expert_static.predict(sample)[0]

# ==============================================================================
# 5. LISSAGE BIOLOGIQUE ET ÉVALUATION FINALE (DOUBLE BILAN)
# ==============================================================================
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

print("\n" + "="*60)
print(" 📊 BILAN DES PERFORMANCES DU PIPELINE UNIQUE HISTORIQUE 📊")
print("="*60)
gatekeeper_acc = accuracy_score(y_test_binary, predictions_binary)
print(f"Précision de l'aiguillage du Gatekeeper : {gatekeeper_acc * 100:.2f}%\n")

# Génération automatique du double affichage de confusion et du rapport macro
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)