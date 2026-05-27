import numpy as np
from sklearn.svm import SVC

# Importation de notre architecture modulaire standardisée
from data_loader import load_features
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET PARAMÈTRES DU KIFF
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"

# 🎯 RENTRE LES PARAMÈTRES DE TA CAPTURE D'ÉCRAN ICI :
CUSTOM_C = 100          # Remplace par ton C
CUSTOM_GAMMA = 0.01     # Remplace par ton gamma
CUSTOM_KERNEL = 'rbf'   # Remplace par ton kernel (ex: 'rbf', 'linear', 'poly')

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
# 2. ENTRAÎNEMENT DIRECT DU MODÈLE (DIRECT FIT)
# ==============================================================================
print(f"\nInitialisation du SVM Custom (C={CUSTOM_C}, gamma={CUSTOM_GAMMA}, kernel='{CUSTOM_KERNEL}')...")
print("Entraînement direct sur les 21 sujets en cours...")

# On configure le modèle avec tes paramètres exacts
custom_svm_model = SVC(
    C=CUSTOM_C, 
    gamma=CUSTOM_GAMMA, 
    kernel=CUSTOM_KERNEL, 
    random_state=42 # Pour avoir exactement le même résultat à chaque lancement
)

# Entraînement classique (Pas de K-Fold car on ne cherche plus d'hyperparamètres)
custom_svm_model.fit(X_train_full, y_train_full)

print("Entraînement terminé avec succès !")

# ==============================================================================
# 3. INFERENCE SUR LE JEU DE TEST ET LISSAGE BIOLOGIQUE
# ==============================================================================
print("\nInférence immédiate sur le jeu de test (9 sujets vierges)...")

# Prédiction respectant la chronologie
y_pred_raw = custom_svm_model.predict(X_test)

# Application du lissage comportemental par notre filtre mode
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 4. ÉVALUATION MACRO ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)