import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

# Importation de tes modules communs personnalisés
from data_loader import load_raw_signals
from cross_validation import get_subject_folds
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"

# Architecture du CNN 1D
KERNEL_SIZE = 7        
FILTERS_BASE = 64      
DROPOUT_RATE = 0.5     

# Paramètres d'apprentissage
EPOCHS_MAX = 50        
BATCH_SIZE = 64        
PATIENCE = 15          
K_FOLDS = 7            

# Configuration du lissage temporel (LOOK_AROUND = 2 -> fenêtre de 5 trames ≈ 6.4s)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# Fixation des seeds pour la reproductibilité mathématique
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES CHRONOLOGIQUES ET STABLES
# ==============================================================================
print("Chargement et tri chronologique des signaux bruts...")
# Chargement automatique via notre module (Tri mergesort intégré par sujet)
X_train_full, y_train_full, subjects_train_full = load_raw_signals(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_raw_signals(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets: {np.unique(subjects_test)})")

# ==============================================================================
# 2. DÉFINITION DE L'ARCHITECTURE DU CNN 1D (Bottleneck + Double Couche Dense)
# ==============================================================================
def build_cnn_model(input_shape=(128, 9)):
    """
    Architecture CNN 1D 'Bottleneck' (4 blocs convolutifs)
    suivie d'une double barrière Dense anti-overfitting.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # --- PHASE D'EXTRACTION (Le Goulot d'Étranglement) ---
        # Bloc Convolutif 1
        Conv1D(filters=64, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Bloc Convolutif 2 (Réduction des filtres)
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Bloc Convolutif 3
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Bloc Convolutif 4
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        # --- PHASE DE CLASSIFICATION (Robustesse et Anti-Overfitting) ---
        # Première couche dense
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE), # Fixé à 0.5
        
        # Deuxième couche dense
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE), # Fixé à 0.5
        
        # Couche de sortie (6 activités)
        Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==============================================================================
# 3. VALIDATION CROISÉE ÉTANCHE (GROUP K-FOLD PAR SUJET)
# ==============================================================================
print(f"\nLancement du GroupKFold ({K_FOLDS} splits) pour optimiser les époques...")
folds = get_subject_folds(X_train_full, y_train_full, subjects_train_full, n_splits=K_FOLDS)

fold_epochs = []

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n--- Entraînement sur le Fold {fold + 1}/{K_FOLDS} ---")
    
    # Découpage étanche des sujets sans rupture chronologique interne
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]
    
    model_fold = build_cnn_model(input_shape=(128, 9))
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=PATIENCE, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Entraînement avec shuffle=False obligatoire pour conserver la chronologie
    history = model_fold.fit(
        X_tr, y_tr,
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        validation_data=(X_va, y_va),
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Extraction de l'époque optimale pour ce pli
    stopped_epoch = early_stop.stopped_epoch
    if stopped_epoch == 0:
        optimal_fold_epoch = EPOCHS_MAX
    else:
        optimal_fold_epoch = stopped_epoch - PATIENCE + 1
        
    fold_epochs.append(optimal_fold_epoch)
    print(f"Fold {fold + 1} arrêté à l'époque idéale : {optimal_fold_epoch}")

# Calcul du nombre d'époques idéal moyen
optimal_epochs = int(np.mean(fold_epochs))
print(f"\nNombre d'époques optimal déterminé par Cross-Validation : {optimal_epochs}")

# ==============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE FINAL SUR L'ENSEMBLE DU JEU DE TRAIN
# ==============================================================================
print(f"\nEntraînement du modèle final sur les 21 sujets (Époques fixées à {optimal_epochs})...")
final_model = build_cnn_model(input_shape=(128, 9))

final_model.fit(
    X_train_full, y_train_full,
    epochs=optimal_epochs,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Respect de la structure temporelle globale
    verbose=1
)

# ==============================================================================
# 5. ÉVALUATION FINALE ET LISSAGE BIOLOGIQUE SUR LE JEU DE TEST
# ==============================================================================
print("\nÉvaluation sur le jeu de Test (9 sujets vierges de tout entraînement)...")

# Prédiction brute (Sortie probabilités -> Argmax pour obtenir la classe)
y_pred_probs = final_model.predict(X_test)
y_pred_raw = np.argmax(y_pred_probs, axis=1)

# Application du filtre mode glissant issu de notre module post_processing
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 6. ÉVALUATION ET AFFICHAGE BI-MATRICE
# ==============================================================================
# Génère les deux rapports (Macro) et affiche les deux matrices de confusion côte à côte
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)