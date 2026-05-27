import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Input, LSTM
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

# Hyperparamètres de l'architecture Hybride
KERNEL_SIZE = 7        
FILTERS_BASE = 64      
LSTM_UNITS = 128       
RECURRENT_DROPOUT = 0.2  # Protection contre le surapprentissage interne de la mémoire
DROPOUT_RATE = 0.5       # Appliqué sur la double barrière dense finale

# Paramètres d'apprentissage
EPOCHS_MAX = 100       # Augmenté à 100 pour laisser le temps au LSTM de converger
BATCH_SIZE = 64        
PATIENCE = 20          # Patience accrue pour stabiliser l'apprentissage récurrent
K_FOLDS = 7            

# Configuration du lissage temporel (Filtre mode de 5 trames ≈ 6.4 secondes)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# Fixation des seeds pour la reproductibilité mathématique
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES CHRONOLOGIQUES ET STABLES
# ==============================================================================
print("Chargement et tri chronologique des signaux bruts (9 canaux)...")
X_train_full, y_train_full, subjects_train_full = load_raw_signals(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_raw_signals(BASE_DIR, dataset_type='test')

print(f"Dataset de Train : {X_train_full.shape} (Sujets unique: {np.unique(subjects_train_full)})")
print(f"Dataset de Test  : {X_test.shape} (Sujets unique: {np.unique(subjects_test)})")

# ==============================================================================
# 2. DÉFINITION DE L'ARCHITECTURE HYBRIDE CNN-LSTM
# ==============================================================================
def build_cnn_lstm_model(input_shape=(128, 9)):
    """
    Modèle hybride : 
    - 4 blocs convolutifs profonds sans Flatten (Bottleneck temporel de 128 -> 8 points).
    - 1 couche LSTM pour capturer les dépendances chronologiques.
    - Double couche dense avec Dropout(0.5) pour la classification robuste.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # --- BLOC 1 : Extraction Convolutive Initiale ---
        Conv1D(filters=FILTERS_BASE, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # --- BLOC 2 : Réduction de dimension spatio-temporelle ---
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # --- BLOC 3 : Condensation des motifs complexes ---
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # --- BLOC 4 : Goulot d'étranglement final (Sortie de taille (None, 8, 32)) ---
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # --- COUCHE RÉCURRENTE (Pas de Flatten avant pour conserver la séquence) ---
        # return_sequences=False car on veut un vecteur de résumé final pour classer la trame
        LSTM(units=LSTM_UNITS, recurrent_dropout=RECURRENT_DROPOUT, return_sequences=False, activation='tanh'),
        
        # --- COUCHES DE CLASSIFICATION (Double barrière anti-overfitting) ---
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        # Sortie sur 6 classes d'activités
        Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==============================================================================
# 3. VALIDATION CROISÉE ÉTANCHE PAR SUJET (DÉTERMINATION DES ÉPOQUES)
# ==============================================================================
print(f"\nLancement du GroupKFold ({K_FOLDS} splits) sur l'architecture CNN-LSTM...")
folds = get_subject_folds(X_train_full, y_train_full, subjects_train_full, n_splits=K_FOLDS)

fold_epochs = []

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n--- Évaluation du Fold {fold + 1}/{K_FOLDS} ---")
    
    # Isolation étanche des sujets
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]
    
    model_fold = build_cnn_lstm_model(input_shape=(128, 9))
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=PATIENCE, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Entraînement séquentiel strict (shuffle=False pour préserver l'ordre)
    model_fold.fit(
        X_tr, y_tr,
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        validation_data=(X_va, y_va),
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )
    
    stopped_epoch = early_stop.stopped_epoch
    optimal_fold_epoch = stopped_epoch - PATIENCE + 1 if stopped_epoch != 0 else EPOCHS_MAX
    fold_epochs.append(optimal_fold_epoch)
    print(f"Fold {fold + 1} convergé à l'époque : {optimal_fold_epoch}")

# Moyennage mathématique des époques de convergence
optimal_epochs = int(np.mean(fold_epochs))
print(f"\nNombre d'époques idéales calculé : {optimal_epochs}")

# ==============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE HYBRIDE FINAL
# ==============================================================================
print(f"\nEntraînement final du CNN-LSTM sur les 21 sujets (Fixé à {optimal_epochs} époques)...")
final_model = build_cnn_lstm_model(input_shape=(128, 9))

final_model.fit(
    X_train_full, y_train_full,
    epochs=optimal_epochs,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)

# ==============================================================================
# 5. INFERENCE ET LISSAGE TEMPÈRE
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets exclus du processus)...")
y_pred_probs = final_model.predict(X_test)
y_pred_raw = np.argmax(y_pred_probs, axis=1)

# Application du lissage comportemental par vote majoritaire local
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 6. BILAN ET AFFICHAGE DES RÉSULTATS COMPARATIFS
# ==============================================================================
# Génère les rapports macro et affiche les matrices de confusion brute vs lissée
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)