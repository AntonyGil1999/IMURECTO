import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Input, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Importation des modules communs de l'architecture
from data_loader import load_raw_signals
from cross_validation import get_subject_folds
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"

# Hyperparamètres de l'architecture CNN + Bi-LSTM
KERNEL_SIZE = 7        
FILTERS_BASE = 64      
LSTM_UNITS = 128       
RECURRENT_DROPOUT = 0.1  # Fixé à 0.1 selon tes spécifications pour le Bi-LSTM
DROPOUT_RATE = 0.5       

# Paramètres d'apprentissage
EPOCHS_MAX = 100       
BATCH_SIZE = 64        
PATIENCE = 20          
K_FOLDS = 7            

# Configuration du lissage (Filtre mode de 5 trames)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# Fixation des seeds pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. CHARGEMENT CHRONOLOGIQUE DES SIGNAUX
# ==============================================================================
print("Chargement des signaux bruts (9 canaux) avec tri chronologique strict...")
X_train_full, y_train_full, subjects_train_full = load_raw_signals(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_raw_signals(BASE_DIR, dataset_type='test')

# ==============================================================================
# 2. DÉFINITION DE L'ARCHITECTURE HYBRIDE BI-DIRECTIONNELLE
# ==============================================================================
def build_cnn_bilstm_model(input_shape=(128, 9)):
    """
    Architecture CNN 'Bottleneck' suivie d'un LSTM Bidirectionnel.
    Analyse le mouvement en combinant la causalité (passé->futur) 
    et l'anticipation (futur->passé).
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # --- BLOC CNN : Extraction et Réduction ---
        Conv1D(filters=FILTERS_BASE, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # --- COUCHE RÉCURRENTE BI-DIRECTIONNELLE ---
        # Le Wrapper Bidirectional dédouble le LSTM : un passe à l'endroit, un passe à l'envers
        Bidirectional(
            LSTM(units=LSTM_UNITS, recurrent_dropout=RECURRENT_DROPOUT, return_sequences=False, activation='tanh')
        ),
        
        # --- CLASSIFICATION ROBUSTE ---
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==============================================================================
# 3. VALIDATION CROISÉE ÉTANCHE (GROUP K-FOLD)
# ==============================================================================
print(f"\nRecherche des époques optimales via GroupKFold ({K_FOLDS} splits)...")
folds = get_subject_folds(X_train_full, y_train_full, subjects_train_full, n_splits=K_FOLDS)

fold_epochs = []

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n--- Évaluation du Fold {fold + 1}/{K_FOLDS} ---")
    
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]
    
    model_fold = build_cnn_bilstm_model(input_shape=(128, 9))
    
    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=PATIENCE, 
        restore_best_weights=True,
        verbose=1
    )
    
    model_fold.fit(
        X_tr, y_tr,
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        validation_data=(X_va, y_va),
        shuffle=False,  # Vital pour le LSTM
        callbacks=[early_stop],
        verbose=0
    )
    
    stopped_epoch = early_stop.stopped_epoch
    optimal_fold_epoch = stopped_epoch - PATIENCE + 1 if stopped_epoch != 0 else EPOCHS_MAX
    fold_epochs.append(optimal_fold_epoch)
    print(f"Fold {fold + 1} convergé à l'époque : {optimal_fold_epoch}")

optimal_epochs = int(np.mean(fold_epochs))
print(f"\nNombre d'époques idéal (Moyenne) : {optimal_epochs}")

# ==============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE FINAL
# ==============================================================================
print(f"\nEntraînement du CNN Bi-LSTM sur les 21 sujets (Fixé à {optimal_epochs} époques)...")
final_model = build_cnn_bilstm_model(input_shape=(128, 9))

final_model.fit(
    X_train_full, y_train_full,
    epochs=optimal_epochs,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)

# ==============================================================================
# 5. INFERENCE ET ÉVALUATION
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")
y_pred_probs = final_model.predict(X_test)
y_pred_raw = np.argmax(y_pred_probs, axis=1)

# Lissage temporel
y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# Rapport et affichage des matrices
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)