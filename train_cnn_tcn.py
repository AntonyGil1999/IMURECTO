import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Importation de notre architecture modulaire
from data_loader import load_raw_signals
from cross_validation import get_subject_folds
from post_processing import apply_mode_filter
from evaluation import evaluate_full_pipeline

# ==============================================================================
# 0. CONFIGURATION ET HYPERPARAMÈTRES
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
K_FOLDS = 7

# Hyperparamètres de l'architecture CNN + TCN
KERNEL_SIZE = 7        
FILTERS_BASE = 64      
TCN_FILTERS = 64       # Filtres pour l'analyse temporelle dilatée
DROPOUT_RATE = 0.5     

# Paramètres d'apprentissage
EPOCHS_MAX = 100       
BATCH_SIZE = 64        
PATIENCE = 20          

# Configuration du lissage temporel (Fenêtre de 5 trames ≈ 6.4 secondes)
LOOK_AROUND = 2 
FILTER_WINDOW = (LOOK_AROUND * 2) + 1 

# Reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. CHARGEMENT CHRONOLOGIQUE DES SIGNAUX BRUTS (9 CANAUX)
# ==============================================================================
print("Chargement des signaux bruts avec tri chronologique strict...")
X_train_full, y_train_full, subjects_train_full = load_raw_signals(BASE_DIR, dataset_type='train')
X_test, y_test, subjects_test = load_raw_signals(BASE_DIR, dataset_type='test')

# ==============================================================================
# 2. DÉFINITION DE L'ARCHITECTURE HYBRIDE CNN-TCN
# ==============================================================================
def build_cnn_tcn_model(input_shape=(128, 9)):
    """
    Remplace le LSTM par un bloc TCN (Temporal Convolutional Network).
    Utilise des convolutions causales (ne triche pas en regardant le futur) 
    et dilatées pour capturer la dynamique temporelle.
    """
    inputs = Input(shape=input_shape)
    
    # --- PHASE 1 : BLOC CNN (Le Bottleneck Spatio-Temporel) ---
    x = Conv1D(filters=FILTERS_BASE, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=32, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    # À ce stade, la séquence est réduite (ex: 8 trames compressées)

    # --- PHASE 2 : BLOC TCN (Analyse des séquences temporelles) ---
    # Convolution Causale + Dilatation de 1 (Regarde juste avant)
    x = Conv1D(filters=TCN_FILTERS, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(x)
    # Convolution Causale + Dilatation de 2 (Saute un pas en arrière)
    x = Conv1D(filters=TCN_FILTERS, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(x)
    # Convolution Causale + Dilatation de 4 (Vue large sur le passé)
    x = Conv1D(filters=TCN_FILTERS, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')(x)
    
    # Condensation du temps en un seul vecteur final
    x = GlobalAveragePooling1D()(x)
    
    # --- PHASE 3 : CLASSIFICATION ROBUSTE ---
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(6, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
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
    
    model_fold = build_cnn_tcn_model(input_shape=(128, 9))
    
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
        shuffle=False,  # Vital pour l'intégrité temporelle
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
print(f"\nEntraînement du CNN-TCN final sur les 21 sujets (Fixé à {optimal_epochs} époques)...")
final_model = build_cnn_tcn_model(input_shape=(128, 9))

final_model.fit(
    X_train_full, y_train_full,
    epochs=optimal_epochs,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)

# ==============================================================================
# 5. INFERENCE ET LISSAGE BIOLOGIQUE
# ==============================================================================
print("\nInférence sur le jeu de test (9 sujets vierges)...")
y_pred_probs = final_model.predict(X_test)
y_pred_raw = np.argmax(y_pred_probs, axis=1)

y_pred_smooth = apply_mode_filter(y_pred_raw, window_size=FILTER_WINDOW)

# ==============================================================================
# 6. ÉVALUATION ET AFFICHAGE BI-MATRICE
# ==============================================================================
evaluate_full_pipeline(y_test, y_pred_raw, y_pred_smooth)