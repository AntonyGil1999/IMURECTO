import numpy as np
import os

def sort_chronologically(X, y, subjects):
    """
    Trie les données pour regrouper par sujet tout en conservant 
    strictement l'ordre chronologique des fenêtres (pas de shuffle temporel).
    """
    # mergesort est crucial ici car c'est un tri stable (conserve l'ordre initial)
    sort_indices = np.argsort(subjects.flatten(), kind='mergesort')
    
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]
    subjects_sorted = subjects[sort_indices]
    
    return X_sorted, y_sorted, subjects_sorted

def load_features(data_path, dataset_type='train'):
    """
    Charge les 561 features pré-calculées (pour SVM, RF, FCN, etc.)
    et corrige les labels de [1-6] à [0-5].
    """
    X_path = os.path.join(data_path, dataset_type, f'X_{dataset_type}.txt')
    y_path = os.path.join(data_path, dataset_type, f'y_{dataset_type}.txt')
    sub_path = os.path.join(data_path, dataset_type, f'subject_{dataset_type}.txt')

    X = np.loadtxt(X_path)
    y = np.loadtxt(y_path) - 1  # Ajustement pour Keras/Scikit-learn (0 à 5)
    subjects = np.loadtxt(sub_path)

    # Tri chronologique obligatoire
    return sort_chronologically(X, y, subjects)

def load_raw_signals(data_path, dataset_type='train'):
    """
    Charge les signaux bruts (9 canaux) pour les réseaux de neurones (CNN, LSTM).
    Format de sortie : (N_samples, 128, 9)
    """
    signal_types = [
        'body_acc_x_', 'body_acc_y_', 'body_acc_z_',
        'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
        'total_acc_x_', 'total_acc_y_', 'total_acc_z_'
    ]
    
    signal_dir = os.path.join(data_path, dataset_type, 'Inertial Signals')
    X_signals = []
    
    for sig in signal_types:
        filename = f"{sig}{dataset_type}.txt"
        filepath = os.path.join(signal_dir, filename)
        # Charge le fichier (N_samples, 128) et l'ajoute à la liste
        X_signals.append(np.loadtxt(filepath))
    
    # Transpose pour obtenir la forme (N_samples, 128_timesteps, 9_channels)
    X = np.dstack(X_signals)
    
    y_path = os.path.join(data_path, dataset_type, f'y_{dataset_type}.txt')
    sub_path = os.path.join(data_path, dataset_type, f'subject_{dataset_type}.txt')
    
    y = np.loadtxt(y_path) - 1
    subjects = np.loadtxt(sub_path)

    # Tri chronologique obligatoire
    return sort_chronologically(X, y, subjects)