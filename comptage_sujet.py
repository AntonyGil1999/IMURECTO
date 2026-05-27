import numpy as np
import os

# ==============================================================================
# CONFIGURATION DES CHEMINS (Identique à ton code principal)
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")

# ==============================================================================
# CHARGEMENT ET COMPTAGE DES SUJETS (Individus)
# ==============================================================================
# Lecture directe des fichiers contenant les identifiants des sujets
sujets_train_bruts = np.loadtxt(os.path.join(TRAIN_DIR, "subject_train.txt"), dtype=int)
sujets_test_bruts = np.loadtxt(os.path.join(TEST_DIR, "subject_test.txt"), dtype=int)

# Extraction des numéros uniques de sujets (sans doublons)
sujets_train_uniques = np.unique(sujets_train_bruts)
sujets_test_uniques = np.unique(sujets_test_bruts)

# ==============================================================================
# AFFICHAGE DES RÉSULTATS
# ==============================================================================
print("=" * 50)
print("ANALYSE DES INDIVIDUS (SUJETS) DU DATASET")
print("=" * 50)

print(f"Nombre de sujets dans le jeu TRAIN : {len(sujets_train_uniques)}")
print(f"Liste des numéros de sujets (Train) : {sujets_train_uniques.tolist()}")
print("-" * 50)

print(f"Nombre de sujets dans le jeu TEST  : {len(sujets_test_uniques)}")
print(f"Liste des numéros de sujets (Test)  : {sujets_test_uniques.tolist()}")
print("=" * 50)