import numpy as np
import os

# 1. Définition des chemins exacts
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026\Train"
SIGNAL_FILE = os.path.join(BASE_DIR, "Inertial Signals", "total_acc_y_train.txt")
LABEL_FILE = os.path.join(BASE_DIR, "y_train.txt")

print("Chargement des fichiers pour vérification (Question 10)...")

# 2. Chargement brut des données
signaux = np.loadtxt(SIGNAL_FILE)
etiquettes = np.loadtxt(LABEL_FILE, dtype=int)

# 3. Récupération des dimensions
nb_fenetres_signaux = signaux.shape[0]
taille_fenetre = signaux.shape[1]
nb_etiquettes = etiquettes.shape[0]

# 4. Affichage des preuves
print("\n" + "="*50)
print("PREUVE MATHÉMATIQUE - POSITIONNEMENT DES ÉTIQUETTES")
print("="*50)

print(f"-> Fichier Signal (total_acc_y) : Il contient {nb_fenetres_signaux} lignes.")
print(f"-> Chaque ligne contient EXACTEMENT : {taille_fenetre} points (échantillons).")

print(f"\n-> Fichier Étiquettes (y_train) : Il contient {nb_etiquettes} lignes.")

try:
    points_par_etiquette = etiquettes.shape[1]
    print(f"-> Chaque ligne contient EXACTEMENT : {points_par_etiquette} étiquettes.")
except IndexError:
    print("-> Chaque ligne contient EXACTEMENT : 1 seule étiquette (C'est un vecteur 1D).")

print("-" * 50)

# 5. Conclusion logique
if nb_fenetres_signaux == nb_etiquettes:
    print("=> CONCLUSION :")
    print(f"Pour chaque bloc de {taille_fenetre} échantillons temporels, le dataset")
    print("ne fournit qu'UNE SEULE ET UNIQUE étiquette.")
    print("L'étiquette est donc bien positionnée par 'fenêtre globale' (2,56s)")
    print("et non pas point par point.")
else:
    print("=> ERREUR : Les fichiers ne correspondent pas.")