import numpy as np
import matplotlib.pyplot as plt
import os

# Chemin exact vers tes données brutes
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026\Train\Inertial Signals"

# Liste des 9 fichiers
fichiers = [
    "total_acc_x_train.txt", "total_acc_y_train.txt", "total_acc_z_train.txt",
    "body_acc_x_train.txt", "body_acc_y_train.txt", "body_acc_z_train.txt",
    "body_gyro_x_train.txt", "body_gyro_y_train.txt", "body_gyro_z_train.txt"
]

print("Chargement d'environ 1 million de points par capteur. Patientez...")

# Création d'une grande figure avec 9 lignes
fig, axes = plt.subplots(9, 1, figsize=(15, 20), sharex=True)
fig.suptitle("Visualisation 100% brute : Intégralité du Train Set", fontsize=16, y=0.98)

# Boucle pour lire et afficher chaque fichier
for i, nom_fichier in enumerate(fichiers):
    chemin = os.path.join(BASE_DIR, nom_fichier)
    
    # 1. Lecture du fichier texte
    donnees = np.loadtxt(chemin)
    
    # 2. Aplatissement : on met toutes les lignes (fenêtres) bout à bout en un seul long vecteur 1D
    signal_complet = donnees.flatten()
    
    # 3. Affichage
    print(f"[{i+1}/9] {nom_fichier} tracé ({len(signal_complet)} points)")
    axes[i].plot(signal_complet, color='blue', linewidth=0.1)
    
    # Mise en forme basique
    axes[i].set_ylabel(nom_fichier.replace('_train.txt', ''), rotation=0, labelpad=40, ha='right', fontweight='bold')
    axes[i].margins(x=0) # Retire les espaces blancs sur les côtés

axes[-1].set_xlabel("Nombre d'échantillons (Brut)", fontsize=12)

print("\nGénération de l'image (cela peut prendre quelques secondes vu la quantité de points)...")
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()