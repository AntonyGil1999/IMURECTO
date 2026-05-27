import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Définition des chemins
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026\Train"
INERTIAL_DIR = os.path.join(BASE_DIR, "Inertial Signals")

fichiers = [
    "total_acc_x_train.txt", "total_acc_y_train.txt", "total_acc_z_train.txt",
    "body_acc_x_train.txt", "body_acc_y_train.txt", "body_acc_z_train.txt",
    "body_gyro_x_train.txt", "body_gyro_y_train.txt", "body_gyro_z_train.txt"
]
SUBJECT_FILE = os.path.join(BASE_DIR, "subject_train.txt")

# 2. Isolement du Sujet 1
print("Chargement des identifiants des sujets...")
subjects = np.loadtxt(SUBJECT_FILE, dtype=int)
sujet_cible = 1
indices_sujet = np.where(subjects == sujet_cible)[0]

print(f"Sujet {sujet_cible} isolé : {len(indices_sujet)} fenêtres trouvées.")

# 3. Préparation de la figure
fig, axes = plt.subplots(9, 1, figsize=(15, 20), sharex=True)
fig.suptitle(f"Visualisation brute : Intégralité du Sujet {sujet_cible}", fontsize=16, y=0.98)

# 4. Boucle de chargement et d'affichage
for i, nom_fichier in enumerate(fichiers):
    chemin = os.path.join(INERTIAL_DIR, nom_fichier)
    
    # Lecture du fichier complet
    donnees = np.loadtxt(chemin)
    
    # Filtre pour ne garder que les lignes du Sujet 1
    donnees_sujet = donnees[indices_sujet]
    
    # Aplatissement direct (les fenêtres sont mises bout à bout telles quelles)
    signal_complet = donnees_sujet.flatten()
    
    # Affichage
    print(f"[{i+1}/9] {nom_fichier} tracé ({len(signal_complet)} points)")
    axes[i].plot(signal_complet, color='blue', linewidth=0.5)
    
    # Mise en forme basique
    axes[i].set_ylabel(nom_fichier.replace('_train.txt', ''), rotation=0, labelpad=40, ha='right', fontweight='bold')
    axes[i].margins(x=0)

axes[-1].set_xlabel("Nombre d'échantillons bruts (avec chevauchement inclus)", fontsize=12)

print("\nGénération du graphique...")
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()