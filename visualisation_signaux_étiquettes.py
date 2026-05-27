# Fichier : visualisation_9_signaux.py
# Objectif : Visualiser la continuité temporelle des 9 signaux bruts.
# Chargement STRICTEMENT DIRECT depuis les fichiers .txt d'origine.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 1. Définition des chemins
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026\Train"
INERTIAL_DIR = os.path.join(BASE_DIR, "Inertial Signals")

# Les 9 fichiers à charger
signal_files = [
    "total_acc_x_train.txt", "total_acc_y_train.txt", "total_acc_z_train.txt",
    "body_acc_x_train.txt", "body_acc_y_train.txt", "body_acc_z_train.txt",
    "body_gyro_x_train.txt", "body_gyro_y_train.txt", "body_gyro_z_train.txt"
]

LABEL_FILE = os.path.join(BASE_DIR, "y_train.txt")
SUBJECT_FILE = os.path.join(BASE_DIR, "subject_train.txt")

# Dictionnaire des activités
activity_map = {
    1: ('Marche', '#1f77b4'),           # Bleu
    2: ('Monter escaliers', '#ff7f0e'), # Orange
    3: ('Descendre escaliers', '#2ca02c'),# Vert
    4: ('Assis', '#d62728'),            # Rouge
    5: ('Debout', '#9467bd'),           # Violet
    6: ('Allongé', '#8c564b')           # Marron
}

# 2. Chargement des labels et sujets
print("Lecture des étiquettes et des sujets...")
y_train = np.loadtxt(LABEL_FILE, dtype=int)
subjects = np.loadtxt(SUBJECT_FILE, dtype=int)

sujet_cible = 5

indices_sujet = np.where(subjects == sujet_cible)[0]
y_sujet = y_train[indices_sujet]

print(f"Sujet {sujet_cible} isolé : {len(indices_sujet)} fenêtres à traiter.")

# 3. Reconstruction de la ligne de temps des étiquettes (gère le chevauchement)
labels_continus = []
labels_continus.extend([y_sujet[0]] * 128)
for i in range(1, len(y_sujet)):
    labels_continus.extend([y_sujet[i]] * 64)
labels_continus = np.array(labels_continus)
temps = np.arange(len(labels_continus)) / 50.0  # Axe du temps en secondes

# 4. Chargement et reconstruction des 9 signaux
signaux_continus = []

for filename in signal_files:
    print(f"Chargement de {filename}...")
    filepath = os.path.join(INERTIAL_DIR, filename)
    
    # Chargement du fichier txt brut
    data_brute = np.loadtxt(filepath)
    # Isolement du sujet 1
    data_sujet = data_brute[indices_sujet]
    
    # Reconstruction (chevauchement 50%)
    sig_continu = []
    sig_continu.extend(data_sujet[0, :])
    for i in range(1, len(data_sujet)):
        sig_continu.extend(data_sujet[i, 64:128])
        
    signaux_continus.append(np.array(sig_continu))

print("\nGénération du graphique complet...")

# 5. Création du graphique (9 sous-graphiques)
fig, axes = plt.subplots(9, 1, figsize=(18, 22), sharex=True)
fig.suptitle(f"Sujet {sujet_cible} : Visualisation des 9 Signaux Bruts et Transitions", fontsize=16, y=0.99)

# Repérage des changements de zones (pour colorer le fond)
changement_idx = np.where(labels_continus[:-1] != labels_continus[1:])[0]

# Boucle sur chaque capteur
for i, ax in enumerate(axes):
    nom_signal = signal_files[i].replace("_train.txt", "")
    ax.plot(temps, signaux_continus[i], color='black', linewidth=0.5)
    ax.set_ylabel(nom_signal, rotation=0, labelpad=40, ha='right', fontsize=9, fontweight='bold')
    
    # Coloration de l'arrière-plan pour CHAQUE sous-graphique
    debut_zone = 0
    for fin_zone in changement_idx:
        label = labels_continus[debut_zone]
        _, couleur = activity_map[label]
        ax.axvspan(temps[debut_zone], temps[fin_zone], color=couleur, alpha=0.3)
        debut_zone = fin_zone + 1
        
    # Dernière zone
    label_fin = labels_continus[debut_zone]
    _, couleur_fin = activity_map[label_fin]
    ax.axvspan(temps[debut_zone], temps[-1], color=couleur_fin, alpha=0.3)

# 6. Mise en page finale
axes[-1].set_xlabel("Temps (secondes)", fontsize=12, fontweight='bold')

# Légende unique placée en haut
legend_patches = [mpatches.Patch(color=c, alpha=0.3, label=n) for k, (n, c) in activity_map.items()]
fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.99, 0.98), ncol=6)

plt.tight_layout()
plt.subplots_adjust(top=0.96) # Laisse de la place pour le titre et la légende
plt.show()