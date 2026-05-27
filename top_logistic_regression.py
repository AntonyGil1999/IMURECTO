import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# Importation de notre architecture modulaire
from data_loader import load_features

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
BASE_DIR = r"C:\Dev\Python\ML\Projet IMU\MAJ_12_05_2026"
# Le fichier texte contenant le nom des 561 variables (fourni dans le dataset original)
FEATURES_FILE = os.path.join(BASE_DIR, "features.txt") 

ACTIVITIES = ["Marche", "Monter", "Descendre", "Assis", "Debout", "Allongé"]

# ==============================================================================
# 1. CHARGEMENT DES NOMS DE VARIABLES ET DES DONNÉES
# ==============================================================================
print("Chargement des données expertes...")
X_train_full, y_train_full, subjects_train_full = load_features(BASE_DIR, dataset_type='train')

# Tentative de chargement du nom réel des 561 features
feature_names = []
try:
    with open(FEATURES_FILE, 'r') as f:
        # Chaque ligne ressemble à "1 tBodyAcc-mean()-X"
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_names.append(" ".join(parts[1:]))
    print("Noms des 561 variables chargés avec succès !")
except FileNotFoundError:
    print(f"Fichier {FEATURES_FILE} introuvable. Utilisation de noms génériques.")
    feature_names = [f"Feature_{i+1}" for i in range(561)]

feature_names = np.array(feature_names)

# ==============================================================================
# 2. ENTRAÎNEMENT DE LA BOÎTE BLANCHE (REGRESSION LOGISTIQUE)
# ==============================================================================
print("\nEntraînement de la Régression Logistique (Boîte Blanche)...")
# On utilise les meilleurs paramètres trouvés lors de notre GridSearch précédent
lr_model = LogisticRegression(C=10, solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train_full, y_train_full)

# ==============================================================================
# 3. EXTRACTION DU CERVEAU DE L'ALGORITHME (EXPLAINABLE AI)
# ==============================================================================
print("\n" + "="*60)
print("🧠 TOP 10 DES FEATURES LES PLUS IMPORTANTES PAR ACTIVITÉ 🧠")
print("="*60)

# lr_model.coef_ contient les poids. Sa dimension est (6 classes, 561 features)
coefficients = lr_model.coef_

for i, activity in enumerate(ACTIVITIES):
    print(f"\n▶ Pour détecter l'activité : {activity.upper()}")
    
    # Récupération des poids pour la classe actuelle
    class_weights = coefficients[i]
    
    # argsort trie du plus petit au plus grand. On prend les 3 derniers (les plus grands)
    # et on les inverse [::-1] pour avoir le 1er, 2ème, 3ème.
    top_10_indices = np.argsort(class_weights)[-10:][::-1]
    
    for rank, idx in enumerate(top_10_indices):
        feat_name = feature_names[idx]
        weight = class_weights[idx]
        print(f"  {rank+1}. {feat_name} (Poids d'importance : {weight:.2f})")
        
print("\n" + "="*60)