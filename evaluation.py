import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Dictionnaire global pour les activités
ACTIVITY_LABELS = {
    0: 'Marche',
    1: 'Monter',
    2: 'Descendre',
    3: 'Assis',
    4: 'Debout',
    5: 'Allongé'
}

def print_evaluation_report(y_true, y_pred, title="Rapport de Classification"):
    """
    Affiche le rapport de classification avec average='macro'.
    """
    print(f"\n{title}")
    print("-" * len(title))
    target_names = [ACTIVITY_LABELS[i] for i in range(6)]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

def plot_comparative_confusion_matrices(y_true, y_pred_raw, y_pred_smooth):
    """
    Affiche côte à côte deux matrices de confusion : l'une sans filtre, l'autre avec le lissage temporel.
    Idéal pour mettre en évidence la correction des confusions (ex: Assis/Debout).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    target_names = [ACTIVITY_LABELS[i] for i in range(6)]
    
    # Matrice Sans Filtre
    cm_raw = confusion_matrix(y_true, y_pred_raw)
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=target_names, yticklabels=target_names)
    axes[0].set_title('Matrice de Confusion : SANS Lissage', fontweight='bold')
    axes[0].set_ylabel('Vérité Terrain')
    axes[0].set_xlabel('Prédiction')
    
    # Matrice Avec Filtre
    cm_smooth = confusion_matrix(y_true, y_pred_smooth)
    sns.heatmap(cm_smooth, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=target_names, yticklabels=target_names)
    axes[1].set_title('Matrice de Confusion : AVEC Lissage Temporel', fontweight='bold')
    axes[1].set_ylabel('Vérité Terrain')
    axes[1].set_xlabel('Prédiction')
    
    plt.tight_layout()
    plt.show()
    
def evaluate_full_pipeline(y_true, y_pred_raw, y_pred_smooth):
    """
    Fonction englobante pour générer tout le bilan d'un modèle d'un seul coup.
    """
    print_evaluation_report(y_true, y_pred_raw, title="RÉSULTATS : Modèle Brut (Sans Filtre)")
    print_evaluation_report(y_true, y_pred_smooth, title="RÉSULTATS : Modèle + Lissage Temporel")
    plot_comparative_confusion_matrices(y_true, y_pred_raw, y_pred_smooth)