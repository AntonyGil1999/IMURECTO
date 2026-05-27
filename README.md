\# IMURECTO : Reconnaissance d'Activité Humaine (HAR) par signaux inertiels



Ce projet met en place un pipeline complet de Machine Learning et Deep Learning pour classifier 6 activités physiques (Marche, Monter, Descendre, Assis, Debout, Allongé) à partir des signaux inertiels bruts et calculés d'un smartphone (Accéléromètre \& Gyroscope). 



L'étude s'appuie sur le dataset public \*\*Human Activity Recognition Using Smartphones\*\*.

\*(Note : Les dossiers `Train/` et `Test/` ne sont pas inclus dans ce dépôt. Vous pouvez télécharger le dataset original \[ici - insère le lien UCI] et placer les dossiers à la racine).\*
Mais on peut retrouver les données directement ici : https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones



\## 🏗️ Architecture du Projet



Le code est conçu de manière modulaire, respectant la chronologie des signaux et empêchant toute fuite de données entre les sujets.



\### 1. Modules Communs

\* `data\_loader.py` : Chargement chronologique stable des features et des signaux bruts.

\* `cross\_validation.py` : Génération de GroupKFold stricts par sujets (zéro data leakage).

\* `post\_processing.py` : Filtre mode temporel glissant pour garantir la continuité biomécanique.

\* `evaluation.py` : Bilan macro et génération comparative des matrices de confusion (Brut vs Lissé).



\### 2. Modélisation Classique \& Optimisation

Implémentation et optimisation par GridSearch de multiples algorithmes :

\* Modèles de base : SVM (`train\_svm.py`), KNN, Régression Logistique, Arbres de décision.

\* Modèles ensemblistes : Random Forest, XGBoost, AdaBoost, Gradient Boosting.

\* \*\*Méta-modèle optimisé :\*\* `train\_hierarchical\_state\_machine.py` et `train\_voting\_classifier.py` isolent la dynamique de la statique, atteignant une précision lissée de \*\*97,62 %\*\* avec un sous-ensemble réduit de 52 caractéristiques expertes.



\### 3. Deep Learning \& Analyse Séquentielle

Entraînement direct sur les matrices de signaux bruts (128 échantillons x 9 canaux) :

\* `train\_cnn.py` : Réseau convolutif 1D en goulot d'étranglement.

\* Modèles récurrents (`train\_cnn\_lstm.py`, `train\_cnn\_bilstm.py`). 

\* \*Axe de recherche : Mise en évidence des limites des architectures récurrentes face aux coupures artificielles (absence de transitions) du dataset d'origine.\*



\### 4. Exploration \& Visualisation

\* Scripts d'analyse des signaux temporels (`visualisation\_signaux\_étiquettes.py`, etc.).

\* Explicabilité : `top\_logistic\_regression.py` pour isoler les variables biomécaniques les plus discriminantes.



\## 🚀 Comment exécuter

1\. Cloner le dépôt.

2\. Télécharger les dossiers `Train` et `Test` du dataset UCI HAR à la racine.

3\. Lancer n'importe quel script de la famille `train\_...py`. Le pipeline modulaire se charge de l'import, de la validation croisée, du lissage et de l'affichage du double bilan final.

