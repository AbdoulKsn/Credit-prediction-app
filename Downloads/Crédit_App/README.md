# 🏦 Système Intelligent d'Octroi de Crédit

## 📌 Description

Ce projet est une application web développée avec **Streamlit** permettant de prédire l'éligibilité d'un client à l'obtention d'un prêt bancaire grâce au **Machine Learning**.

L'application analyse les informations personnelles et financières d'un demandeur puis estime automatiquement si le prêt a de fortes chances d'être **accordé** ou **refusé**.

Ce projet a été réalisé dans le cadre d'un projet pratique en Intelligence Artificielle et Machine Learning.

---

## 🚀 Fonctionnalités

* Interface web intuitive développée avec Streamlit
* Prédiction instantanée de l'éligibilité à un prêt
* Calcul automatique de la probabilité d'acceptation
* Prétraitement automatique des données
* Utilisation d'un modèle de Machine Learning entraîné sur un jeu de données bancaire
* Affichage des informations analysées

---

## 🧠 Technologies utilisées

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Joblib
* Streamlit

---

## 📊 Variables utilisées

Le modèle prend en compte les informations suivantes :

* Sexe
* Statut matrimonial
* Nombre de personnes à charge
* Niveau d'étude
* Travailleur indépendant
* Salaire du demandeur
* Salaire du co-demandeur
* Montant du prêt demandé
* Durée du prêt
* Historique de crédit
* Zone de résidence

Des variables dérivées sont également calculées automatiquement :

* Revenu total
* Ratio Prêt / Revenu

---

## 🤖 Modèle de Machine Learning

Le modèle utilisé est un **Random Forest Classifier**.

### Étapes de préparation des données

* Nettoyage des valeurs manquantes
* Encodage des variables catégorielles
* Création de nouvelles variables (Feature Engineering)
* Standardisation des données avec `StandardScaler`
* Équilibrage des classes grâce à **SMOTE**
* Entraînement du modèle Random Forest

Les fichiers générés sont :

* `modele_credit.pkl`
* `scaler_credit.pkl`
* `colonnes_modele.pkl`

---

## 📁 Structure du projet

```text
Credit_App/
│
├── app.py
├── modele_credit.pkl
├── scaler_credit.pkl
├── colonnes_modele.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebook/
    └── Projet_Credit_Bancaire.ipynb
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/votre-utilisateur/Credit_App.git
```

### 2. Accéder au dossier

```bash
cd Credit_App
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

---

## 📷 Aperçu de l'application

L'application permet de :

* saisir les informations du demandeur ;
* analyser automatiquement le dossier ;
* afficher la probabilité d'acceptation du prêt ;
* indiquer si le crédit est accordé ou refusé.

---

## 📈 Améliorations apportées

Afin d'améliorer les performances du modèle, plusieurs optimisations ont été réalisées :

* utilisation de **SMOTE** pour équilibrer les classes ;
* optimisation des paramètres du Random Forest ;
* création de nouvelles variables explicatives ;
* standardisation des données ;
* amélioration de la précision des prédictions.

---

## 🎯 Perspectives d'amélioration

* Déploiement sur Streamlit Cloud
* Utilisation de XGBoost ou LightGBM
* Explication des prédictions avec SHAP
* Authentification des utilisateurs
* Historique des demandes de prêt
* Connexion à une base de données

---

## 👨‍💻 Auteur

**Abdoul Koné**

Master en Informatique – Développement d'Applications

---

## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique et peut être librement utilisé à des fins d'apprentissage.
