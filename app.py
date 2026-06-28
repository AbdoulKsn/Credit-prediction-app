import streamlit as st
import pandas as pd
import joblib

# 1. Configuration du titre de la page
st.set_page_config(page_title="Octroi de Crédit", layout="centered")
st.title("🏦 Simulateur d'Octroi de Crédit Bancaire")
st.write("Saisissez les informations du client pour prédire son éligibilité.")

# 2. Chargement sécurisé du modèle et du scaler
@st.cache_resource # Évite de recharger le fichier à chaque clic
def charger_fichiers():
    modele = joblib.load("random_forest_credit.pkl")
    scaler = joblib.load("scaler_credit.pkl") # Supprimez si non utilisé
    return modele, scaler

try:
    model, scaler = charger_fichiers()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")

# 3. Création des champs de saisie pour l'utilisateur
st.subheader("Informations financières")
revenu = st.number_input("Revenu mensuel du demandeur ($)", min_value=0, value=5000)
montant = st.number_input("Montant du prêt demandé ($)", min_value=0, value=150)

# 4. Bouton de prédiction et calcul
if st.button("Analyser le dossier"):
    # Créer un DataFrame avec les mêmes noms de colonnes que l'entraînement
    donnies_saisies = pd.DataFrame({
        "ApplicantIncome": [revenu],
        "LoanAmount": [montant]
    })
    
    # Appliquer le scaler si nécessaire
    donnies_preparees = scaler.transform(donnies_saisies)
    
    # Prédire (0 ou 1)
    prediction = model.predict(donnies_preparees)[0]
    
    # Affichage du résultat final
    st.markdown("---")
    if prediction == 1:
        st.success("🎉 **Favorable** : Le crédit est pré-approuvé.")
    else:
        st.error("❌ **Défavorable** : Le risque est trop élevé pour accorder le prêt.")
