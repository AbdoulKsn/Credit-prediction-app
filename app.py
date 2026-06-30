import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------------------------------

st.set_page_config(
    page_title="Système d'Octroi de Crédit",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------------
# STYLE CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    text-align: center;
    color: #0A3D62;
    font-size: 45px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.stButton>button {
    width: 100%;
    background-color: #0A3D62;
    color: white;
    border-radius: 10px;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1B4F72;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# CHARGEMENT DU MODELE
# -------------------------------------------------------

@st.cache_resource
def charger_modele():

    modele = joblib.load("modele_credit.pkl")
    scaler = joblib.load("scaler_credit.pkl")
    colonnes_modele = joblib.load("colonnes_modele.pkl")

    return modele, scaler, colonnes_modele


try:
    model, scaler, colonnes_modele = charger_modele()

except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.stop()

# -------------------------------------------------------
# TITRE
# -------------------------------------------------------

st.markdown(
    '<p class="title">🏦 Système Intelligent d\'Octroi de Crédit</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Prédiction automatique de l\'éligibilité d\'un client grâce au Machine Learning</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------
# FORMULAIRE
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Informations Personnelles")

    gender = st.selectbox(
        "Sexe",
        [0, 1],
        format_func=lambda x: "Femme" if x == 0 else "Homme"
    )

    married = st.selectbox(
        "Statut Matrimonial",
        [0, 1],
        format_func=lambda x: "Non Marié" if x == 0 else "Marié"
    )

    dependents = st.selectbox(
        "Nombre de personnes à charge",
        [0, 1, 2, 3]
    )

    education = st.selectbox(
        "Niveau d'étude",
        [0, 1],
        format_func=lambda x: "Diplômé" if x == 0 else "Non Diplômé"
    )

    self_employed = st.selectbox(
        "Travailleur Indépendant",
        [0, 1],
        format_func=lambda x: "Non" if x == 0 else "Oui"
    )

with col2:

    st.subheader("💰 Informations Financières")

    applicant_income = st.number_input(
        "Salaire du demandeur ($)",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Salaire du co-demandeur ($)",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Montant du prêt demandé ($)",
        min_value=0,
        value=150
    )

    loan_term = st.number_input(
        "Durée du prêt (mois)",
        min_value=1,
        value=360
    )

    credit_history = st.selectbox(
        "Historique de Crédit",
        [0, 1],
        format_func=lambda x: "Mauvais" if x == 0 else "Bon"
    )

    property_area = st.selectbox(
        "Zone de Résidence",
        [0, 1, 2],
        format_func=lambda x:
            "Rural" if x == 0
            else "Semi-Urbain" if x == 1
            else "Urbain"
    )

st.markdown("---")

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if st.button("🔍 Analyser le Dossier"):

    # Variables calculées

    total_income = applicant_income + coapplicant_income

    loan_income_ratio = loan_amount / (total_income + 1)

    # Création des données

    donnees = pd.DataFrame({

        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],

        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],

        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],

        'Total_Income': [total_income],
        'Loan_Income_Ratio': [loan_income_ratio]

    })

    # Ajouter automatiquement les colonnes manquantes

    for col in colonnes_modele:
        if col not in donnees.columns:
            donnees[col] = 0

    # Respecter l'ordre exact utilisé pendant l'entraînement

    donnees = donnees[colonnes_modele]

    try:

        # Normalisation

        donnees_scaled = scaler.transform(donnees)

        # Prédiction

        prediction = model.predict(donnees_scaled)[0]

        # Probabilité

        proba = model.predict_proba(donnees_scaled)[0][1] * 100

        st.markdown("## 📊 Résultat de l'Analyse")

        st.progress(int(proba))

        st.metric(
            label="Probabilité d'acceptation",
            value=f"{proba:.2f}%"
        )

        if prediction == 1:

            st.success(
                "✅ Crédit approuvé.\n\nLe profil du client présente un risque faible."
            )

            st.balloons()

        else:

            st.error(
                "❌ Crédit refusé.\n\nLe profil du client présente un risque élevé."
            )

        # Affichage des données

        with st.expander("📋 Informations analysées"):

            st.dataframe(donnees)

    except Exception as e:

        st.error(f"Erreur lors de la prédiction : {e}")
