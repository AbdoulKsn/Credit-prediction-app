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
# CHARGEMENT MODELE
# -------------------------------------------------------

@st.cache_resource
def charger_modele():
    modele = joblib.load("random_forest_credit.pkl")
    scaler = joblib.load("scaler_credit.pkl")
    return modele, scaler

try:
    model, scaler = charger_modele()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

# -------------------------------------------------------
# TITRE
# -------------------------------------------------------

st.markdown(
    '<p class="title">🏦 Système Intelligent d\'Octroi de Crédit</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Analyse bancaire basée sur IA + règles métier</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------
# FORMULAIRE
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Informations personnelles")

    gender = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")

    married = st.selectbox("Marié", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

    dependents = st.selectbox("Personnes à charge", [0, 1, 2, 3])

    education = st.selectbox("Éducation", [0, 1], format_func=lambda x: "Diplômé" if x == 0 else "Non diplômé")

    self_employed = st.selectbox("Indépendant", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

with col2:

    st.subheader("💰 Informations financières")

    total_income = st.number_input("Revenu total", min_value=0, value=5000)

    loan_amount = st.number_input("Montant du prêt", min_value=0, value=150)

    loan_term = st.number_input("Durée (mois)", min_value=1, value=360)

    credit_history = st.selectbox(
        "Historique de crédit",
        [0, 1],
        format_func=lambda x: "Mauvais" if x == 0 else "Bon"
    )

    property_area = st.selectbox(
        "Zone",
        [0, 1, 2],
        format_func=lambda x: "Rural" if x == 0 else "Semi-Urbain" if x == 1 else "Urbain"
    )

st.markdown("---")

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if st.button("🔍 Analyser le Dossier"):

    # ---------------- DATA ----------------
    donnees = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'Total_Income': [total_income]
    })

    # ---------------- ML ----------------
    donnees_scaled = scaler.transform(donnees)

    prediction = model.predict(donnees_scaled)[0]
    proba_ml = model.predict_proba(donnees_scaled)[0][1] * 100

    # ---------------- RÈGLES BANCAIRES ----------------
    risk_score = 0

    # mauvais historique = refus quasi automatique
    if credit_history == 0:
        risk_score += 50

    # ratio prêt/revenu
    ratio = loan_amount / (total_income + 1)

    if ratio > 0.5:
        risk_score += 30
    elif ratio > 0.3:
        risk_score += 15

    # dépendants
    if dependents == 3:
        risk_score += 10

    # indépendant
    if self_employed == 1:
        risk_score += 5

    # prêt trop élevé
    if loan_amount > 10000:
        risk_score += 10

    # ---------------- SCORE FINAL ----------------
    final_score = (proba_ml * 0.6) + (risk_score * 0.4)

    # ---------------- AFFICHAGE ----------------
    st.markdown("## 📊 Résultat de l'analyse bancaire")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Probabilité ML", f"{proba_ml:.2f}%")

    with colB:
        st.metric("Score risque", f"{risk_score}/100")

    with colC:
        st.metric("Score final", f"{final_score:.2f}")

    st.progress(int(min(proba_ml, 100)))

    st.markdown("---")

    # ---------------- DÉCISION BANQUE ----------------

    if credit_history == 0 or final_score > 70:

        st.error("❌ CRÉDIT REFUSÉ")
        st.warning("Risque bancaire trop élevé")

    elif final_score > 45:

        st.warning("⚠️ CRÉDIT ACCORDÉ SOUS CONDITIONS")
        st.info("Taux d’intérêt recommandé élevé")

    else:

        st.success("✅ CRÉDIT ACCORDÉ")
        st.balloons()

    # ---------------- DETAILS ----------------
    with st.expander("📋 Voir les données analysées"):
        st.dataframe(donnees)
