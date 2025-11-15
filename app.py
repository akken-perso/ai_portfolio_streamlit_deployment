# 1. Imports et Chargement du Modèle
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Note aux utilisateurs :
# Charger le modèle (Assurez-vous que le chemin est correct !)
# Vous devez aussi recharger le scaler utilisé pour les colonnes numériques !
# Pour simplifier, nous allons supposer que l'entraînement utilise un scaler simple.

# Pour un vrai projet, le modèle et le scaler DOIVENT être sauvegardés ensemble.
# Pour l'exercice, chargez uniquement le modèle Random Forest.
model = joblib.load('saved_models/final_titanic_classifier.pkl')

# --- Mise en place de la page Streamlit ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Prédicteur de Survie du Titanic (Modèle Random Forest)")
st.write("Entrez les caractéristiques d'un passager pour prédire sa probabilité de survie.")

# --- Création des champs d'entrée (Widgets) ---

with st.form("prediction_form"):
    # Caractéristiques Numériques
    age = st.slider("Âge (Age)", 0.0, 80.0, 30.0)
    fare = st.slider("Tarif (Fare)", 0.0, 500.0, 50.0)
    sibsp = st.slider("Nombre de Frères/Époux (SibSp)", 0, 8, 0)
    parch = st.slider("Nombre de Parents/Enfants (Parch)", 0, 6, 0)
    pclass = st.selectbox("Classe de Billet (Pclass)", [1, 2, 3], index=2)

    # Caractéristiques Catégorielles (Simulant l'Encodage One-Hot)
    sex = st.radio("Sexe", ('male', 'female'))
    embarked = st.selectbox("Port d'Embarquement", ('S', 'C', 'Q'))
    title_raw = st.selectbox("Titre", ('Mr', 'Miss', 'Mrs', 'Rare'))

    submitted = st.form_submit_button("Prédire la Survie")

# --- Logique de Traitement et de Prédiction ---

if submitted:
    # Création du DataFrame d'entrée (Important : l'ordre des colonnes DOIT correspondre à X_train)
    input_df = pd.DataFrame({
        'Pclass': [pclass], 
        'Age': [age], 
        'SibSp': [sibsp], 
        'Parch': [parch], 
        'Fare': [fare], 
        # Les colonnes encodées commencent ici
        'Sex_male': [1 if sex == 'male' else 0],
        
        # Encodage Embarked (S, C, Q)
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0],
        
        # Encodage des Titres (Mr, Miss, Mrs, Rare)
        'Title_Miss': [1 if title_raw == 'Miss' else 0],
        'Title_Mr': [1 if title_raw == 'Mr' else 0],
        'Title_Mrs': [1 if title_raw == 'Mrs' else 0],
        'Title_Rare': [1 if title_raw == 'Rare' else 0],
    })
    
    # --- IMPORTANT : Application du Scaler si nécessaire ---
    # Pour un Random Forest, le scaling n'est PAS obligatoire,
    # mais si vous aviez utilisé la Régression Logistique,
    # vous auriez dû recharger le StandardScaler et l'appliquer ici !
    
    # 1. Obtenir la prédiction et les probabilités
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1][0] # Probabilité de survie

    # 2. Affichage des Résultats
    st.markdown("---")
    if prediction[0] == 1:
        st.success(f"✅ Le passager **survivrait** (Probabilité: {prediction_proba*100:.2f}%)")
    else:
        st.error(f"❌ Le passager **ne survivrait pas** (Probabilité: {100 - prediction_proba*100:.2f}%)")
    
    st.bar_chart({'Mort': 1 - prediction_proba, 'Survie': prediction_proba})
