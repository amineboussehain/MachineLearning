import streamlit as st
import pandas as pd
import pickle

# Charger le modèle sauvegardé
model_path = "random_forest_model.pkl"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Colonnes attendues par le modèle
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("Impossible d'extraire les colonnes attendues. Vérifiez le modèle.")
    st.stop()

# Interface utilisateur
st.title("Application de prédiction avec Machine Learning")

# Saisie des données utilisateur
input_data = {col: 0 for col in expected_columns}  # Initialisation des colonnes à 0
input_data['yearOfRegistration'] = st.number_input("Année d'enregistrement", min_value=1900, max_value=2023, value=2010)
input_data['powerPS'] = st.number_input("Puissance en PS", min_value=1, max_value=1000, value=100)
input_data['kilometer'] = st.number_input("Kilométrage", min_value=0, max_value=200000, value=50000)
input_data['monthOfRegistration'] = st.number_input("Mois d'enregistrement", min_value=1, max_value=12, value=6)
input_data['notRepairedDamage'] = st.selectbox("Réparations nécessaires", ["ja", "nein"], index=1)
input_data['notRepairedDamage'] = 1 if input_data['notRepairedDamage'] == "ja" else 0

# Encodage dynamique
vehicle_type = st.selectbox("Type de véhicule", ['bus', 'cabrio', 'coupe', 'kleinwagen', 'kombi', 'limousine', 'suv'])
input_data[f"vehicleType_{vehicle_type}"] = 1

fuel_type = st.selectbox("Type de carburant", ['benzin', 'cng', 'diesel', 'elektro', 'hybrid', 'lpg'])
input_data[f"fuelType_{fuel_type}"] = 1

brand = st.selectbox("Marque", [
    'audi', 'bmw', 'chevrolet', 'chrysler', 'citroen', 'dacia', 'daewoo', 'daihatsu',
    'fiat', 'ford', 'honda', 'hyundai', 'jaguar', 'jeep', 'kia', 'lada', 'lancia',
    'land_rover', 'mazda', 'mercedes_benz', 'mini', 'mitsubishi', 'nissan', 'opel',
    'peugeot', 'porsche', 'renault', 'rover', 'saab', 'seat', 'skoda', 'smart',
    'sonstige_autos', 'subaru', 'suzuki', 'toyota', 'trabant', 'volkswagen', 'volvo'
])
input_data[f"brand_{brand}"] = 1

model_input = st.selectbox("Modèle", ['a3', 'focus', 'golf', 'passat', 'polo'])
input_data[f"model_{model_input}"] = 1

# Conversion en DataFrame
features = pd.DataFrame([input_data])

# Gestion des colonnes manquantes et ordre
for col in expected_columns:
    if col not in features.columns:
        features[col] = 0
features = features[expected_columns]

# Prédiction
if st.button("Prédire"):
    try:
        prediction = model.predict(features)
        st.write(f"La prédiction est : {prediction[0]}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
