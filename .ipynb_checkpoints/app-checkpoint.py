import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Charger le modèle sauvegardé
model_path = "random_forest_model.pkl"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Le fichier du modèle n'a pas été trouvé. Assurez-vous que 'random_forest_model.pkl' existe dans le répertoire.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Colonnes utilisées pour l'entraînement
columns_training = [
    'yearOfRegistration', 'powerPS', 'kilometer', 'monthOfRegistration',
    'notRepairedDamage', 'vehicleType_bus', 'vehicleType_cabrio', 
    'vehicleType_coupe', 'vehicleType_kleinwagen', 'vehicleType_kombi',
    'vehicleType_limousine', 'vehicleType_suv', 'fuelType_benzin', 
    'fuelType_cng', 'fuelType_diesel', 'fuelType_elektro', 
    'fuelType_hybrid', 'fuelType_lpg', 'gearbox_manuell', 
    'brand_audi', 'brand_bmw', 'brand_chevrolet', 'brand_chrysler', 
    'brand_citroen', 'brand_dacia', 'brand_daewoo', 'brand_daihatsu', 
    'brand_fiat', 'brand_ford', 'brand_honda', 'brand_hyundai', 
    'brand_jaguar', 'brand_jeep', 'brand_kia', 'brand_lada', 
    'brand_lancia', 'brand_land_rover', 'brand_mazda', 
    'brand_mercedes_benz', 'brand_mini', 'brand_mitsubishi', 
    'brand_nissan', 'brand_opel', 'brand_peugeot', 'brand_porsche', 
    'brand_renault', 'brand_rover', 'brand_saab', 'brand_seat', 
    'brand_skoda', 'brand_smart', 'brand_sonstige_autos', 
    'brand_subaru', 'brand_suzuki', 'brand_toyota', 'brand_trabant', 
    'brand_volkswagen', 'brand_volvo', 'model_a3', 'model_focus', 
    'model_golf', 'model_passat', 'model_polo'
]

# Interface utilisateur
st.title("Application de prédiction avec Machine Learning")
st.write("Veuillez entrer les caractéristiques :")

# Collecte des données utilisateur
yearOfRegistration = st.number_input("Année d'enregistrement (yearOfRegistration)", min_value=1900, max_value=2023, value=2010)
powerPS = st.number_input("Puissance en PS (powerPS)", min_value=1, max_value=1000, value=100)
model_input = st.selectbox("Modèle (model)", ['a3', 'focus', 'golf', 'passat', 'polo'])
kilometer = st.number_input("Kilométrage (kilometer)", min_value=0, max_value=200000, value=50000)
monthOfRegistration = st.number_input("Mois d'enregistrement (monthOfRegistration)", min_value=1, max_value=12, value=6)
notRepairedDamage = st.selectbox("Réparations nécessaires (notRepairedDamage)", ["ja", "nein"], index=1)

# Préparation des données utilisateur
input_data = {col: 0 for col in columns_training}  # Initialisez toutes les colonnes à 0

# Mettre à jour les colonnes avec les valeurs utilisateur
input_data['yearOfRegistration'] = yearOfRegistration
input_data['powerPS'] = powerPS
input_data['kilometer'] = kilometer
input_data['monthOfRegistration'] = monthOfRegistration
input_data['notRepairedDamage'] = 1 if notRepairedDamage == "ja" else 0
input_data[f"model_{model_input}"] = 1  # Activez le modèle sélectionné

# Encodage one-hot pour vehicleType
vehicle_type = st.selectbox("Type de véhicule (vehicleType)", ['bus', 'cabrio', 'coupe', 'kleinwagen', 'kombi', 'limousine', 'suv'])
input_data[f"vehicleType_{vehicle_type}"] = 1

# Encodage one-hot pour fuelType
fuel_type = st.selectbox("Type de carburant (fuelType)", ['benzin', 'cng', 'diesel', 'elektro', 'hybrid', 'lpg'])
input_data[f"fuelType_{fuel_type}"] = 1

# Encodage one-hot pour brand
brand = st.selectbox("Marque (brand)", [
    'audi', 'bmw', 'chevrolet', 'chrysler', 'citroen', 'dacia', 'daewoo', 'daihatsu',
    'fiat', 'ford', 'honda', 'hyundai', 'jaguar', 'jeep', 'kia', 'lada', 'lancia',
    'land_rover', 'mazda', 'mercedes_benz', 'mini', 'mitsubishi', 'nissan', 'opel',
    'peugeot', 'porsche', 'renault', 'rover', 'saab', 'seat', 'skoda', 'smart',
    'sonstige_autos', 'subaru', 'suzuki', 'toyota', 'trabant', 'volkswagen', 'volvo'
])
input_data[f"brand_{brand}"] = 1

# Convertir en DataFrame
features = pd.DataFrame([input_data])

# Prédiction
if st.button("Prédire"):
    try:
        prediction = model.predict(features)
        st.write(f"La prédiction est : {prediction[0]}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
