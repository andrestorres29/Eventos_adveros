# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:18:35 2024

@author: torre
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

def main():
    st.title("Implementación con LR / SVM")
    
    # Crear columnas para la imagen y los datos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        image = Image.open('logo.png')
        st.image(image, use_column_width=True)
        st.write('### Irvin A. Torres')
        st.write('### Matrícula: 315463')
        st.write('### Machine Learning MIC')
    
    with col2:
        year = st.number_input('Año:', min_value=2005.0, max_value=2015.0)
        county = st.selectbox('Condado:', options=['Statewide', 'Alameda', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Modoc', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Plumas', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba'])
    
    with col3:
        psi_description = st.selectbox('Descripción PSI:', options=['Retained Surgical Item or Unretrieved Device Fragment', 'Iatrogenic Pneumothorax', 'Central Venous Catheter-Related Blood Stream Infection', 'Postoperative Wound Dehiscence', 'Accidental Puncture or Laceration', 'Transfusion Reaction', 'Perioperative Hemorrhage or Hematoma']) 
        model_option = st.selectbox("Selecciona el modelo para la predicción:", options=["Regresion lineal", "SVM", "Random forest", "Ensamble"])

    # Diccionario con los modelos y sus valores de R²
    model_r2_values = {
        "Regresion lineal": 0.7945, 
        "SVM": 0.8583,  
        "Random forest": 0.9833,  
        "Ensamble": 0.8711   
    }

    # Diccionario con las rutas de los modelos guardados
    model_files = {
        "Regresion lineal": 'best_model_lineal.sav',
        "SVM": 'bestmodelSVM.sav',
        "Random forest": 'Random_forest.sav',
        "Ensamble": 'stacked_regressor_model_comprimido.sav'
    }

    # Cargar el modelo seleccionado
    model = joblib.load(model_files[model_option])

    # Mostrar el R² correspondiente al modelo seleccionado
    r2_value = model_r2_values[model_option]
    st.write(f"R² del modelo {model_option}: {r2_value:.2f}")

    # Botón para realizar la predicción
    if st.button('Predecir'):
        input_data = pd.DataFrame({'Year': [year], 'County': [county], 'PSIDescription': [psi_description]})
        prediction = model.predict(input_data)
        st.write(f'Predicción de ObsRate: {prediction[0]}')

if __name__ == '__main__': 
    main()
