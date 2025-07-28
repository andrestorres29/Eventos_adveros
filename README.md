# ⚠️ Análisis y Predicción de Eventos Adversos - California

Este proyecto tiene como objetivo analizar eventos adversos reportados en el estado de **California**, y predecir tanto la **cantidad** como el **tipo de evento** ocurrido, considerando datos agregados por condado (county) y a nivel estatal.

Se aplicaron modelos de machine learning para identificar patrones espaciales y temporales, permitiendo generar alertas tempranas y visualizaciones útiles para la toma de decisiones.

---

## 🎯 Objetivo

- Analizar eventos adversos registrados en California.
- Identificar tendencias por tipo de evento, región y periodo.
- Predecir el número de eventos y su tipología por condado.
- Proveer visualizaciones geográficas del riesgo.

---

## 📁 Dataset

- Datos oficiales de eventos adversos del estado de California.
- Variables clave: tipo de evento, fecha, severidad, ubicación (county), y categoría.
- Dataset preprocesado para entrenamiento y prueba de modelos predictivos.

---

## 🧰 Herramientas y tecnologías utilizadas

- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Modelos de clasificación y regresión
- Mapas y análisis espacial con `plotly`, `folium` o `geopandas` 
- Jupyter Notebook
- GitHub para control de versiones

---

## 🤖 Modelos implementados

- Clasificación del tipo de evento adverso según características contextuales.
- Regresión para predecir la cantidad de eventos por condado.
- Comparativa de rendimiento entre modelos base (Random Forest, Regresión Lineal, etc.).
- Los resultado fueron los siguientes:
Voting Regressor R^2: 0.8000
AdaBoost Regressor R^2: 0.8424
Bagging Regressor R^2: 0.8550
Gradient Boosting Regressor R^2: 0.8711
XGBoost Regressor R^2: 0.8323

---

## 🚀 Cómo ejecutar

1. Clona el repositorio:
```bash
git clone https://github.com/andrestorres29/Proyecto-final-eventos-adversos.git
cd Proyecto-final-eventos-adversos
