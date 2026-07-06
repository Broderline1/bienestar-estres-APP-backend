"""
Este script genera 2 graficas para demostrar que el modelo de regresion funciona bien.
Correrlo con: python graficas.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# Cargamos los datos y preparamos igual que en entrenar.py
# ------------------------------------------------------------------
df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
df[["Presion_Sistolica", "Presion_Diastolica"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)

codificadores = {}
for col in ["Gender", "Occupation", "BMI Category"]:
    le = LabelEncoder()
    df[col + "_cod"] = le.fit_transform(df[col])
    codificadores[col] = le

columnas_features = [
    "Age", "Sleep Duration", "Physical Activity Level",
    "Heart Rate", "Daily Steps", "Presion_Sistolica",
    "Presion_Diastolica", "Gender_cod", "Occupation_cod", "BMI Category_cod",
]

X = df[columnas_features]
y = df["Quality of Sleep"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_regresion = joblib.load("modelos/modelo_regresion.pkl")
y_pred = modelo_regresion.predict(X_test)

# ------------------------------------------------------------------
# GRAFICA 1: Valores reales vs valores predichos
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='#8b5cf6', alpha=0.6, edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Predicción perfecta')
plt.xlabel('Calidad de sueño REAL', fontsize=13)
plt.ylabel('Calidad de sueño PREDICHA', fontsize=13)
plt.title('Valores Reales vs Valores Predichos\n(Modelo de Regresión)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafica_real_vs_predicho.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 1 guardada: grafica_real_vs_predicho.png")

# ------------------------------------------------------------------
# GRAFICA 2: Distribucion de errores (residuos)
# ------------------------------------------------------------------
errores = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.hist(errores, bins=20, color='#a855f7', edgecolor='white', linewidth=0.5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0 (perfecto)')
plt.xlabel('Error de predicción (Real - Predicho)', fontsize=13)
plt.ylabel('Cantidad de predicciones', fontsize=13)
plt.title('Distribución de Errores del Modelo\n(Entre más cerca del 0, mejor)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafica_errores.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafica 2 guardada: grafica_errores.png")

print("\nListo! Las 2 graficas se guardaron en la carpeta estres-backend")