"""
Este archivo NO entrena nada. Solo sirve para "asomarnos" a los datos
y ver qué tenemos, antes de construir los modelos.

Corre esto con: python3 explorar.py
"""
import pandas as pd

# Cargamos el CSV (como si abrieras un Excel, pero en código)
df = pd.read_csv("data/sleep_health_and_lifestyle.csv")

print("=" * 50)
print("1) Las primeras 5 filas (para ver cómo se ve la tabla)")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("2) Cuántas filas y columnas tenemos")
print("=" * 50)
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

print("\n" + "=" * 50)
print("3) Nombres de columnas y tipo de dato de cada una")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("4) Buscamos datos vacíos (huecos que hay que arreglar)")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("5) Estadísticas básicas de las columnas numéricas")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("6) Valores únicos en columnas de texto (para saber qué categorías hay)")
print("=" * 50)
for col in ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]:
    print(f"\n{col}: {df[col].unique()}")
