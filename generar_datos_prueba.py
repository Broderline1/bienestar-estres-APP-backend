"""
Este script SOLO es para que puedas probar el proyecto completo
ANTES de bajar el CSV real de Kaggle.

Genera un archivo con las mismas columnas que trae el dataset
"Sleep Health and Lifestyle Dataset" de Kaggle, pero con datos
inventados al azar.

Cuando ya tengas el CSV real, bórralo y pon el de verdad en su lugar
con el mismo nombre: data/sleep_health_and_lifestyle.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 374  # el dataset real trae 374 personas, usamos el mismo tamaño

generos = np.random.choice(["Male", "Female"], n)
ocupaciones = np.random.choice(
    ["Doctor", "Engineer", "Teacher", "Nurse", "Lawyer", "Accountant", "Salesperson"], n
)
bmi = np.random.choice(["Normal", "Overweight", "Obese"], n, p=[0.5, 0.35, 0.15])
trastorno = np.random.choice(["None", "Insomnia", "Sleep Apnea"], n, p=[0.6, 0.2, 0.2])

df = pd.DataFrame({
    "Person ID": range(1, n + 1),
    "Gender": generos,
    "Age": np.random.randint(20, 60, n),
    "Occupation": ocupaciones,
    "Sleep Duration": np.round(np.random.uniform(4.5, 9, n), 1),
    "Quality of Sleep": np.random.randint(3, 10, n),
    "Physical Activity Level": np.random.randint(20, 90, n),
    "Stress Level": np.random.randint(1, 10, n),
    "BMI Category": bmi,
    "Blood Pressure": [f"{np.random.randint(110,140)}/{np.random.randint(70,90)}" for _ in range(n)],
    "Heart Rate": np.random.randint(60, 90, n),
    "Daily Steps": np.random.randint(3000, 12000, n),
    "Sleep Disorder": trastorno,
})

df.to_csv("data/sleep_health_and_lifestyle.csv", index=False)
print(f"Listo! Se generaron {n} registros de prueba en data/sleep_health_and_lifestyle.csv")
print(df.head())
