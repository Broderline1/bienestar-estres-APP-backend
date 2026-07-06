"""
Este script hace 4 cosas, en orden:
1. Carga y limpia los datos
2. Prepara las columnas (convierte texto a números)
3. Entrena un modelo de CLASIFICACION (nivel de estrés: Bajo/Medio/Alto)
4. Entrena un modelo de REGRESION (predice la calidad de sueño, un número)
5. Guarda ambos modelos en archivos .pkl para usarlos despues en la API

Corre esto con: python entrenar.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

# ------------------------------------------------------------------
# PASO 1: Cargar los datos
# ------------------------------------------------------------------
df = pd.read_csv("data/sleep_health_and_lifestyle_dataset.csv")

# Arreglamos los huecos: si no tiene trastorno de sueño, lo marcamos como "None"
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# Unificamos "Normal Weight" y "Normal" porque son lo mismo
df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")

# La columna "Blood Pressure" viene como texto tipo "120/80"
# la partimos en dos columnas numericas: sistolica y diastolica
df[["Presion_Sistolica", "Presion_Diastolica"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)


# ------------------------------------------------------------------
# PASO 2: Crear la columna de CLASIFICACION (target 1)
# ------------------------------------------------------------------
# Convertimos el "Stress Level" (numero del 1 al 10) en 3 categorias.
# Esto se llama "binning" o "agrupar en rangos".
def categorizar_estres(valor):
    if valor <= 4:
        return "Bajo"
    elif valor <= 6:
        return "Medio"
    else:
        return "Alto"

df["Nivel_Estres_Categoria"] = df["Stress Level"].apply(categorizar_estres)

print("Distribucion de las categorias de estres:")
print(df["Nivel_Estres_Categoria"].value_counts())

# ------------------------------------------------------------------
# PASO 3: Convertir columnas de texto a numeros
# ------------------------------------------------------------------
# Los modelos NO entienden texto, solo numeros. Por eso "codificamos"
# cada columna de texto. Guardamos cada "codificador" porque despues
# la API los va a necesitar para traducir lo que mande Angular.
codificadores = {}
columnas_categoricas = ["Gender", "Occupation", "BMI Category"]

for col in columnas_categoricas:
    le = LabelEncoder()
    df[col + "_cod"] = le.fit_transform(df[col])
    codificadores[col] = le  # lo guardamos para usarlo despues

# ------------------------------------------------------------------
# PASO 4: Elegir las columnas que va a usar el modelo (features)
# ------------------------------------------------------------------
columnas_features = [
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Heart Rate",
    "Daily Steps",
    "Presion_Sistolica",
    "Presion_Diastolica",
    "Gender_cod",
    "Occupation_cod",
    "BMI Category_cod",
]

X = df[columnas_features]

# ------------------------------------------------------------------
# PASO 5: Entrenar el modelo de CLASIFICACION
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("ENTRENANDO MODELO DE CLASIFICACION (Nivel de Estres)")
print("=" * 50)

y_clasificacion = df["Nivel_Estres_Categoria"]

# Separamos 80% para entrenar y 20% para probar (el "examen sorpresa")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clasificacion, test_size=0.2, random_state=42
)

modelo_clasificacion = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_clasificacion.fit(X_train, y_train)

# Probamos que tan bien le fue
predicciones = modelo_clasificacion.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"Precision del modelo (accuracy): {precision:.2%}")
print("\nReporte detallado:")
print(classification_report(y_test, predicciones))

# ------------------------------------------------------------------
# PASO 6: Entrenar el modelo de REGRESION
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("ENTRENANDO MODELO DE REGRESION (Calidad de Sueño)")
print("=" * 50)

y_regresion = df["Quality of Sleep"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_regresion, test_size=0.2, random_state=42
)

modelo_regresion = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_regresion.fit(X_train_r, y_train_r)

predicciones_r = modelo_regresion.predict(X_test_r)
error_promedio = mean_absolute_error(y_test_r, predicciones_r)
print(f"Error promedio (MAE): {error_promedio:.2f} puntos de calidad de sueño")
print("(Esto significa que en promedio, el modelo se equivoca por esta cantidad)")

# ------------------------------------------------------------------
# PASO 7: Guardar TODO lo que la API va a necesitar despues
# ------------------------------------------------------------------
joblib.dump(modelo_clasificacion, "modelos/modelo_clasificacion.pkl")
joblib.dump(modelo_regresion, "modelos/modelo_regresion.pkl")
joblib.dump(codificadores, "modelos/codificadores.pkl")
joblib.dump(columnas_features, "modelos/columnas_features.pkl")

print("\n" + "=" * 50)
print("LISTO! Modelos guardados en la carpeta modelos/")
print("=" * 50)
