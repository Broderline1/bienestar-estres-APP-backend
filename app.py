from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

modelo_clasificacion = joblib.load("modelos/modelo_clasificacion.pkl")
modelo_regresion = joblib.load("modelos/modelo_regresion.pkl")
codificadores = joblib.load("modelos/codificadores.pkl")
columnas_features = joblib.load("modelos/columnas_features.pkl")

# Valores promedio del dataset para los campos que quitamos del formulario
RITMO_CARDIACO_PROMEDIO = 70
PRESION_SISTOLICA_PROMEDIO = 128
PRESION_DIASTOLICA_PROMEDIO = 85

def armar_mensaje(nivel_estres, calidad_predicha, horas_sueno_actuales):
    if nivel_estres == "Alto":
        return (
            f"Estas en un nivel de estres ALTO. "
            f"Con tus habitos actuales, tu calidad de sueño estimada es de {calidad_predicha:.1f}/10. "
            f"Te recomendamos hacer ajustes pronto, como aumentar tus horas de sueño y bajarle a las horas de pantalla."
        )
    elif nivel_estres == "Medio":
        return (
            f"Tu nivel de estres es MEDIO, vas en un punto donde hay que tener cuidado. "
            f"Tu calidad de sueño estimada es de {calidad_predicha:.1f}/10. "
            f"Pequeños cambios en tu rutina de sueño pueden ayudarte a mejorar."
        )
    else:
        return (
            f"Tu nivel de estres es BAJO, vas bien. "
            f"Tu calidad de sueño estimada es de {calidad_predicha:.1f}/10. Sigue asi."
        )

@app.route("/", methods=["GET"])
def inicio():
    return jsonify({"mensaje": "API de bienestar/estres funcionando correctamente"})

@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json()

    try:
        genero_cod = codificadores["Gender"].transform([datos["genero"]])[0]
        ocupacion_cod = codificadores["Occupation"].transform([datos["ocupacion"]])[0]
        bmi_cod = codificadores["BMI Category"].transform([datos["bmi_categoria"]])[0]

        # Los 3 campos que quitamos del formulario los rellenamos con el promedio
        fila = pd.DataFrame([{
            "Age": datos["edad"],
            "Sleep Duration": datos["horas_sueno"],
            "Physical Activity Level": datos["actividad_fisica"],
            "Heart Rate": RITMO_CARDIACO_PROMEDIO,
            "Daily Steps": datos["pasos_diarios"],
            "Presion_Sistolica": PRESION_SISTOLICA_PROMEDIO,
            "Presion_Diastolica": PRESION_DIASTOLICA_PROMEDIO,
            "Gender_cod": genero_cod,
            "Occupation_cod": ocupacion_cod,
            "BMI Category_cod": bmi_cod,
        }])[columnas_features]

        nivel_estres = modelo_clasificacion.predict(fila)[0]
        calidad_predicha = modelo_regresion.predict(fila)[0]

        mensaje = armar_mensaje(nivel_estres, calidad_predicha, datos["horas_sueno"])

        return jsonify({
            "nivel_estres": nivel_estres,
            "calidad_sueno_predicha": round(float(calidad_predicha), 1),
            "mensaje": mensaje
        })

    except KeyError as e:
        return jsonify({"error": f"Falta el campo: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)