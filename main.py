import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SolicitudCredito(BaseModel):
    edad: int
    ingresos_anuales: float
    antiguedad_laboral: float
    monto_solicitado: float
    tasa_interes: float
    porcentaje_ingreso_credito: float
    anios_historial_crediticio: int
    tipo_vivienda_HIPOTECA: int
    tipo_vivienda_PROPIO: int
    tipo_vivienda_OTRO: int
    motivo_credito_EDUCACION: int
    motivo_credito_EMPRENDIMIENTO: int
    motivo_credito_MEJORA_HOGAR: int
    motivo_credito_MEDICO: int
    motivo_credito_PERSONAL: int
    historial_morosidad_SI: int
    calificacion_credito_B: int
    calificacion_credito_C: int
    calificacion_credito_D: int
    calificacion_credito_E: int
    calificacion_credito_F: int
    calificacion_credito_G: int


app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="Servicio web para evaluar el riesgo de default financiero utilizando Machine Learning",
    version="1.0.0",
)

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_actual, "models", "modelo_crediticio.pkl")

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError("El archivo del modelo entrenado no fue encontrado.")

with open(ruta_modelo, "rb") as archivo:
    modelo_crediticio = pickle.load(archivo)


@app.get("/")
def verificar_estado_api():
    return {
        "estado": "API Activa",
        "proyecto": "Riesgo Crediticio Explicable con XGBoost",
    }


@app.post("/predecir")
def evaluar_riesgo_credito(solicitud: SolicitudCredito):
    try:
        datos_cliente = pd.DataFrame([solicitud.model_dump()])

        columnas_modelo = modelo_crediticio.feature_names_in_
        datos_cliente = datos_cliente[columnas_modelo]

        probabilidades = modelo_crediticio.predict_proba(datos_cliente)
        probabilidad_default = probabilidades[0][1]

        decision_prediccion = modelo_crediticio.predict(datos_cliente)
        dictamen_final = "RECHAZADO" if decision_prediccion[0] == 1 else "APROBADO"

        return {
            "probabilidad_riesgo": round(float(probabilidad_default), 4),
            "dictamen": dictamen_final,
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

