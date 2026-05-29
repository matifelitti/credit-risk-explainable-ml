from fastapi import FastAPI

app = FastAPI(
    title="API de Predicción de Riesgo Crediticio",
    description="Servicio web para evaluar el riesgo de default financiero utilizando Machine Learning",
    version="1.0.0",
)


@app.get("/")
def leer_raiz():
    return {"estado": "API Activa", "proyecto": "Riesgo Crediticio Explicable"}
