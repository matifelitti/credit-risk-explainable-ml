import os
import pickle
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate


class AuditorEquidadModelo:

    def __init__(self, ruta_modelo: str, ruta_datos_x: str, ruta_datos_y: str):
        self.ruta_modelo = ruta_modelo
        self.ruta_datos_x = ruta_datos_x
        self.ruta_datos_y = ruta_datos_y
        self.modelo = None
        self.X_prueba = None
        self.y_prueba = None

    def cargar_recursos(self):
        with open(self.ruta_modelo, "rb") as archivo:
            self.modelo = pickle.load(archivo)
        self.X_prueba = pd.read_csv(self.ruta_datos_x)
        self.y_prueba = pd.read_csv(self.ruta_datos_y).values.ravel()

    def auditar_sesgo_edad(self, umbral_edad_joven: int = 25):
        predicciones = self.modelo.predict(self.X_prueba)

        grupo_edad = self.X_prueba["edad"].apply(
            lambda x: "Joven" if x <= umbral_edad_joven else "Estandar"
        )

        marco_metricas = MetricFrame(
            metrics=selection_rate,
            y_true=self.y_prueba,
            y_pred=predicciones,
            sensitive_features=grupo_edad,
        )

        tasa_seleccion_por_grupo = marco_metricas.by_group
        tasa_joven = tasa_seleccion_por_grupo["Joven"]
        tasa_estandar = tasa_seleccion_por_grupo["Estandar"]

        impacto_dispar = (
            tasa_joven / tasa_estandar if tasa_estandar > 0 else 0
        )

        print("--- REPORTE DE AUDITORÍA DE EQUIDAD (SESGO POR EDAD) ---")
        print(f"Tasa de aprobación - Clientes Jóvenes: {tasa_joven:.4f}")
        print(f"Tasa de aprobación - Clientes Estándar: {tasa_estandar:.4f}")
        print(f"Índice de Impacto Dispar: {impacto_dispar:.4f}")

        if impacto_dispar < 0.80:
            print("Resultado: Alerta de Sesgo. El modelo discrimina a jóvenes.")
        else:
            print("Resultado: Cumplimiento Normativo Exitoso. Modelo Equitativo.")


if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    archivo_modelo = os.path.join(ruta_raiz, "models", "modelo_crediticio.pkl")
    archivo_X = os.path.join(ruta_raiz, "data", "processed", "X_prueba.csv")
    archivo_y = os.path.join(ruta_raiz, "data", "processed", "y_prueba.csv")

    auditor = AuditorEquidadModelo(archivo_modelo, archivo_X, archivo_y)
    auditor.cargar_recursos()
    auditor.auditar_sesgo_edad()
