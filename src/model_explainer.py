import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap


class ExplicadorModeloCredito:

    def __init__(self, ruta_modelo: str, ruta_datos: str, ruta_salida_graficos: str):
        self.ruta_modelo = ruta_modelo
        self.ruta_datos = ruta_datos
        self.ruta_salida_graficos = ruta_salida_graficos
        self.modelo = None
        self.X_prueba = None
        self.explicador = None
        self.valores_shap = None

    def cargar_recursos(self):
        with open(self.ruta_modelo, "rb") as archivo:
            self.modelo = pickle.load(archivo)
        self.X_prueba = pd.read_csv(self.ruta_datos)

    def inicializar_explicador(self):
        self.explicador = shap.TreeExplainer(self.modelo)
        self.valores_shap = self.explicador(self.X_prueba)

    def exportar_analisis_global(self):
        os.makedirs(self.ruta_salida_graficos, exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.valores_shap, self.X_prueba, show=False, plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.ruta_salida_graficos, "importancia_global_variables.png")
        )
        plt.close()


    def exportar_analisis_individual(self, indice_cliente: int = 0):
        os.makedirs(self.ruta_salida_graficos, exist_ok=True)
        plt.figure(figsize=(12, 4))
        shap.plots.waterfall(self.valores_shap[indice_cliente], show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.ruta_salida_graficos, f"explicacion_cliente_{indice_cliente}.png"
            )
        )
        plt.close()


if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    archivo_modelo = os.path.join(ruta_raiz, "models", "modelo_crediticio.pkl")
    archivo_datos = os.path.join(ruta_raiz, "data", "processed", "X_prueba.csv")
    carpeta_graficos = os.path.join(ruta_raiz, "plots")

    analista = ExplicadorModeloCredito(
        archivo_modelo, archivo_datos, carpeta_graficos
    )
    analista.cargar_recursos()
    analista.inicializar_explicador()
    analista.exportar_analisis_global()
    analista.exportar_analisis_individual(indice_cliente=0)
