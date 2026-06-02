import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score


class EntrenadorModeloCredito:

    def __init__(self, directorio_datos: str, directorio_modelo: str):
        self.directorio_datos = directorio_datos
        self.directorio_modelo = directorio_modelo
        self.modelo = None

    def cargar_conjuntos_datos(self):
        X_entrenamiento = pd.read_csv(
            os.path.join(self.directorio_datos, "X_entrenamiento.csv")
        )
        X_prueba = pd.read_csv(
            os.path.join(self.directorio_datos, "X_prueba.csv")
        )
        y_entrenamiento = pd.read_csv(
            os.path.join(self.directorio_datos, "y_entrenamiento.csv")
        )
        y_prueba = pd.read_csv(
            os.path.join(self.directorio_datos, "y_prueba.csv")
        )

        return (
            X_entrenamiento,
            X_prueba,
            y_entrenamiento.values.ravel(),
            y_prueba.values.ravel(),
        )

    def entrenar_xgboost(self, X_entrenamiento, y_entrenamiento):
        proporcion_clases = (len(y_entrenamiento) - sum(y_entrenamiento)) / sum(
            y_entrenamiento
        )

        self.modelo = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=proporcion_clases,
            random_state=42,
            eval_metric="logloss",
        )
        self.modelo.fit(X_entrenamiento, y_entrenamiento)
        return self.modelo

    def evaluar_rendimiento(self, X_prueba, y_prueba):
        predicciones = self.modelo.predict(X_prueba)
        probabilidades = self.modelo.predict_proba(X_prueba)[:, 1]

        reporte = classification_report(y_prueba, predicciones)
        auc_roc = roc_auc_score(y_prueba, probabilidades)

        print(reporte)
        print(f"AUC-ROC Score: {auc_roc:.4f}")

    def guardar_modelo_entrenado(self):
        os.makedirs(self.directorio_modelo, exist_ok=True)
        ruta_archivo = os.path.join(self.directorio_modelo, "modelo_crediticio.pkl")
        with open(ruta_archivo, "wb") as archivo:
            pickle.dump(self.modelo, archivo)


if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_actual)
    directorio_datos_procesados = os.path.join(ruta_raiz, "data", "processed")
    directorio_salida_modelo = os.path.join(ruta_raiz, "models")

    entrenador = EntrenadorModeloCredito(
        directorio_datos_procesados, directorio_salida_modelo
    )
    X_train, X_test, y_train, y_test = entrenador.cargar_conjuntos_datos()

    entrenador.entrenar_xgboost(X_train, y_train)
    entrenador.evaluar_rendimiento(X_test, y_test)
    entrenador.guardar_modelo_entrenado()
