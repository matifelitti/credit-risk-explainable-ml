import os
import pandas as pd
from sklearn.model_selection import train_test_split


class ProcesadorDatosCredito:

    def __init__(self, ruta_datos_brutos: str):
        self.ruta_datos_brutos = ruta_datos_brutos
        self.dataframe = None

    def cargar_y_traducir_datos(self) -> pd.DataFrame:
        self.dataframe = pd.read_csv(self.ruta_datos_brutos)

        diccionario_columnas = {
            "person_age": "edad",
            "person_income": "ingresos_anuales",
            "person_home_ownership": "tipo_vivienda",
            "person_emp_length": "antiguedad_laboral",
            "loan_intent": "motivo_credito",
            "loan_amnt": "monto_solicitado",
            "loan_int_rate": "tasa_interes",
            "loan_status": "estado_credito",
            "loan_percent_income": "porcentaje_ingreso_credito",
            "cb_person_default_on_file": "historial_morosidad",
            "cb_person_cred_hist_length": "anios_historial_crediticio",
        }
        self.dataframe = self.dataframe.rename(columns=diccionario_columnas)

        traduccion_vivienda = {
            "RENT": "ALQUILER",
            "MORTGAGE": "HIPOTECA",
            "OWN": "PROPIO",
            "OTHER": "OTRO",
        }
        self.dataframe["tipo_vivienda"] = self.dataframe[
            "tipo_vivienda"
        ].map(traduccion_vivienda)

        traduccion_motivo = {
            "PERSONAL": "PERSONAL",
            "EDUCATION": "EDUCACION",
            "MEDICAL": "MEDICO",
            "VENTURE": "EMPRENDIMIENTO",
            "HOMEIMPROVEMENT": "MEJORA_HOGAR",
            "DEBTCONSOLIDATION": "CONSOLIDACION_DEUDA",
        }
        self.dataframe["motivo_credito"] = self.dataframe[
            "motivo_credito"
        ].map(traduccion_motivo)

        traduccion_morosidad = {"Y": "SI", "N": "NO"}
        self.dataframe["historial_morosidad"] = self.dataframe[
            "historial_morosidad"
        ].map(traduccion_morosidad)

        return self.dataframe

    def eliminar_valores_atipicos(self) -> pd.DataFrame:
        self.dataframe = self.dataframe[self.dataframe["edad"] <= 100]
        self.dataframe = self.dataframe[
            self.dataframe["antiguedad_laboral"] <= 60
        ]
        return self.dataframe

    def imputar_valores_faltantes(self) -> pd.DataFrame:
        mediana_antiguedad = self.dataframe["antiguedad_laboral"].median()
        self.dataframe["antiguedad_laboral"] = self.dataframe[
            "antiguedad_laboral"
        ].fillna(mediana_antiguedad)

        mediana_interes = self.dataframe["tasa_interes"].median()
        self.dataframe["tasa_interes"] = self.dataframe[
            "tasa_interes"
        ].fillna(mediana_interes)

        return self.dataframe

    def codificar_variables_categoricas(self) -> pd.DataFrame:
        columnas_categoricas = [
            "tipo_vivienda",
            "motivo_credito",
            "historial_morosidad",
        ]
        self.dataframe = pd.get_dummies(
            self.dataframe,
            columns=columnas_categoricas,
            drop_first=True,
            dtype=int,
        )
        return self.dataframe

    def dividir_datos(self, columna_objetivo: str, tamaño_test: float = 0.2):
        X = self.dataframe.drop(columns=[columna_objetivo])
        y = self.dataframe[columna_objetivo]

        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
            train_test_split(
                X, y, test_size=tamaño_test, random_state=42, stratify=y
            )
        )
        return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

    def guardar_datos_procesados(
        self,
        X_entrenamiento,
        X_prueba,
        y_entrenamiento,
        y_prueba,
        directorio_salida: str,
    ):
        os.makedirs(directorio_salida, exist_ok=True)
        X_entrenamiento.to_csv(
            os.path.join(directorio_salida, "X_entrenamiento.csv"), index=False
        )
        X_prueba.to_csv(
            os.path.join(directorio_salida, "X_prueba.csv"), index=False
        )
        y_entrenamiento.to_csv(
            os.path.join(directorio_salida, "y_entrenamiento.csv"), index=False
        )
        y_prueba.to_csv(
            os.path.join(directorio_salida, "y_prueba.csv"), index=False
        )


if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_base = os.path.dirname(ruta_actual)
    ruta_csv = os.path.join(ruta_base, "data", "raw", "credit_risk_dataset.csv")
    ruta_salida = os.path.join(ruta_base, "data", "processed")

    procesador = ProcesadorDatosCredito(ruta_csv)
    procesador.cargar_y_traducir_datos()
    procesador.eliminar_valores_atipicos()
    procesador.imputar_valores_faltantes()
    procesador.codificar_variables_categoricas()

    X_train, X_test, y_train, y_test = procesador.dividir_datos(
        columna_objetivo="estado_credito"
    )
    procesador.guardar_datos_procesados(
        X_train, X_test, y_train, y_test, ruta_salida
    )
