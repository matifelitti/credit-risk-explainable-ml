# 📊 Sistema Predictivo de Riesgo Crediticio con Inteligencia Artificial Explicable (XGBoost + SHAP)

## 🎯 Visión General del Proyecto
Este proyecto desarrolla un pipeline de Machine Learning de extremo a extremo para predecir el riesgo de default financiero (incumplimiento de pago) en solicitantes de crédito. 

A diferencia de los modelos tradicionales de "caja negra", este sistema integra **Inteligencia Artificial Explicable (XAI)** utilizando valores **SHAP**. Esto permite desglosar con precisión matemática el impacto de cada variable (ingresos, nivel de endeudamiento, historial) en la decisión final. El enfoque garantiza el cumplimiento de las regulaciones de transparencia financiera y el derecho a la explicación del consumidor en el sector Fintech y bancario.

---

## 🛠️ Logros Técnicos Clave
* **Pipeline de Datos en Español**: Transformación, limpieza y traducción automatizada de variables financieras críticas desde fuentes de datos brutos.
* **Tratamiento de Anomalías y Nulos**: Robustez en la ingesta mediante la eliminación de valores atípicos y la imputación estadística de datos faltantes mediante medianas sectoriales.
* **Modelado Predictivo de Alto Rendimiento**: Entrenamiento de un clasificador binario avanzado utilizando **XGBoost**, optimizado para manejar datos altamente desbalanceados.
* **Métricas de Éxito de Negocio**: El modelo alcanza un **AUC-ROC Score de 0.9482** y un **F1-Score de 0.81** en la detección de morosidad, superando los estándares exigidos en entornos productivos financieros.
* **Transparencia Normativa (XAI)**: Implementación de la librería **SHAP** para generar explicaciones globales del modelo y auditorías visuales e individuales por cliente (Gráficos de Cascada).

---

## 💻 Stack Tecnológico
* **Lenguaje de Programación**: Python
* **Ingeniería de Datos**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost
* **Explicabilidad de IA (XAI)**: SHAP, Matplotlib
* **Despliegue (En Desarrollo)**: FastAPI

---

## 📂 Estructura del Repositorio
```text
credit-risk-explainable-ml/
│
├── data/                    
│   ├── raw/                 # Dataset financiero original (Excluido de Git)
│   └── processed/           # Conjuntos de entrenamiento y prueba en español (Excluido de Git)
│
├── models/                  # Archivo binario serializado del modelo entrenado (Excluido de Git)
│
├── plots/                   # Gráficos de auditoría y explicabilidad SHAP (Excluido de Git)
│
├── src/                     # Código fuente del proyecto
│   ├── data_processor.py    # Ingesta, limpieza y codificación de variables
│   ├── model_trainer.py     # Entrenamiento de XGBoost y cálculo de métricas (AUC-ROC)
│   └── model_explainer.py   # Pipeline de explicabilidad matemática con SHAP
│
├── main.py                  # Estructura base para el despliegue de la API (FastAPI)
└── .gitignore               # Configuración de exclusión de archivos pesados y locales
```

---

## 🚀 Instalación y Uso

Siga estos pasos para clonar el proyecto, configurar el entorno local y ejecutar el pipeline completo.

### 1. Clonar el repositorio
Abra una terminal y clone este proyecto en su máquina local:
```bash
git clone https://github.com/matifelitti/credit-risk-explainable-ml
cd credit-risk-explainable-ml
```

### 2. Configurar el entorno virtual
Se recomienda usar un entorno virtual para mantener aisladas las dependencias del proyecto.

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias
Instale todas las librerías necesarias con el gestor de paquetes de Python (`pip`):
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline del proyecto
El código está modulado para ejecutarse de manera secuencial desde la raíz del proyecto. Ejecute los scripts en el siguiente orden:

*   **Paso 1: Procesar los datos** (Limpia y prepara el dataset bruto)
    ```bash
    python src/data_processor.py
    ```
*   **Paso 2: Entrenar el modelo** (Genera las métricas AUC-ROC y guarda el modelo entrenado)
    ```bash
    python src/model_trainer.py
    ```
*   **Paso 3: Generar explicabilidad con SHAP** (Crea y exporta los gráficos de auditoría visual)
    ```bash
    python src/model_explainer.py
    ```

---

## 📈 Métricas de Rendimiento Obtenidas
El modelo fue evaluado utilizando un conjunto de prueba independiente y arrojó los siguientes resultados en la consola:

* **Precisión Global (Accuracy)**: 92%
* **Capacidad de Discriminación (AUC-ROC)**: 0.9482
* **Matriz de Rendimiento**:
  * Clientes Cumplidores (Clase 0): F1-Score del 95%
  * Clientes en Riesgo (Clase 1): F1-Score del 81% (Recall del 80%)
