# PetFinderLGBM - Adoption Speed Classifier

**Autor:** Cristian Arenas  
**Lenguajes:** C++ y Python  
**Modelo:** LightGBM (ejecución nativa)  
**Base de datos:** SQLite3  
**Sistema:** Windows 64-bit

---

## Descripción del proyecto

Este proyecto implementa un pipeline completo para el entrenamiento y evaluación de un modelo de clasificación multiclase que predice la velocidad de adopción de mascotas (`AdoptionSpeed`). Utiliza LightGBM en su versión nativa, con un aplicación en C++ para control total del proceso y scripts en Python para análisis visual de resultados.
El modelo es entrenado usando validación cruzada K-Fold (5 folds) y también se entrena un modelo final con todo el dataset. Los resultados se almacenan en SQLite y se visualizan mediante gráficos generados con Python.
Dataset (train.csv) utilizado en el pipeline descargado desde Kaggle: https://www.kaggle.com/competitions/petfinder-adoption-prediction/data.

---

## Requisitos

- Windows 10/11 64-bit
- Python 3.x (para visualización)
- Paquetes Python:  
  `pip install matplotlib seaborn pandas`
- LightGBM nativo (`lightgbm.exe`) --- ya se encuentra junto con los archivos del proyecto - web LightGBM: https://lightgbm.readthedocs.io/en/stable/index.html

---

## Cómo ejecutar

Clonar el proyecto localmente desde tu máquina.

Dentro del directorio raíz vas a encontrar todo el código fuente por si quieres realizar modificaciones o mejoras que consideres necesarias.
Si lo deseas, puedes abrir el proyecto desde algún entorno de desarrollo (en mi caso usé Visual Studio 2022), compilar y ejecutarlo desde ahí mismo.
Si sólo quieres hacer uso del pipeline sin realizar modificaciones, en el directorio raíz del proyecto, encontrarás un archivo zip llamado "PetFinderLGBMReady" 
con todo listo para correr. Descomprimirlo.

En la terminal de Windows (cmd o PowerShell):

Desde el directorio raíz del proyecto:

```bash
cd PetFinderLGBMReady
run_pipeline.bat
```

Al correr el archivo run_pipeline.bat, se inicia el pipeline y se ejecuta como ***PRIMER INSTANCIA*** un script Python llamado "optuna_files_creator.py" que se encarga de correr 
la optimización bayesiana de hiperparámetros mediante la librería Optuna. Puedes usar este script para modificarlo y especificar tu propia configuración, realizar tu propia 
feature engineering con el dataset, cambiar hiperparámetros, semilla a utilizar, etc.

Qué hace este script?

 -Carga y transforma los datos.

 -Realiza feature engineering con nuevas variables como CareLevel, ColorCount, HasPhotoVideo, FeeBin, etc.

 -Define un conjunto de variables explicativas (features) y la variable respuesta (AdoptionSpeed).

 -Divide los datos para optimización.

 -Separa los datos en entrenamiento y validación (80/20) estratificadamente para mantener la distribución de clases.

 -Define un espacio de búsqueda de hiperparámetros.

 -Usa Optuna para explorar automáticamente combinaciones de hiperparámetros de LightGBM como:

	*learning_rate

	*num_leaves

	*min_data_in_leaf

	*feature_fraction

	*bagging_fraction

	*lambda_l1, lambda_l2, etc.

 -Evalúa cada combinación usando Cross-Validation (CV).

 -Por cada combinación de hiperparámetros, se entrena un modelo LightGBM en 3 folds.

 -Se utiliza la métrica Cohen's Kappa como objetivo de maximización, que mide la concordancia entre predicción y realidad.

 -Guarda la mejor configuración.

 -Imprime los mejores hiperparámetros encontrados.

 -Utiliza estos parámetros para generar los archivos de entrenamiento y predicción.

 -Genera archivos para utilizar luego en la aplicación C++ de entrenamiento y análisis.

 -Crea archivos .txt con los datos de entrenamiento y validación por fold.

 -Genera archivos de configuración (config_train_fold_*.txt, config_pred_fold_*.txt, etc.).

 -También crea el dataset completo train_all.txt y su archivo de configuración config_train_all.txt para utilizar en el entrenamiento final con todos los datos.

Una vez que el script Python terminó de correr, pasamos a la ***SEGUNDA INSTANCIA*** donde se inicia el programa en C++ para el entrenamiento, evaluación y análisis 
de los modelos LGBM para predecir la velocidad con la que se adoptan las mascotas en el dataset PetFinder. Se utiliza LGBM nativo para entrenar cada una de las 5 folds y 
el modelo final con todos los datos.

Descripción general de este programa:

 -Modelado con LightGBM nativo.

 -Almacenamiento y trazabilidad con SQLite3.

 -Visualizaciones con Python.

 -Reportes y métricas avanzadas.

 
 Validación Cruzada (K-Fold)

 -Para cada uno de los 5 folds:

  -Entrena un modelo LightGBM usando el archivo config_train_fold_i.txt.

  -Predice sobre el fold de validación con config_pred_fold_i.txt.

  -Calcula las métricas:

   -Accuracy.

   -F1 macro.

   -Cohen’s Kappa.

 -Guarda:

  -Resultados y predicciones en archivos .csv.

  -Matriz de confusión (como imagen y CSV).

  -Métricas en la base de datos resultados.db (tabla resultados).

  -Predicciones en la tabla predicciones.

 -Genera visualizaciones con scripts Python (ejecuta automáticamente los scripts de Python):
  
  -Matríz de confusión por fold y final.

  -Evolución de métricas durante experimentos.
  
  -Importancia de variables.

 -Resumen Promedio:
  
  -Muestra y almacena métricas promedio de todos los folds.
 
Entrenamiento Final sobre todo el Dataset:

 -Entrena utilizando todos los datos con el mejor modelo.
 
 -Realiza predicciones.

 -Calcula métricas finales (Accuracy, F1 y Kappa).

 -Genera y guarda matriz de confusión.
 
 -Almacena resultados en SQLite como modelo final.

 -Visualiza gráficos para análisis.


***Como resultado se genera un modelo binario para llevar a producción***

---

## Métricas utilizadas

- **Accuracy**
- **F1 Macro**
- **Cohen's Kappa**
- **Matriz de confusión**
- **Importancia de variables**

---

## Visualizaciones

Se generan automáticamente:

- `metricas_por_fold.png` → gráfico de barras comparativo
- `evolucion_metricas.png` → evolución temporal
- `ranking_f1_macro.png` → ranking de experimentos
- `conf_matrix_fold_*.png` → matrices de confusión
- `importancia_variables.png` → importancia de variables

---

## Base de datos SQLite

El archivo `resultados.db` guarda:

- Métricas por experimento (accuracy, F1, Kappa)
- Hiperparámetros utilizados
- Predicciones reales y estimadas

Esto permite trazabilidad, auditoría y reanálisis.

---

## Ventajas de esta implementación

- Máximo rendimiento gracias al uso de **LightGBM nativo en C++**
- Código modular y reutilizable
- Visualización clara con Python + Matplotlib
- Resultados persistentes en **SQLite**
- Ideal para **entornos productivos** o **análisis robustos** en competencias


