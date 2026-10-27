# Taller 2: Predicción de Pobreza

Este repositorio contiene el flujo de trabajo del Taller 2. El proyecto parte de
la GEIH (Gran Encuesta Integrada de Hogares), construye una variable de ingreso
potencial por hogar y entrena distintos modelos de clasificación para predecir
si un hogar es pobre (`Pobre = 1(Ingpcug < Lp)`), evaluados con la métrica F1 de
la clase pobre.

El objetivo de esta organización es facilitar la replicabilidad del análisis:
separar datos, código y resultados; documentar el orden de ejecución; y permitir
que otra persona entienda qué insumo produce cada etapa.

Big Data y Machine Learning — Universidad de los Andes, 2026-2. Grupo 05.

## Autores

- Laura Alejandra Puerto Amaya — *202312335*
- Miguel Hernando Torres *2023*
- Angel Esteban Bonilla Escamilla *202424426*

## Estructura del proyecto

```text
Taller_02_grupo_05/
├── 01_data/
│   ├── 01_raw/
│   ├── 02_clean/
│   └── 03_features/
├── 02_code/
├── 03_outputs/
├── 04_documentation/
├── main.R
└── README.md
```

## Carpetas

### `01_data/`

Contiene los datos usados y producidos por el taller.

- `01_data/01_raw/`: las 4 bases originales de la competencia de Kaggle del curso (`train_hogares.csv`, `test_hogares.csv`, `train_personas.csv`, `test_personas.csv`) y `sample_submission.csv`. `train_personas.csv` (~240 MB) se versiona con Git LFS; después de clonar el repositorio se corre `git lfs pull` para traer el archivo real en vez de su puntero.
- `01_data/02_clean/`: `train_clean.csv` / `test_clean.csv`, salida de `01_cleaning.R` — hogares y personas agregados por hogar, con tipos definidos y sin valores faltantes.
- `01_data/03_features/`: `db_train.rds`, `db_test.rds`, `folds.rds`, salida de `02_ingreso.R` — las mismas bases de `02_clean/` ya con la variable de ingreso potencial incorporada, y los folds de validación cruzada compartidos por todos los modelos.

### `02_code/`

Contiene los scripts de R del proyecto, en el orden en que se ejecutan.

- `00_set_up.R`: carga los paquetes del proyecto (vía `pacman`, instala los que falten) y resuelve conflictos de nombres entre paquetes (p. ej. `dplyr::select` vs. `MASS::select`).
- `01_cleaning.R`: importa las 4 bases crudas, limpia hogares y personas (vivienda, arriendo, educación, condiciones laborales, protección social), agrega personas por hogar, y exporta `train_clean.csv` / `test_clean.csv` a `01_data/02_clean/`.
- `02_ingreso.R`: construye la variable de ingreso potencial en dos etapas: (1) modelo de dos partes con XGBoost para el ingreso individual — P(ingreso > 0) × E[ingreso \| ingreso > 0] —, estimado fuera de muestra con 5 folds; (2) agregación al hogar replicando la fórmula del DANE (unidad de gasto, arriendo imputado). Guarda `db_train`, `db_test` y `folds` en `01_data/03_features/`.
- `03_modelos.R`: entrena y evalúa 27 modelos de clasificación — Elastic Net, LDA, QDA, Naive Bayes, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost y LightGBM — con selección de hiperparámetros por ROC o por F1 (umbral calibrado en la curva precisión-recall) y variantes con rebalanceo por oversampling. Exporta un `.csv` de envío a Kaggle por modelo.

### `04_documentation/`

- `Problem_Set_2.pdf`: enunciado del taller.
- `ddi-documentation-spanish-608.pdf`: diccionario de variables del DANE (Pobreza Monetaria 2018).

## Flujo de trabajo

El pipeline general del taller es:

1. Cargar los paquetes del proyecto y resolver conflictos de nombres (`00_set_up.R`).
2. Limpiar las 4 bases crudas y construir la base de hogares con sus variables (`01_cleaning.R`).
3. Estimar el ingreso potencial por persona fuera de muestra y agregarlo al hogar (`02_ingreso.R`).
4. Entrenar y evaluar los modelos de clasificación, calibrar el umbral de cada uno y exportar los envíos a Kaggle (`03_modelos.R`).

## Datos

La fuente es la GEIH, distribuida para este taller a través de la competencia de
Kaggle del curso (enunciado completo en `04_documentation/Problem_Set_2.pdf`).

La base cruda se guarda en:

```text
01_data/01_raw/train_hogares.csv
01_data/01_raw/test_hogares.csv
01_data/01_raw/train_personas.csv
01_data/01_raw/test_personas.csv
```

La base limpia se guarda en:

```text
01_data/02_clean/train_clean.csv
01_data/02_clean/test_clean.csv
```

y la base lista para modelar, con el ingreso potencial y los folds, en:

```text
01_data/03_features/db_train.rds
01_data/03_features/db_test.rds
01_data/03_features/folds.rds
```

## Replicabilidad

Para reproducir todos los resultados, abrir `Taller_02_grupo_05.Rproj` en
RStudio y correr:

```r
source("main.R")
```

`main.R` ubica la raíz del proyecto automáticamente (sin rutas personales),
carga los paquetes necesarios y ejecuta en orden los cuatro scripts de
`02_code/` descritos arriba.

`02_ingreso.R` tarda cerca de 15 minutos. `03_modelos.R` tarda más, porque
entrena los 27 modelos, varios con validación cruzada de 5 folds y búsqueda de
hiperparámetros. Una vez generados, `01_data/03_features/db_train.rds` y
`db_test.rds` permiten correr `03_modelos.R` directamente sin repetir
`02_ingreso.R`.

## Dependencias

Los scripts usan los siguientes paquetes de R, instalados y cargados
automáticamente con `pacman::p_load()` en `00_set_up.R`:

- **Datos y presentación**: `tidyverse`, `readxl`, `writexl`
- **Entrenamiento y modelos**: `caret`, `glmnet`, `MASS`, `naivebayes`, `rpart`, `rpart.plot`, `ranger`, `randomForest`, `adabag`, `gbm`, `xgboost`, `lightgbm`
- **Evaluación**: `pROC`, `MLmetrics`, `Metrics`
- **Organización y replicabilidad**: `rstudioapi`, `conflicted`, `docstring`, `renv`

## Productos principales

`03_modelos.R` exporta un envío a Kaggle por modelo, como un `.csv` nombrado
según el modelo y sus hiperparámetros/umbral, en el directorio de trabajo. Por
ejemplo:

- `RandomForest_F1_prob_<umbral>.csv`
- `XGBoost_F1_prob_<umbral>.csv`
- `LightGBM_F1_prob_<umbral>.csv`
- `LDA.csv`, `QDA.csv`, `NaiveBayes.csv`, `AdaBoost.csv`, `Bagging.csv`

## Software

- R 4.5.2

## Reglas del repositorio

- Mantener los datos crudos en `01_data/01_raw/` y no modificarlos manualmente.
- Guardar las bases procesadas en `01_data/02_clean/` y los insumos listos para modelar en `01_data/03_features/`.
- Mantener todo el código reproducible en `02_code/`, en el orden descrito arriba.
