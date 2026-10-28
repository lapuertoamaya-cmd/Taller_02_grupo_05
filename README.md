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
- Miguel Hernando Torres — *202315331*
- Angel Esteban Bonilla Escamilla — *202424426*

## Estructura del proyecto

```text
Taller_02_grupo_05/
├── 01_data/
│   ├── 01_raw/
│   ├── 02_clean/
│   └── 03_features/
├── 02_code/
├── 03_outputs/
│   ├── descriptivas/
│   │   ├── tables/
│   │   └── figures/
│   └── entrega_final/
├── 04_documentation/
├── main.R
└── README.md
```

## Carpetas

### `01_data/`

Contiene los datos usados y producidos por el taller.

- `01_data/01_raw/`: las 4 bases originales de la competencia de Kaggle del curso (`train_hogares.csv`, `test_hogares.csv`, `train_personas.csv`, `test_personas.csv`) y `sample_submission.csv`. `train_personas.csv` (~240 MB) se versiona con Git LFS; después de clonar el repositorio se corre `git lfs pull` para traer el archivo real en vez de su puntero.
- `01_data/02_clean/`: `train_clean.csv` / `test_clean.csv`, salida de `01_cleaning.R` — hogares y personas agregados por hogar, con tipos definidos y sin valores faltantes.
- `01_data/03_features/`: `db_train.rds`, `db_test.rds`, `folds.rds`, salida de `02_creation_income_var.R` — las mismas bases de `02_clean/` ya con la variable de ingreso potencial incorporada, y los folds de validación cruzada compartidos por todos los modelos.

### `02_code/`

Contiene los scripts de R del proyecto, en el orden en que se ejecutan.

- `00_set_up.R`: carga los paquetes del proyecto (vía `pacman`, instala los que falten) y resuelve conflictos de nombres entre paquetes (p. ej. `dplyr::select` vs. `MASS::select`).
- `01_cleaning.R`: importa las 4 bases crudas, limpia hogares y personas (vivienda, arriendo, educación, condiciones laborales, protección social), agrega personas por hogar, y exporta `train_clean.csv` / `test_clean.csv` a `01_data/02_clean/`.
- `02_creation_income_var.R`: construye la variable de ingreso potencial en dos etapas: (1) modelo de dos partes con XGBoost para el ingreso individual — P(ingreso > 0) × E[ingreso \| ingreso > 0] —, estimado fuera de muestra con 5 folds; (2) agregación al hogar replicando la fórmula del DANE (unidad de gasto, arriendo imputado). Guarda `db_train`, `db_test` y `folds` en `01_data/03_features/`.
- `03_descriptive_analysis.R`: revisa dimensiones de las cuatro bases, desbalance de pobreza, distribuciones del ingreso observado y predicho, y correlaciones simples. Guarda tres tablas CSV y dos gráficas PDF en `03_outputs/descriptivas/`. Usa nombres descriptivos del diccionario y excluye `id` de las correlaciones.
- `04_modelos_1_27.R`: entrena los modelos 01-27 y exporta sus parametros, metricas de CV, matrices de confusion, importancia disponible y predicciones.
- `05_modelos_28_33.R`: entrena seis bloques adicionales desde las bases originales, selecciona sus ganadores y exporta los modelos 28-33.

### `04_documentation/`

- `Problem_Set_2.pdf`: enunciado del taller.
- `ddi-documentation-spanish-608.pdf`: diccionario de variables del DANE (Pobreza Monetaria 2018).
- `diccionario_variables.xlsx`: nombres y definiciones de las variables; las descriptivas usan la hoja `Variables hogares` para etiquetar sus resultados.

## Flujo de trabajo

El pipeline general del taller es:

1. Cargar los paquetes del proyecto y resolver conflictos de nombres (`00_set_up.R`).
2. Limpiar las 4 bases crudas y construir la base de hogares con sus variables (`01_cleaning.R`).
3. Estimar el ingreso potencial por persona fuera de muestra y agregarlo al hogar (`02_creation_income_var.R`).
4. Elaborar las descriptivas y gráficas (`03_descriptive_analysis.R`).
5. Entrenar y evaluar los modelos 01-27 (`04_modelos_1_27.R`).
6. Entrenar los seis bloques adicionales y exportar sus ganadores 28-33 (`05_modelos_28_33.R`).

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
carga los paquetes y ejecuta `00_set_up.R`, `01_cleaning.R`,
`02_creation_income_var.R`, `03_descriptive_analysis.R`, `04_modelos_1_27.R`
y `05_modelos_28_33.R`. Las descriptivas están activas en el pipeline.

`main.R` también incluye `06_modelos_34_37.R`; ese archivo debe incorporarse
para poder completar esa última llamada, pues todavía no está en `02_code/`.

Los bloques 04 y 05 entrenan y evalúan sus modelos; la ejecución completa puede
tardar muchas horas según el equipo. El 02 reutiliza sus bases preparadas cuando
las entradas no han cambiado. El 05 comparte entre sus modelos 32 y 33 las
matrices de ingreso RF que calcula durante esa misma ejecución.

### Ejecutar únicamente las descriptivas

Con las rutas y paquetes de `main.R` ya definidos y los archivos `db_train.rds`
y `db_test.rds` generados por el 02, ejecutar:

```r
source(file.path(dir_codigo, "03_descriptive_analysis.R"), encoding = "UTF-8")
```

Esta llamada no vuelve a entrenar ingresos ni modelos de clasificación. Crea
las carpetas de salida cuando faltan y actualiza los cinco archivos de descriptivas.

## Dependencias

Los scripts usan los siguientes paquetes de R, instalados y cargados
automáticamente con `p_load()` en `00_set_up.R`:

- **Datos y presentación**: `tidyverse`, `readxl`, `writexl`
- **Entrenamiento y modelos**: `caret`, `glmnet`, `MASS`, `naivebayes`, `rpart`, `rpart.plot`, `ranger`, `randomForest`, `adabag`, `gbm`, `xgboost`, `lightgbm`
- **Evaluación**: `pROC`, `MLmetrics`, `Metrics`
- **Organización y replicabilidad**: `rstudioapi`, `conflicted`, `docstring`, `renv`

## Productos principales

El 03 genera, en el orden del análisis:

```text
03_outputs/descriptivas/
├── tables/
│   ├── 01_dimensiones_bases.csv
│   ├── 02_desbalance_clases.csv
│   └── 03_descriptivas_variables.csv
└── figures/
    ├── 01_distribucion_ingreso_por_persona.pdf
    └── 02_variables_asociadas_pobreza.pdf
```

Las gráficas son vectoriales y permiten ampliar sin perder resolución. La tabla
de descriptivas conserva el código y el nombre legible de cada variable. Las
correlaciones describen asociaciones simples; no son importancia multivariada
ni efectos causales. Las variables constantes no reciben una correlación válida.

Los bloques 04 y 05 alimentan `03_outputs/entrega_final/`:

- `predicciones/`: CSV numerados por modelo, exclusivamente con `id` y `pobre`.
- `registros/`: parámetros, grillas, métricas e importancia disponible por modelo.
- `comparacion_modelos.xlsx`: F1 de CV, corte, precisión, recall y los cuatro conteos de la matriz de confusión, además de parámetros y otras características del ajuste. Los modelos pendientes quedan identificados.
- `importancia_variables.xlsx`: importancia por variable y modelo, indicando su métrica. No comparar directamente escalas diferentes.
- `modelos/` : modelos.

Cada exportación reconstruye los dos Excel desde todos los registros presentes.
Conservar los registros para mantener los resultados anteriores y cerrar los
Excel durante la ejecución para evitar bloqueos de escritura.

Las métricas CV usan predicciones fuera de fold, no predicciones del ajuste final
sobre sus propios datos de entrenamiento. Las especificaciones con distintas
bases o particiones no constituyen una comparación controlada entre bases.

## Software

- La version de R y de los paquetes utilizada en cada ajuste queda registrada en sus resultados.
- La semilla del flujo es 2026; fijar la semilla no bloquea las versiones de los paquetes.

## Reglas del repositorio

- Mantener los datos crudos en `01_data/01_raw/` y no modificarlos manualmente.
- Guardar las bases procesadas en `01_data/02_clean/` y los insumos listos para modelar en `01_data/03_features/`.
- Mantener todo el código reproducible en `02_code/`, en el orden descrito arriba.
