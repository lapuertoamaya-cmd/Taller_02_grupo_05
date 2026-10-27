# Taller 2: punto de entrada. Guardar este archivo en la raiz del proyecto.
# RStudio: abrir Taller_02_grupo_05.Rproj y ejecutar source("main.R").
# Terminal: Rscript /ruta/al/proyecto/main.R
# No se necesitan rutas personales ni objetos de una sesion anterior.

# 1. Preparar el repositorio de paquetes y el acceso opcional a RStudio.
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!require("rstudioapi", quietly = TRUE)) {
  install.packages("rstudioapi")
}
library(rstudioapi)

# 2. Localizar la raiz desde el archivo ejecutado o un proyecto reconocible.
# source() y Rscript tienen prioridad sobre la pestana activa de RStudio.
buscar_raiz <- function(candidatos) {
  for (candidato in unique(candidatos)) {
    if (is.na(candidato) || !nzchar(candidato) || !file.exists(candidato)) next
    ruta <- normalizePath(candidato, winslash = "/", mustWork = TRUE)
    if (!dir.exists(ruta)) ruta <- dirname(ruta)
    repeat {
      if (file.exists(file.path(ruta, "Taller_02_grupo_05.Rproj")) &&
          file.exists(file.path(ruta, "main.R")) &&
          file.exists(file.path(ruta, "02_code", "00_set_up.R"))) return(ruta)
      padre <- dirname(ruta)
      if (identical(padre, ruta)) break
      ruta <- padre
    }
  }
  stop("No se encontro la raiz. Abra el .Rproj o ejecute source() con la ruta de main.R.")
}

archivos_source <- unlist(lapply(sys.frames(), function(marco) {
  if (!is.null(marco$ofile)) marco$ofile else character()
}), use.names = FALSE)
argumentos <- commandArgs(trailingOnly = FALSE)
archivo_terminal <- sub("^--file=", "", argumentos[grepl("^--file=", argumentos)])
candidatos <- c(rev(archivos_source), archivo_terminal)
if (rstudioapi::isAvailable()) {
  candidatos <- c(candidatos, rstudioapi::getActiveProject(),
                  rstudioapi::getSourceEditorContext()$path)
}
root <- buscar_raiz(c(candidatos, getwd()))
setwd(root)

# Rutas compartidas por los scripts; usar file.path() al leer o guardar archivos.
dir_datos_raw <- file.path(root, "01_data", "01_raw")
dir_clean <- file.path(root, "01_data", "02_clean")
dir_features <- file.path(root, "01_data", "03_features")
dir_codigo <- file.path(root, "02_code")
dir_resultados <- file.path(root, "03_outputs")
dir_documentacion <- file.path(root, "04_documentation")

# 3. Cargar los paquetes de todo el flujo: limpieza, construccion de variables
# de ingreso y modelos. pacman instala los que falten.
# Nota: con xgboost 3.x (version actual de CRAN) funcionan la etapa de ingreso
# y el bucle de XGBoost sin caret; method = "xgbTree" de caret requiere
# xgboost 1.7.
if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  # Manipulacion de datos
  tidyverse,    # dplyr, tidyr, readr, tibble, purrr
  conflicted,   # resuelve conflictos de nombres entre paquetes
  Matrix,       # matrices dispersas para la etapa de ingreso

  # Entrenamiento, validacion cruzada y metricas
  caret,        # createFolds, trainControl, train y confusionMatrix
  pROC,         # curvas ROC y PR para calibrar el umbral

  # Algoritmos
  glmnet,       # Elastic Net
  MASS,         # LDA y QDA
  naivebayes,   # Naive Bayes
  ranger,       # Bagging y Random Forest
  adabag,       # AdaBoost.M1
  gbm,          # Gradient Boosting
  xgboost,      # etapa 1 del ingreso y modelo XGBoost
  lightgbm,     # LightGBM
  rpart,        # CART

  # Documentacion de funciones
  docstring     # permite consultar los comentarios #' de las funciones
)
# MASS y dplyr comparten select(); stats y dplyr comparten filter().
conflicted::conflicts_prefer(dplyr::filter, dplyr::select, .quiet = TRUE)

# 4. Fijar la secuencia aleatoria. Los modelos paralelos requeriran semillas
# adicionales por fold/modelo cuando se implemente su entrenamiento.
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(2026)

# 5. Orden explicito de ejecucion. Agregar aqui los siguientes scripts cuando
# existan, cada uno separado por coma. No detectar archivos automaticamente:
# el orden forma parte del analisis.
# 02_ingreso.R entrena la etapa de ingreso (cerca de 15 minutos con 2 nucleos)
# y guarda db_train, db_test y folds en 01_data/03_features/. Para modelar sin
# repetirlo, se puede comentar esa linea: 03_modelos.R lee los RDS.
scripts <- c(
  "02_code/00_set_up.R",
  "02_code/01_cleaning.R",
  "02_code/02_ingreso.R",
  "02_code/03_modelos.R"
)
faltantes <- scripts[!file.exists(file.path(root, scripts))]
if (length(faltantes)) {
  stop("Faltan scripts: ", paste(faltantes, collapse = ", "))
}
for (script in scripts) {
  message("Ejecutando: ", script)
  source(file.path(root, script), local = .GlobalEnv, encoding = "UTF-8")
}
message("Flujo terminado. Raiz: ", root)
# Cuando el entorno este probado, renv permitira guardar versiones exactas.
# No generar ni sobrescribir renv.lock automaticamente en cada ejecucion.
