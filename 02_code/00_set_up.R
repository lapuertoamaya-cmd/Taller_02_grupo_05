# ------------------------------------------------------------------------------
# Archivo: 00_set_up.R
# Proyecto: Taller 2 - Big Data y Machine Learning
#
# Descripcion:
#   Cargar pacman, instalar/cargar los demas paquetes y resolver conflictos
#   entre funciones con el mismo nombre.
# Input:
#   Paquetes de R instalados; internet para instalar los que falten.
# Output:
#   Paquetes cargados y preferencias de funciones definidas en la sesion.
#   No importa bases ni exporta archivos.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Cargar pacman
# ------------------------------------------------------------------------------

if (!require("pacman")) {
  install.packages("pacman")
  library(pacman)
}

# ------------------------------------------------------------------------------
# (2) Cargar los demas paquetes
# ------------------------------------------------------------------------------

# p_load() instala los paquetes faltantes y carga los instalados.
p_load(
  # (2.1) Datos y presentacion
  tidyverse,     # Lectura de CSV, limpieza, agregacion, uniones y graficos.
  readxl,        # Lectura de los archivos Excel de documentacion.
  writexl,       # Exportacion de tablas descriptivas a Excel.

  # (2.2) Entrenamiento y modelos
  stats,         # lm() y glm(): modelo lineal, logit y probit (incluido en R).
  lava,          # Dependencia indirecta de caret que falta en el entorno actual.
  caret,         # Validacion cruzada, tuning y rebalanceo dentro de los folds.
  glmnet,        # Ridge, lasso y elastic net para regresion y clasificacion.
  MASS,          # Analisis discriminante lineal (LDA) y cuadratico (QDA).
  naivebayes,    # Naive Bayes; motor naive_bayes de caret.
  rpart,         # Arboles de clasificacion y regresion (CART).
  rpart.plot,    # Visualizacion de los arboles ajustados con rpart.
  ranger,        # Bagging y random forest con implementacion eficiente.
  randomForest,  # Implementacion alternativa de random forest.
  adabag,        # AdaBoost.M1 y bagging de arboles.
  gbm,           # Gradient boosting de arboles.
  xgboost,       # Extreme Gradient Boosting (XGBoost).
  lightgbm,      # Light Gradient Boosting Machine (LightGBM).

  # (2.3) Evaluacion
  pROC,          # Curvas ROC, AUC y evaluacion de umbrales.
  MLmetrics,     # Metricas de clasificacion; dependencia de prSummary de caret.
  Metrics,       # Metricas de regresion y clasificacion.

  # (2.4) Organizacion y replicabilidad
  rstudioapi,    # Acceso al proyecto y al editor cuando se usa RStudio.
  conflicted,    # Resuelve funciones homonimas entre paquetes.
  docstring,     # Consulta de documentacion escrita dentro de funciones.
  renv           # Permite fijar/restaurar versiones cuando se valide el entorno.
)

# ------------------------------------------------------------------------------
# (3) Resolver conflictos entre funciones
# ------------------------------------------------------------------------------

# Elegir dplyr cuando otros paquetes tengan funciones con el mismo nombre.
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("lag", "dplyr")