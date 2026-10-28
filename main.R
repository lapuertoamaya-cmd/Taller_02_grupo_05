# ------------------------------------------------------------------------------
# Archivo: main.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Ejecutar en orden el setup, la limpieza, el ingreso y los dos bloques de modelos.
# Input:
#   Abrir Taller_02_grupo_05.Rproj en RStudio; bases originales en 01_data/01_raw/.
# Output:
#   Bases de modelado, CSV numerados y los dos Excel de resultados.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Definir la raiz del proyecto
# ------------------------------------------------------------------------------

if (!require("rstudioapi")) install.packages("rstudioapi")
library(rstudioapi)
root <- getActiveProject()
setwd(root)

# ------------------------------------------------------------------------------
# (2) Definir rutas y semilla
# ------------------------------------------------------------------------------

dir_codigo <- file.path(root, "02_code")
dir_datos_raw <- file.path(root, "01_data", "01_raw")
dir_clean <- file.path(root, "01_data", "02_clean")
dir_features <- file.path(root, "01_data", "03_features")
dir_resultados <- file.path(root, "03_outputs")
dir_documentacion <- file.path(root, "04_documentation")
set.seed(2026)

# ------------------------------------------------------------------------------
# (3) Ejecutar los scripts en orden
# ------------------------------------------------------------------------------

scripts <- c(
  "00_set_up.R",
  "01_cleaning.R",
  "02_creation_income_var.R",
  "03_descriptive_analysis.R",
  "04_modelos_1_27.R",
  "05_modelos_28_33.R"
)
# Luego se agregara 06_modelos_34_en_adelante.R, cuando este elaborado.
for (script in scripts) source(file.path(dir_codigo, script), encoding = "UTF-8")
