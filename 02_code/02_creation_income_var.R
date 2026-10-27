# ------------------------------------------------------------------------------
# Archivo: 02_creation_income_var.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Construir el ingreso predicho y guardar las bases de modelado; reutilizarlas si las entradas no cambiaron.
# Input:
#   Bases originales y versiones limpias de 01_cleaning.R; rutas y paquetes de main.R.
# Output:
#   db_train.rds, db_test.rds y folds.rds en 01_data/03_features/, con huella de sus entradas.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Comprobar si el ingreso ya esta calculado con estas entradas
# ------------------------------------------------------------------------------

archivos_ingreso <- c(
  file.path(dir_datos_raw, c("train_personas.csv", "test_personas.csv", "train_hogares.csv", "test_hogares.csv")),
  file.path(dir_codigo, c("01_cleaning.R", "02_creation_income_var.R"))
)
huella_ingreso <- unname(tools::md5sum(archivos_ingreso))
archivos_features <- file.path(dir_features, c("db_train.rds", "db_test.rds", "folds.rds"))
archivo_huella <- file.path(dir_features, "huella_ingreso.rds")
if (all(file.exists(c(archivos_features, archivo_huella))) &&
    identical(readRDS(archivo_huella), huella_ingreso)) {
  db_train <- readRDS(archivos_features[1])
  db_test <- readRDS(archivos_features[2])
  folds <- readRDS(archivos_features[3])
  message("Ingreso ya calculado: se reutilizan las bases guardadas.")
} else {
# ------------------------------------------------------------------------------
# (2) Construir el ingreso y las bases de modelado
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (2.1) Definir las funciones
# ------------------------------------------------------------------------------

# (2.1.1) Etapa 1: predictores de la ecuacion de ingreso individual
# Variables originales de personas que existen en test (sin identificadores,
# factores de expansion ni Depto) mas variables construidas en 01_cleaning.R.
vars_etapa1_raw <- setdiff(names(db_test_personas),
                           c("id", "Orden", "Fex_c", "Fex_dpto", "Depto", "Pet"))
vars_etapa1_clean <- c("anios_educ", "bin_informal", "bin_informal_ipm",
                       "bin_trabajador_vulnerable", "bin_salud_no_contributiva",
                       "bin_salud_subsidiada", "num_beneficios_laborales",
                       "bin_bonificaciones", "bin_subsidios", "bin_rentista",
                       "bin_remesas", "bin_pensionista", "bin_ingresos_capital",
                       "bin_ayuda_instituciones", "bin_segundo_trabajo",
                       "horas_segundo_trabajo")
vars_etapa1 <- c(vars_etapa1_raw, vars_etapa1_clean)

# Variables categoricas sin orden natural
vars_factor_etapa1 <- c("Dominio", "Oficio", "P6050", "P6100", "P6240",
                        "P6430", "P7050", "P7350")

fun_preprocess_etapa1 <- function(.raw_train, .raw_test, .clean_train, .clean_test) {
  #' Preparar las bases de personas para la ecuacion de ingreso
  #'
  #' @description Conserva la poblacion en edad de trabajar (Ingtot es NA
  #' exactamente para quienes no estan en la PET y su ingreso se fija en 0 al
  #' agregar). Une las variables construidas en 01_cleaning.R por id y
  #' orden. P7045 con 98/99 se trata como faltante. Los NA (no aplica o
  #' desconocido) se codifican como -1 en las numericas y "NA" en las
  #' categoricas. Los niveles de los factores se fijan con train y test.
  #'
  #' @returns \code{list} con \code{train} y \code{test}. Train incluye
  #' y_log_ingtot = log1p(Ingtot).
  prep <- function(.raw, .clean) {
    .raw |>
      filter(!is.na(Pet)) |>
      left_join(.clean |> select(id, id_persona, all_of(vars_etapa1_clean)),
                by = c("id", "Orden" = "id_persona")) |>
      mutate(P7045 = if_else(P7045 %in% c(98, 99), NA_real_, P7045),
             cod_parentesco = P6050) |>
      select(id, cod_parentesco, any_of("Ingtot"),
             all_of(vars_etapa1)) |>
      mutate(across(all_of(vars_factor_etapa1),
                    ~ replace_na(as.character(.x), "NA")),
             across(setdiff(vars_etapa1, vars_factor_etapa1),
                    ~ replace_na(as.numeric(.x), -1)))
  }

  db_tr <- prep(.raw_train, .clean_train) |>
    mutate(y_log_ingtot = log1p(Ingtot)) |>
    select(-Ingtot)
  db_te <- prep(.raw_test, .clean_test)

  for (v in vars_factor_etapa1) {
    niveles <- sort(union(unique(db_tr[[v]]), unique(db_te[[v]])))
    db_tr[[v]] <- factor(db_tr[[v]], levels = niveles)
    db_te[[v]] <- factor(db_te[[v]], levels = niveles)
  }

  list(train = db_tr, test = db_te)
}


fun_matriz_etapa1 <- function(.db) {
  #' Matriz dispersa de predictores de la etapa 1 para xgboost
  Matrix::sparse.model.matrix(~ . - 1, data = .db[, vars_etapa1])
}


# Hiperparametros de las dos partes de la etapa 1 (fijos, sin tuning)
params_etapa1 <- list(eta = 0.05, max_depth = 8, min_child_weight = 20,
                      subsample = 0.8, colsample_bytree = 0.8,
                      tree_method = "hist", nthread = 2)

fun_xgb_etapa1 <- function(.x_tr, .y_tr, .x_new) {
  #' Modelo de dos partes (xgboost) para el ingreso total individual
  #'
  #' @description Parte 1: clasificador de P(Y > 0). Parte 2: regresion de
  #' log1p(Y) entrenada solo con quienes tienen ingreso positivo.
  #'
  #' @returns \code{numeric}. Ingreso esperado P(Y > 0) * E[Y | Y > 0].
  con_ingreso_tr <- .y_tr > 0

  set.seed(2026)
  modelo_p <- xgb.train(
    params = c(params_etapa1, objective = "binary:logistic"),
    data = xgb.DMatrix(.x_tr, label = as.integer(con_ingreso_tr)),
    nrounds = 400, verbose = 0
  )

  set.seed(2026)
  modelo_mu <- xgb.train(
    params = c(params_etapa1, objective = "reg:squarederror"),
    data = xgb.DMatrix(.x_tr[con_ingreso_tr, ], label = .y_tr[con_ingreso_tr]),
    nrounds = 600, verbose = 0
  )

  d_new <- xgb.DMatrix(.x_new)
  predict(modelo_p, d_new) * pmax(expm1(predict(modelo_mu, d_new)), 0)
}


# (2.1.2) Etapa 2: variables del hogar con la formula del DANE
fun_agregar_ingreso <- function(.db_personas_pred, .db_hogares_raw) {
  #' Variables de ingreso predicho del hogar
  #'
  #' @description Suma el ingreso predicho de la unidad de gasto (excluye
  #' P6050 = 6, 7 y 8), suma el arriendo imputado solo cuando P5090 es 1 o 4
  #' y P5130 no es 98 ni 99, y divide por Npersug. Con el ingreso observado,
  #' esta regla reproduce Pobre en el 99,9% de los hogares de train.
  #' Agrega el ingreso maximo, el numero de perceptores (ingreso predicho
  #' mayor a 200.000 pesos) y la participacion del jefe.
  ing_ug <- .db_personas_pred |>
    filter(!cod_parentesco %in% c(6, 7, 8)) |>
    group_by(id) |>
    summarize(num_ing_ug_hat = sum(num_ing_total_hat),
              num_ing_max_hat = max(num_ing_total_hat),
              num_perceptores_hat = sum(num_ing_total_hat > 2e5),
              num_ing_jefe_hat = sum(num_ing_total_hat[cod_parentesco == 1]),
              .groups = "drop")

  .db_hogares_raw |>
    select(id, Lp, Npersug, P5090, P5130) |>
    left_join(ing_ug, by = "id") |>
    mutate(
      across(c(num_ing_ug_hat, num_ing_max_hat, num_perceptores_hat,
               num_ing_jefe_hat), ~ replace_na(.x, 0)),
      num_arriendo_imp_dane = ifelse(P5090 %in% c(1, 4) & !(P5130 %in% c(98, 99)),
                                     replace_na(P5130, 0), 0),
      n_ug = pmax(Npersug, 1),
      num_ing_pc_hat = (num_ing_ug_hat + num_arriendo_imp_dane) / n_ug,
      num_ratio_lp_hat = num_ing_pc_hat / Lp,
      num_ratio_lp_sin_arr_hat = num_ing_ug_hat / n_ug / Lp,
      prop_ing_jefe_hat = ifelse(num_ing_ug_hat > 0,
                                 num_ing_jefe_hat / num_ing_ug_hat, 0)
    ) |>
    select(id, num_ing_ug_hat, num_ing_pc_hat, num_ratio_lp_hat,
           num_ratio_lp_sin_arr_hat, num_ing_max_hat, num_perceptores_hat,
           prop_ing_jefe_hat)
}


# ------------------------------------------------------------------------------
# (2.2) Bases de partida: hogares listos para modelar (01_cleaning.R)
# ------------------------------------------------------------------------------

db_train <- train_clean
db_test  <- test_clean

id_train <- db_train$id
stopifnot(identical(id_train, db_train_hogares$id))


# ------------------------------------------------------------------------------
# (2.3) Folds comunes (los mismos de la seccion de controles de los modelos)
# ------------------------------------------------------------------------------

set.seed(2026)
folds <- createFolds(db_train$Pobre, k = 5, returnTrain = TRUE)

# Fold de cada hogar: el hogar pertenece al fold k cuando queda fuera de
# folds[[k]] (returnTrain = TRUE). Cada persona hereda el fold de su hogar.
fold_hogar <- integer(length(id_train))
for (k in seq_along(folds)) fold_hogar[-folds[[k]]] <- k


# ------------------------------------------------------------------------------
# (2.4) Etapa 1: ingreso total individual
# ------------------------------------------------------------------------------

etapa1 <- fun_preprocess_etapa1(db_train_personas, db_test_personas,
                                db_train_personas_clean, db_test_personas_clean)
db_train_etapa1 <- etapa1$train |>
  left_join(tibble(id = id_train, fold = fold_hogar), by = "id")
db_test_etapa1 <- etapa1$test
rm(etapa1)
stopifnot(!anyNA(db_train_etapa1$fold))

x_train_etapa1 <- fun_matriz_etapa1(db_train_etapa1)
x_test_etapa1  <- fun_matriz_etapa1(db_test_etapa1)
stopifnot(identical(colnames(x_train_etapa1), colnames(x_test_etapa1)),
          nrow(x_train_etapa1) == nrow(db_train_etapa1),
          nrow(x_test_etapa1) == nrow(db_test_etapa1))

# Fuera de muestra para train: el modelo del fold k se entrena con las
# personas de los otros cuatro folds y predice las del fold k.
db_train_etapa1$num_ing_total_hat <- NA_real_
for (k in seq_along(folds)) {
  en_k <- db_train_etapa1$fold == k
  db_train_etapa1$num_ing_total_hat[en_k] <-
    fun_xgb_etapa1(x_train_etapa1[!en_k, ], db_train_etapa1$y_log_ingtot[!en_k],
                   x_train_etapa1[en_k, ])
  cat("Etapa 1, fold", k, "listo\n")
}

# Test: modelo entrenado con todas las personas de train
db_test_etapa1$num_ing_total_hat <-
  fun_xgb_etapa1(x_train_etapa1, db_train_etapa1$y_log_ingtot, x_test_etapa1)
gc()


# ------------------------------------------------------------------------------
# (2.5) Etapa 2 y bases finales para los modelos
# ------------------------------------------------------------------------------

db_train_ingreso <- fun_agregar_ingreso(db_train_etapa1, db_train_hogares)
db_test_ingreso  <- fun_agregar_ingreso(db_test_etapa1,  db_test_hogares)

db_train <- db_train |>
  left_join(db_train_ingreso, by = "id") |>
  select(-id)

db_test <- db_test |>
  left_join(db_test_ingreso, by = "id")

# Verificacion: no deben quedar NA en los predictores
sum(is.na(db_train))
sum(is.na(db_test[, setdiff(names(db_train), "Pobre")]))


# ------------------------------------------------------------------------------
# (2.6) Exportar las bases con las variables de ingreso
# ------------------------------------------------------------------------------

dir.create(dir_features, recursive = TRUE, showWarnings = FALSE)
saveRDS(db_train, file.path(dir_features, "db_train.rds"))
saveRDS(db_test,  file.path(dir_features, "db_test.rds"))
saveRDS(folds,    file.path(dir_features, "folds.rds"))

# Para modelar sin repetir este script:
# db_train <- readRDS(file.path(dir_features, "db_train.rds"))
# db_test  <- readRDS(file.path(dir_features, "db_test.rds"))
# folds    <- readRDS(file.path(dir_features, "folds.rds"))
  saveRDS(huella_ingreso, archivo_huella)
}

# ------------------------------------------------------------------------------
# (3) Liberar las bases individuales antes de estimar los modelos
# ------------------------------------------------------------------------------

temporales_ingreso <- c(
  grep("^db_(train|test)_(personas|hogares|etapa1|ingreso)", ls(), value = TRUE),
  intersect(c("train_clean", "test_clean", "x_train_etapa1", "x_test_etapa1", "etapa1"), ls())
)
rm(list = temporales_ingreso)
gc(verbose = FALSE)
