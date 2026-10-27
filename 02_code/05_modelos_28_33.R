# ------------------------------------------------------------------------------
# Archivo: 05_modelos_28_33.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Incorporar las seis especificaciones adicionales, conservando sus resultados y variables.
# Input:
#   Resultados guardados en 03_outputs/; funciones de exportacion de 04_modelos_1_27.R.
# Output:
#   Modelos 28 a 33 en los Excel comunes y sus CSV id/pobre.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# (1) Identificar las especificaciones 28 a 33
# ------------------------------------------------------------------------------

especificaciones <- tribble(
  ~numero, ~nombre, ~carpeta, ~comparacion, ~archivo, ~tipo,
  28, "EN_logit_seleccion_lasso", "lasso_elastic_net", "comparacion_modelos.csv", "ganador.rds", "glmnet",
  29, "Random_forest_seleccion_lasso", "random_forest", "comparacion_rf.csv", "ganador_rf.rds", "ranger",
  30, "XGBoost_base", "xgboost", "comparacion_xgb.csv", "configuracion_ganador.rds", "xgb",
  31, "LightGBM_base", "lightgbm", "comparacion_lgb.csv", "configuracion_ganador.rds", "lgb",
  32, "EN_logit_ingreso_predicho", "logit_en_consolidado", "validacion.csv", "ganador.rds", "glmnet",
  33, "XGBoost_ingreso_predicho", "xgb_consolidado", "validacion.csv", "configuracion.rds", "xgb"
)

# ------------------------------------------------------------------------------
# (2) Recuperar el ganador y sus metadatos, sin entrenar ningun modelo
# ------------------------------------------------------------------------------

recuperar_resultado <- function(numero) {
  h <- especificaciones[especificaciones$numero == numero, ]
  stopifnot(nrow(h) == 1L)
  carpeta <- file.path(dir_resultados, h$carpeta)
  registro <- readRDS(file.path(carpeta, h$archivo))
  tabla <- read_csv(file.path(carpeta, h$comparacion), show_col_types = FALSE)
  # Los scripts especificaciones guardaron el ganador en la primera fila de comparacion.
  ganador <- tabla[1, ]
  conf <- registro$configuracion
  if (is.null(conf)) conf <- list(corte = registro$corte, columnas = registro$columnas)
  if (!is.null(conf$corte)) stopifnot(isTRUE(all.equal(as.numeric(conf$corte), ganador$corte)))
  for (v in intersect(c("alpha", "lambda", "max_depth", "rondas"), names(conf))) {
    if (v %in% names(ganador)) stopifnot(isTRUE(all.equal(as.numeric(conf[[v]]), ganador[[v]])))
  }
  # Leer las etiquetas ya calculadas: no modificar el punto de corte guardado.
  archivo_csv <- file.path(carpeta, "clasificacion_test.csv")
  if (!file.exists(archivo_csv)) archivo_csv <- file.path(carpeta, "predicciones_test.csv")
  pred <- read_csv(archivo_csv, col_types = cols(id = col_character()), show_col_types = FALSE) |>
    select(id, pobre)
  variables <- registro$columnas
  if (is.null(variables)) variables <- conf$columnas
  if (is.null(variables)) stop("El ganador no conserva sus columnas: ", carpeta)
  parametros <- registro$parametros
  if (is.null(parametros)) parametros <- as.list(conf[setdiff(names(conf), "columnas")])
  parametros <- c(parametros, as.list(ganador[setdiff(names(ganador), names(parametros))]))
  parametros <- parametros[!grepl("^f1|precision|sensibilidad|especificidad|accuracy|positivos|negativos", names(parametros))]
  if (h$tipo == "glmnet") {
    parametros$family <- if (inherits(registro$modelo, "lognet")) "binomial" else "gaussian"
    parametros$lambda_path <- registro$modelo$lambda
    parametros$standardize <- TRUE
  }
  if (h$tipo == "ranger") {
    parametros$probability <- TRUE
    parametros$replace <- TRUE
    parametros$sample.fraction <- 1
    parametros$max.depth <- 0
    parametros$splitrule <- "gini"
  }
  # Solo copiar valores de metadatos; nunca imputar metricas guardadas faltantes.
  excluir_metricas <- c("f1_validacion", "precision", "sensibilidad", "especificidad", "accuracy",
                          "verdaderos_positivos", "falsos_positivos", "falsos_negativos", "verdaderos_negativos")
  fila <- as.list(ganador[setdiff(names(ganador), excluir_metricas)])
  fila$algoritmo <- h$tipo
  fila$ganador_del_bloque <- if ("modelo" %in% names(ganador)) ganador$modelo else
    if ("configuracion" %in% names(ganador)) ganador$configuracion else
      if ("version" %in% names(ganador)) ganador$version else h$nombre
  fila$base <- if (h$numero <= 31) "Base con seleccion de predictores" else
    "Base consolidada con ingreso RF y matrices propias por fold"
  fila$etapa_ingreso <- if (h$numero <= 31) "Sin ingreso predicho agregado de la consolidacion posterior" else
    "Ingreso RF recalculado dentro de la CV del clasificador; cache guardada"
  fila$F1_CV <- ganador$f1_cv
  fila$f1_cv <- NULL
  fila$F1_OOB <- NA_real_
  fila$punto_corte <- ganador$corte
  fila$corte <- NULL
  fila$criterio_hiperparametros <- "F1 OOF y corte conjuntamente; rondas con parada interna cuando aplica"
  fila$criterio_ganador_bloque <- if (h$numero %in% 31:33) "F1 CV" else
    "F1 de validacion interna entre candidatos; desempate por F1 CV"
  fila$metodo_corte <- "Maximo F1 sobre predicciones OOF; grilla 0.02 a 0.98, paso 0.02"
  fila$n_train <- if (!is.null(registro$grupos)) nrow(registro$grupos) else NA_integer_
  fila$n_CV <- if (!is.null(registro$grupos)) sum(registro$grupos$grupo %in% 2:4) else NA_integer_
  if (h$numero <= 31) {
    par <- read_csv(file.path(dir_resultados, "random_forest", "particiones.csv"), show_col_types = FALSE)
    fila$n_train <- nrow(par)
    fila$n_CV <- sum(par$grupo %in% 2:4)
    fila$distribucion_muestras <- "20% seleccion, 60% ajuste/CV, 20% validacion interna"
  } else {
    oof <- file.path(carpeta, "oof_ganador.csv")
    if (file.exists(oof)) fila$n_CV <- nrow(read_csv(oof, col_select = 1, show_col_types = FALSE))
    fila$distribucion_muestras <- "CV en la muestra de ajuste; validacion interna separada"
  }
  fila$folds <- 5L
  fila$semilla_flujo <- 2026L
  fila$rebalanceo <- "Ninguno"
  fila$n_predictores <- length(variables)
  fila$parametros_completos <- texto_entrega(parametros)
  for (v in intersect(c("alpha", "lambda", "family", "standardize", "mtry", "arboles", "min_node_size",
                         "max_depth", "eta", "rondas", "min_child_weight", "subsample", "colsample_bytree",
                         "num_leaves", "learning_rate", "lambda_l1", "lambda_l2", "min_data_in_leaf",
                         "feature_fraction", "bagging_fraction", "sample.fraction", "splitrule", "replace"), names(parametros))) {
    if (length(parametros[[v]]) == 1L) fila[[v]] <- parametros[[v]]
  }
  fila$archivo_origen <- file.path("03_outputs", h$carpeta, h$archivo)
  fila$huella_resultado <- unname(tools::md5sum(file.path(carpeta, h$archivo)))
  fila$comparabilidad <- "Especificacion con base, seleccion y particiones propias; no es una comparacion controlada entre bases"
  fila$version_paquete <- if (!is.null(registro$version_paquete)) registro$version_paquete else
    "No guardada en el resultado guardado"

  # Grilla realmente guardada, cuando existe; no reconstruir una grilla inexistente.
  grilla <- data.frame()
  busqueda <- file.path(carpeta, "busqueda_cv.csv")
  if (file.exists(busqueda)) {
    grilla <- read_csv(busqueda, show_col_types = FALSE)
    fila$origen_grilla <- file.path("03_outputs", h$carpeta, "busqueda_cv.csv")
  } else {
    fila$origen_grilla <- "No se guardo la grilla completa; comparar candidatos en el archivo de origen"
    grilla <- tabla
  }
  if (h$numero == 28) {
    # El archivo de resultados solo guardo el ganador de cada candidato.
    # Completar la grilla DOCUMENTADA por el codigo conservado, distinguiendo
    # su procedencia de una captura inmutable de aquella ejecucion guardada.
    entorno <- new.env(parent = baseenv())
    for (e in parse(file.path(root, "02_code", "04_lasso_elastic_net.R"))) {
      if (is.call(e) && identical(e[[1]], as.name("<-")) && is.symbol(e[[2]]) &&
          as.character(e[[2]]) %in% c("alphas_en", "lambdas", "configuracion")) eval(e, entorno)
    }
    grilla <- bind_rows(lapply(seq_len(nrow(entorno$configuracion)), function(j) {
      candidato <- entorno$configuracion[j, ]
      expand.grid(candidato = candidato$nombre, familia = candidato$familia,
                   seleccion = candidato$seleccion,
                   alpha = if (candidato$seleccion == "todas") 1 else entorno$alphas_en,
                   lambda = entorno$lambdas, stringsAsFactors = FALSE)
    }))
    fila$origen_grilla <- "Documentada en 02_code/04_lasso_elastic_net.R conservado; no captura inmutable de la ejecucion"
    fila$grilla_cortes <- "0.02 a 0.98; paso 0.02"
  }
  # La hoja Grillas muestra parametros candidatos; las metricas corresponden
  # exclusivamente al ganador en la hoja Comparacion.
  grilla <- grilla[, !grepl("^f1|precision|sensibilidad|especificidad|accuracy|positivos|negativos", names(grilla)), drop = FALSE]

  # Importancia del ganador, no de candidatos descartados.
  importancia <- NULL
  metrica <- "No calculada"
  if (h$tipo == "glmnet") {
    co <- as.matrix(coef(registro$modelo, s = conf$lambda))[, 1]
    co <- co[names(co) != "(Intercept)"]
    importancia <- tibble(variable = names(co), importancia = abs(as.numeric(co)))
    metrica <- "Valor absoluto del coeficiente guardado; depende de la escala, no permutacion"
  } else if (h$tipo == "ranger") {
    im <- registro$modelo$variable.importance
    if (length(im)) {
      importancia <- tibble(variable = names(im), importancia = as.numeric(im))
      metrica <- "Permutacion OOB: incremento del error Brier (ranger probabilistico)"
    }
  } else {
    ruta_im <- file.path(carpeta, "importancia_gain.csv")
    if (file.exists(ruta_im)) {
      im <- read_csv(ruta_im, show_col_types = FALSE)
      importancia <- tibble(variable = im$Feature, importancia = im$Gain)
    } else if (h$tipo == "xgb" && file.exists(file.path(carpeta, "ganador.ubj"))) {
      m <- xgb.load(file.path(carpeta, "ganador.ubj"))
      im <- as.data.frame(xgb.importance(model = m))
      importancia <- tibble(variable = im$Feature, importancia = im$Gain)
      rm(m)
    }
    if (!is.null(importancia)) metrica <- paste("Ganancia de divisiones", h$tipo, "; no permutacion")
  }

  registrar_modelo(h$numero, h$nombre, pred, fila, variables, grilla,
                    importancia, metrica)
  rm(registro, pred, importancia)
  gc(verbose = FALSE)
}

# ------------------------------------------------------------------------------
# (3) Exportar las seis especificaciones
# ------------------------------------------------------------------------------

for (numero in 28:33) recuperar_resultado(numero)
