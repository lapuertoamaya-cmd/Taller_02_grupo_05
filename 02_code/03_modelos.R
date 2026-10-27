# ------------------------------------------------------------------------------
# Archivo: 03_modelos.R
# Proyecto: Taller 2 - Big Data y Machine Learning
#
# Descripcion:
#   Entrenar y evaluar los 27 modelos de clasificacion de pobreza con el mismo
#   protocolo: mismos predictores, mismos folds, seleccion de hiperparametros
#   por ROC (umbral fijo 0.5) o por F de la clase "Si" (umbral calibrado en
#   la curva PR con predicciones fuera de muestra) y, en algunos modelos,
#   rebalanceo por oversampling.
#   Modelos 1 a 21: Elastic Net, LDA, QDA, Naive Bayes, Bagging, Random
#   Forest, AdaBoost y Gradient Boosting. Modelos 22 a 27: XGBoost y LightGBM.
# Input:
#   db_train, db_test y folds creados por 02_ingreso.R (en memoria o desde
#   01_data/03_features/).
# Output:
#   Un CSV por modelo con las predicciones de test (id, pobre) para Kaggle,
#   en el directorio de trabajo.
# ------------------------------------------------------------------------------


# 0. BASES DE PARTIDA ----------------------------------------------------------

# Si 02_ingreso.R no se ejecuto en esta sesion, se leen sus resultados.
if (!exists("db_train") || !exists("db_test") || !exists("folds")) {
  db_train <- readRDS(file.path(dir_features, "db_train.rds"))
  db_test  <- readRDS(file.path(dir_features, "db_test.rds"))
  folds    <- readRDS(file.path(dir_features, "folds.rds"))
}


# 1. FUNCION AUXILIAR DE METRICAS (usada por todos los modelos) --------------

calcular_metricas <- function(model, umbral, strategy) {
  #' Calcular precision, recall y F1 para un modelo glmnet con un umbral dado
  #'
  #' @description Filtra las predicciones out-of-fold (\code{model$pred})
  #' correspondientes a la mejor combinación de hiperparámetros
  #' (\code{model$bestTune}), aplica el umbral de clasificación indicado y
  #' calcula la matriz de confusión con sus métricas de precision, recall y
  #' F1, evitando el problema de precisión flotante de \code{inner_join()}.
  #'
  #' @param model Objeto \code{train} de caret (method = "glmnet").
  #' @param umbral \code{numeric}. Umbral de clasificación sobre la
  #' probabilidad de la clase "Si".
  #' @param strategy \code{character}. Nombre descriptivo de la estrategia,
  #' para identificar la fila en la tabla final.
  #'
  #' @returns \code{tibble} de una fila con alpha, lambda, threshold,
  #' precision, recall y F1.
  pred_model <- model$pred |>
    filter(alpha == model$bestTune$alpha,
           lambda == model$bestTune$lambda) |>
    mutate(pred_clase = factor(ifelse(Si >= umbral, "Si", "No"),
                               levels = levels(obs)))
  
  cm <- confusionMatrix(
    data = pred_model$pred_clase,
    reference = pred_model$obs,
    positive = "Si",
    mode = "prec_recall"
  )
  
  tibble(
    strategy = strategy,
    alpha = model$bestTune$alpha,
    lambda = model$bestTune$lambda,
    threshold = umbral,
    precision = unname(cm$byClass["Precision"]),
    recall = unname(cm$byClass["Recall"]),
    F1 = unname(cm$byClass["F1"])
  )
}


# 2. FOLDS Y CONTROLES COMUNES -------------------------------------------------

# Los folds se crean en 02_ingreso.R (set.seed(2026), createFolds con
# returnTrain = TRUE) y son los mismos de la etapa de ingreso. No se vuelven
# a crear aqui para garantizar que ambos usen exactamente la misma particion.
stopifnot(exists("folds"), length(folds) == 5)

# Control con métrica ROC (umbral-independiente): usado para seleccionar
# hiperparámetros de forma robusta.
ctrl_roc <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Control con métrica F1 (prSummary, umbral fijo de 0.5 internamente): usado
# para seleccionar hiperparámetros maximizando F1 directamente.
# F1 de la clase "Si" (prSummary calcula F para la primera categoría, "No")
f1_si <- function(data, lev = NULL, model = NULL) {
  tp <- sum(data$obs == "Si" & data$pred == "Si")
  fp <- sum(data$obs == "No" & data$pred == "Si")
  fn <- sum(data$obs == "Si" & data$pred == "No")
  c(Precision = ifelse(tp + fp == 0, 0, tp / (tp + fp)),
    Recall    = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
    F         = ifelse(2 * tp + fp + fn == 0, 0, 2 * tp / (2 * tp + fp + fn)))
}

ctrl_f1 <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = f1_si,
  savePredictions = "final"
)

# Función de resumen combinada: Accuracy (defaultSummary) + F1 (prSummary)
multiStats <- function(data, lev = NULL, model = NULL) {
  c(defaultSummary(data, lev, model),
    f1_si(data, lev, model))
}

ctrl_accuracy <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = multiStats,
  savePredictions = "final"
)

# Control con oversampling. Se usa ROC como métrica de selección
ctrl_up <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up",
  savePredictions = "final"
)
ctrl_up_f1 <- trainControl(
  method = "cv",
  number = 5,
  index = folds,
  classProbs = TRUE,
  summaryFunction = f1_si,
  sampling = "up",
  savePredictions = "final"
)

# Grilla de hiperparámetros para elastic net (común a todos los modelos glmnet)
tune_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.05),
  lambda = 10^seq(log10(100), log10(0.001), length.out = 20)
)


# MODELO 1: LOGIT ELASTIC NET - SELECCION POR ROC, UMBRAL FIJO 0.5 -------------
logit_roc <- train(
  Pobre ~ .,
  data = db_train,
  metric = "ROC",
  maximize = TRUE,
  trControl = ctrl_roc,
  tuneGrid = tune_grid,
  method = "glmnet",
  family = "binomial",
  standardize = TRUE
)

logit_roc
logit_roc$bestTune

pred_cv_logit_roc <- logit_roc$pred |>
  filter(alpha == logit_roc$bestTune$alpha,
         lambda == logit_roc$bestTune$lambda)

coefs_logit_roc <- coef(logit_roc$finalModel, s = logit_roc$bestTune$lambda)
coefs_df_logit_roc <- data.frame(
  variable = rownames(coefs_logit_roc),
  coeficiente = as.numeric(coefs_logit_roc)
) |> filter(coeficiente != 0) |> arrange(desc(abs(coeficiente)))
coefs_df_logit_roc

cm_logit_roc <- calcular_metricas(logit_roc, umbral = 0.5,
                                  strategy = "Logit EN - ROC - umbral 0.5")
cm_logit_roc

prob_logit_roc <- predict(logit_roc, newdata = db_test, type = "prob")[["Si"]]

predictSample_logit_roc <- db_test |>
  mutate(pobre_prob = prob_logit_roc,
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

lambda_str_logit_roc <- round(logit_roc$bestTune$lambda, 4)
alpha_str_logit_roc  <- round(logit_roc$bestTune$alpha, 2)
name_logit_roc <- paste0("EN_lambda_", lambda_str_logit_roc,
                         "_alpha_", alpha_str_logit_roc, ".csv")

write.csv(predictSample_logit_roc, name_logit_roc, row.names = FALSE)



# MODELO 2: LOGIT ELASTIC NET - SELECCION POR F1, UMBRAL OPTIMO EN CURVA PR-----

# --- 2a. Versión con selección por Accuracy (para comparar) ---
logit_accuracy <- train(
  Pobre ~ .,
  data = db_train,
  metric = "Accuracy",
  maximize = TRUE,
  trControl = ctrl_accuracy,
  tuneGrid = tune_grid,
  method = "glmnet",
  family = "binomial",
  standardize = TRUE
)

logit_accuracy$bestTune

# --- 2b. Versión con selección por F1 (prSummary) ---
logit_f1 <- train(
  Pobre ~ .,
  data = db_train,
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_f1,
  tuneGrid = tune_grid,
  method = "glmnet",
  family = "binomial",
  standardize = TRUE
)

logit_f1$bestTune

y_obs_logitf1  <- logit_f1$pred$obs[logit_f1$pred$lambda == logit_f1$bestTune$lambda &
                                      logit_f1$pred$alpha == logit_f1$bestTune$alpha]
p_pred_logitf1 <- logit_f1$pred$Si[logit_f1$pred$lambda == logit_f1$bestTune$lambda &
                                     logit_f1$pred$alpha == logit_f1$bestTune$alpha]

roc_logitf1 <- pROC::roc(
  response  = y_obs_logitf1,
  predictor = p_pred_logitf1,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_logitf1 <- data.frame(pROC::coords(
  roc = roc_logitf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_logitf1 <- pr_cutoffs_logitf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_logitf1

cm_logitf1_base <- calcular_metricas(logit_f1, umbral = 0.5,
                                     strategy = "Logit EN - F1 - umbral 0.5")
cm_logitf1_pr   <- calcular_metricas(logit_f1, umbral = best_cutoff_logitf1$threshold,
                                     strategy = "Logit EN - F1 - umbral óptimo PR")

model_comparison_logit <- bind_rows(cm_logit_roc, cm_logitf1_base, cm_logitf1_pr) |>
  arrange(desc(F1))
model_comparison_logit

umbral_logitf1 <- best_cutoff_logitf1$threshold

predictSample_logitf1 <- db_test |>
  mutate(pobre_prob = predict(logit_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_logitf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

lambda_str_logitf1 <- round(logit_f1$bestTune$lambda, 4)
alpha_str_logitf1  <- round(logit_f1$bestTune$alpha, 2)
prob_str_logitf1   <- round(umbral_logitf1, 3)
name_logitf1 <- paste0("EN_lambda_", lambda_str_logitf1,
                       "_alpha_", alpha_str_logitf1,
                       "_prob_", prob_str_logitf1, ".csv")

write.csv(predictSample_logitf1, name_logitf1, row.names = FALSE)



# MODELO 3: LDA - SELECCION POR ROC, UMBRAL FIJO 0.5---------------------------


lda_roc <- train(
  Pobre ~ .,
  data = db_train,
  method = "lda",
  metric = "ROC",
  trControl = ctrl_roc,
  preProcess = c("center", "scale")
)

lda_roc

pred_cv_lda_roc <- lda_roc$pred

cm_lda_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_lda_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_lda_roc$obs)),
  reference = pred_cv_lda_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_lda_roc

predictSample_lda_roc <- db_test |>
  mutate(pobre_prob = predict(lda_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_lda_roc, "LDA.csv", row.names = FALSE)



# MODELO 4: LDA - SELECCION POR F1, UMBRAL OPTIMO EN CURVA PR------------------

lda_f1 <- train(
  Pobre ~ .,
  data = db_train,
  method = "lda",
  metric = "F",
  trControl = ctrl_f1,
  preProcess = c("center", "scale")
)

lda_f1

pred_cv_lda_f1 <- lda_f1$pred

roc_ldaf1 <- pROC::roc(
  response  = pred_cv_lda_f1$obs,
  predictor = pred_cv_lda_f1$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_ldaf1 <- data.frame(pROC::coords(
  roc = roc_ldaf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_ldaf1 <- pr_cutoffs_ldaf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_ldaf1

pred_cv_lda_f1 <- pred_cv_lda_f1 |>
  mutate(pred_clase = factor(ifelse(Si >= best_cutoff_ldaf1$threshold, "Si", "No"),
                             levels = levels(obs)))

cm_ldaf1_base <- confusionMatrix(
  data = factor(ifelse(pred_cv_lda_f1$Si >= 0.5, "Si", "No"), levels = levels(pred_cv_lda_f1$obs)),
  reference = pred_cv_lda_f1$obs, positive = "Si", mode = "prec_recall"
)

cm_ldaf1_pr <- confusionMatrix(
  data = pred_cv_lda_f1$pred_clase,
  reference = pred_cv_lda_f1$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_ldaf1_pr

umbral_ldaf1 <- best_cutoff_ldaf1$threshold

predictSample_ldaf1 <- db_test |>
  mutate(pobre_prob = predict(lda_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_ldaf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_ldaf1 <- paste0("LDA_F1_prob_", round(umbral_ldaf1, 3), ".csv")
write.csv(predictSample_ldaf1, name_ldaf1, row.names = FALSE)



# MODELO 5: QDA - SELECCION POR ROC, UMBRAL FIJO 0.5----------------------------

# QDA exige rango completo dentro de cada clase. Se excluyen las variables que
# son combinaciones lineales exactas de otras:
#   num_privaciones = suma de las cinco privaciones
#   clase = indicadora de dominio "RURAL"; depto contiene dominios completos
#   cat_tenencia determina bin_casa_propia_pagada, bin_paga_amortizacion y
#     bin_arriendo_imput
#   carga_cuidado = num_menores10 + num_mayores65
#   num_desocupados_ug y pct_desocupados: PET = ocupados + desocupados +
#     inactivos
vars_excluir_qda <- c("num_privaciones", "clase", "depto", "cat_tenencia",
                      "carga_cuidado", "num_desocupados_ug", "pct_desocupados")


qda_roc <- train(
  Pobre ~ .,
  data = db_train |> select(-all_of(vars_excluir_qda)),
  method = "qda",
  metric = "ROC",
  trControl = ctrl_roc,
  preProcess = c("center", "scale")
)

qda_roc

pred_cv_qda_roc <- qda_roc$pred

cm_qda_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_qda_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_qda_roc$obs)),
  reference = pred_cv_qda_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_qda_roc

predictSample_qda_roc <- db_test |>
  mutate(pobre_prob = predict(qda_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_qda_roc, "QDA.csv", row.names = FALSE)



# MODELO 6: QDA - SELECCION POR F1, UMBRAL OPTIMO EN CURVA PR-------------------


qda_f1 <- train(
  Pobre ~ .,
  data = db_train |> select(-all_of(vars_excluir_qda)),
  method = "qda",
  metric = "F",
  trControl = ctrl_f1,
  preProcess = c("center", "scale")
)

qda_f1

pred_cv_qda_f1 <- qda_f1$pred

roc_qdaf1 <- pROC::roc(
  response  = pred_cv_qda_f1$obs,
  predictor = pred_cv_qda_f1$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_qdaf1 <- data.frame(pROC::coords(
  roc = roc_qdaf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_qdaf1 <- pr_cutoffs_qdaf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_qdaf1

pred_cv_qda_f1 <- pred_cv_qda_f1 |>
  mutate(pred_clase = factor(ifelse(Si >= best_cutoff_qdaf1$threshold, "Si", "No"),
                             levels = levels(obs)))

cm_qdaf1_base <- confusionMatrix(
  data = factor(ifelse(pred_cv_qda_f1$Si >= 0.5, "Si", "No"), levels = levels(pred_cv_qda_f1$obs)),
  reference = pred_cv_qda_f1$obs, positive = "Si", mode = "prec_recall"
)

cm_qdaf1_pr <- confusionMatrix(
  data = pred_cv_qda_f1$pred_clase,
  reference = pred_cv_qda_f1$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_qdaf1_pr

umbral_qdaf1 <- best_cutoff_qdaf1$threshold

predictSample_qdaf1 <- db_test |>
  mutate(pobre_prob = predict(qda_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_qdaf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_qdaf1 <- paste0("QDA_F1_prob_", round(umbral_qdaf1, 3), ".csv")
write.csv(predictSample_qdaf1, name_qdaf1, row.names = FALSE)



# MODELO 7: NAIVE BAYES - SELECCION POR ROC, UMBRAL FIJO 0.5--------------------


tune_grid_nb <- expand.grid(
  laplace = 0,
  usekernel = c(TRUE, FALSE),
  adjust = 1
)

nb_roc <- train(
  Pobre ~ .,
  data = db_train,
  method = "naive_bayes",
  metric = "ROC",
  trControl = ctrl_roc,
  tuneGrid = tune_grid_nb
)

nb_roc
nb_roc$bestTune

pred_cv_nb_roc <- nb_roc$pred |>
  filter(usekernel == nb_roc$bestTune$usekernel,
         laplace == nb_roc$bestTune$laplace,
         adjust == nb_roc$bestTune$adjust)

cm_nb_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_nb_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_nb_roc$obs)),
  reference = pred_cv_nb_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_nb_roc

predictSample_nb_roc <- db_test |>
  mutate(pobre_prob = predict(nb_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_nb_roc, "NaiveBayes.csv", row.names = FALSE)


# MODELO 8: LOGIT ELASTIC NET CON OVERSAMPLING - UMBRAL FIJO 0.5---------------

logit_up05 <- train(
  Pobre ~ .,
  data = db_train,
  metric = "ROC",
  maximize = TRUE,
  trControl = ctrl_up,
  tuneGrid = tune_grid,
  method = "glmnet",
  family = "binomial",
  standardize = TRUE
)

logit_up05
logit_up05$bestTune

cm_logitup05 <- calcular_metricas(logit_up05, umbral = 0.5,
                                  strategy = "Logit EN - Oversampling - umbral 0.5")
cm_logitup05

predictSample_logitup05 <- db_test |>
  mutate(pobre_prob = predict(logit_up05, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

lambda_str_logitup05 <- round(logit_up05$bestTune$lambda, 4)
alpha_str_logitup05  <- round(logit_up05$bestTune$alpha, 2)
name_logitup05 <- paste0("EN_up_lambda_", lambda_str_logitup05,
                         "_alpha_", alpha_str_logitup05, ".csv")

write.csv(predictSample_logitup05, name_logitup05, row.names = FALSE)



# MODELO 9: LOGIT ELASTIC NET CON OVERSAMPLING - UMBRAL OPTIMO EN CURVA PR------

set.seed(2026)
logit_upf1 <- train(
  Pobre ~ .,
  data = db_train,
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_up_f1,
  tuneGrid = tune_grid,
  method = "glmnet",
  family = "binomial",
  standardize = TRUE
)

logit_upf1
logit_upf1$bestTune

pred_cv_logitup <- logit_upf1$pred |>
  filter(alpha == logit_upf1$bestTune$alpha,
         lambda == logit_upf1$bestTune$lambda)

# Verificación de colapso: sd(Si) debe ser claramente mayor que 0
summary(pred_cv_logitup$Si)
sd(pred_cv_logitup$Si)

roc_logitupf1 <- pROC::roc(
  response  = pred_cv_logitup$obs,
  predictor = pred_cv_logitup$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_logitupf1 <- data.frame(pROC::coords(
  roc = roc_logitupf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_logitupf1 <- pr_cutoffs_logitupf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_logitupf1

cm_logitupf1_pr <- calcular_metricas(logit_upf1, umbral = best_cutoff_logitupf1$threshold,
                                     strategy = "Logit EN - Oversampling - F - umbral óptimo PR")
cm_logitupf1_pr

model_comparison_up <- bind_rows(
  cm_logit_roc,
  cm_logitf1_pr,
  cm_logitup05,
  cm_logitupf1_pr
) |> arrange(desc(F1))

model_comparison_up

umbral_logitupf1 <- best_cutoff_logitupf1$threshold

predictSample_logitupf1 <- db_test |>
  mutate(pobre_prob = predict(logit_upf1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_logitupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

lambda_str_logitupf1 <- round(logit_upf1$bestTune$lambda, 4)
alpha_str_logitupf1  <- round(logit_upf1$bestTune$alpha, 2)
prob_str_logitupf1   <- round(umbral_logitupf1, 3)
name_logitupf1 <- paste0("EN_up_lambda_", lambda_str_logitupf1,
                         "_alpha_", alpha_str_logitupf1,
                         "_prob_", prob_str_logitupf1, ".csv")

write.csv(predictSample_logitupf1, name_logitupf1, row.names = FALSE)
# MODELO 10: BAGGING - UMBRAL FIJO 0.5 -----------------------------------------


# mtry = número total de predictores (bagging usa todas las variables en cada
# partición; esto es lo que lo distingue de random forest).
num_predictores_bag <- ncol(db_train) - 1

set.seed(2026)
bag_fijo <- ranger(
  formula = Pobre ~ .,
  data = db_train,
  probability = TRUE,
  splitrule = "gini",
  num.trees = 500,
  mtry = num_predictores_bag,
  min.node.size = 1,
  importance = "permutation"
)

bag_fijo
bag_fijo$prediction.error

# Probabilidades OOB (out-of-bag), equivalentes a las predicciones out-of-fold
# de caret: cada observación se predice solo con los árboles que no la
# incluyeron en su muestra bootstrap.
pred_cv_bag_fijo <- tibble(
  obs = db_train$Pobre,
  Si  = bag_fijo$predictions[, "Si"]
)

cm_bag_fijo <- confusionMatrix(
  data = factor(ifelse(pred_cv_bag_fijo$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_bag_fijo$obs)),
  reference = pred_cv_bag_fijo$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_bag_fijo

# Importancia de variables (permutation)
importancia_bag_fijo <- sort(bag_fijo$variable.importance, decreasing = TRUE)
importancia_bag_fijo

# Predicción sobre test
prob_bag_fijo <- predict(bag_fijo, data = db_test)$predictions[, "Si"]

predictSample_bag_fijo <- db_test |>
  mutate(pobre_prob = prob_bag_fijo,
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_bag_fijo, "Bagging.csv", row.names = FALSE)


# MODELO 11: BAGGING - UMBRAL OPTIMO EN CURVA PR ------------------------------


# Reutiliza las probabilidades OOB de bag_fijo (mismo modelo, no se re-estima)
roc_bagf1 <- pROC::roc(
  response  = pred_cv_bag_fijo$obs,
  predictor = pred_cv_bag_fijo$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_bagf1 <- data.frame(pROC::coords(
  roc = roc_bagf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_bagf1 <- pr_cutoffs_bagf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_bagf1

cm_bagf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_bag_fijo$Si >= best_cutoff_bagf1$threshold, "Si", "No"),
                levels = levels(pred_cv_bag_fijo$obs)),
  reference = pred_cv_bag_fijo$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_bagf1_pr

umbral_bagf1 <- best_cutoff_bagf1$threshold

predictSample_bagf1 <- db_test |>
  mutate(pobre_prob = prob_bag_fijo,
         pobre_lab = ifelse(pobre_prob >= umbral_bagf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_bagf1 <- paste0("Bagging_F1_prob_", round(umbral_bagf1, 3), ".csv")
write.csv(predictSample_bagf1, name_bagf1, row.names = FALSE)


# MODELO 12: BAGGING CON REBALANCEO - UMBRAL OPTIMO EN CURVA PR ----------------

# ranger no tiene un equivalente directo a sampling = "up" de trainControl.
# El rebalanceo se logra con case.weights: un peso por observación que
# aumenta la probabilidad de que las observaciones de la clase minoritaria
# ("Si") sean seleccionadas en cada muestra bootstrap. El peso de cada clase
# es inversamente proporcional a su frecuencia (n_total / (n_clases * n_clase)),
# de forma que ambas clases contribuyen con el mismo peso total al ajuste.
n_si <- sum(db_train$Pobre == "Si")
n_no <- sum(db_train$Pobre == "No")
n_total <- n_si + n_no

pesos_bag_up <- ifelse(db_train$Pobre == "Si",
                       n_total / (2 * n_si),
                       n_total / (2 * n_no))

set.seed(2026)
bag_up <- ranger(
  formula = Pobre ~ .,
  data = db_train,
  probability = TRUE,
  splitrule = "gini",
  num.trees = 500,
  mtry = num_predictores_bag,
  min.node.size = 1,
  importance = "permutation",
  replace = TRUE,
  case.weights = pesos_bag_up
)

bag_up
bag_up$prediction.error

pred_cv_bag_up <- tibble(
  obs = db_train$Pobre,
  Si  = bag_up$predictions[, "Si"]
)

roc_bagupf1 <- pROC::roc(
  response  = pred_cv_bag_up$obs,
  predictor = pred_cv_bag_up$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_bagupf1 <- data.frame(pROC::coords(
  roc = roc_bagupf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_bagupf1 <- pr_cutoffs_bagupf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_bagupf1

cm_bagupf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_bag_up$Si >= best_cutoff_bagupf1$threshold, "Si", "No"),
                levels = levels(pred_cv_bag_up$obs)),
  reference = pred_cv_bag_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_bagupf1_pr

umbral_bagupf1 <- best_cutoff_bagupf1$threshold

prob_bag_up <- predict(bag_up, data = db_test)$predictions[, "Si"]

predictSample_bagupf1 <- db_test |>
  mutate(pobre_prob = prob_bag_up,
         pobre_lab = ifelse(pobre_prob >= umbral_bagupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_bagupf1 <- paste0("Bagging_up_F1_prob_", round(umbral_bagupf1, 3), ".csv")
write.csv(predictSample_bagupf1, name_bagupf1, row.names = FALSE)

# MODELO 13: RANDOM FOREST (RANGER VIA CARET) - SELECCION POR ROC, UMBRAL 0.5---


rf_grid <- expand.grid(
  mtry = c(2, 5, 7),
  min.node.size = c(1, 5, 10, 20, 40, 50),
  splitrule = "gini"
)

set.seed(2026)
rf_roc <- train(
  Pobre ~ .,
  data = db_train,
  method = "ranger",
  metric = "ROC",
  maximize = TRUE,
  trControl = ctrl_roc,
  tuneGrid = rf_grid,
  num.trees = 500
  # Sin importance: se calcula aparte, una sola vez, sobre el modelo ganador
)

rf_roc
rf_roc$bestTune

pred_cv_rf_roc <- rf_roc$pred |>
  filter(mtry == rf_roc$bestTune$mtry,
         min.node.size == rf_roc$bestTune$min.node.size,
         splitrule == rf_roc$bestTune$splitrule)

cm_rf_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_rf_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_rf_roc$obs)),
  reference = pred_cv_rf_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_rf_roc

# Importancia de variables: se recalcula una sola vez, fijando los
# hiperparámetros ganadores, sobre todo db_train (no durante el tuning).
set.seed(2026)
rf_roc_importancia <- ranger(
  formula = Pobre ~ .,
  data = db_train,
  probability = TRUE,
  splitrule = as.character(rf_roc$bestTune$splitrule),
  num.trees = 500,
  mtry = rf_roc$bestTune$mtry,
  min.node.size = rf_roc$bestTune$min.node.size,
  importance = "permutation"
)

importancia_rf_roc <- sort(rf_roc_importancia$variable.importance, decreasing = TRUE)
importancia_rf_roc

predictSample_rf_roc <- db_test |>
  mutate(pobre_prob = predict(rf_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_rf_roc, "RandomForest.csv", row.names = FALSE)

# MODELO 14: RANDOM FOREST - UMBRAL OPTIMO EN CURVA PR -------------------------


set.seed(2026)
rf_f1 <- train(
  Pobre ~ .,
  data = db_train,
  method = "ranger",
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_f1,
  tuneGrid = rf_grid,
  num.trees = 500
)

rf_f1
rf_f1$bestTune

pred_cv_rf_f1 <- rf_f1$pred |>
  filter(mtry == rf_f1$bestTune$mtry,
         min.node.size == rf_f1$bestTune$min.node.size,
         splitrule == rf_f1$bestTune$splitrule)

# Calibramos el umbral de clasificación maximizando F1 en la curva PR.

roc_rff1 <- pROC::roc(
  response  = pred_cv_rf_f1$obs,
  predictor = pred_cv_rf_f1$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_rff1 <- data.frame(pROC::coords(
  roc = roc_rff1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_rff1 <- pr_cutoffs_rff1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_rff1

cm_rff1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_rf_f1$Si >= best_cutoff_rff1$threshold, "Si", "No"),
                levels = levels(pred_cv_rf_f1$obs)),
  reference = pred_cv_rf_f1$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_rff1_pr

umbral_rff1 <- best_cutoff_rff1$threshold

predictSample_rff1 <- db_test |>
  mutate(pobre_prob = predict(rf_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_rff1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_rff1 <- paste0("RandomForest_F1_prob_", round(umbral_rff1, 3), ".csv")
write.csv(predictSample_rff1, name_rff1, row.names = FALSE)
# MODELO 15: RANDOM FOREST CON OVERSAMPLING SELECCION POR F, UMBRAL OPTIMO EN CURVA PR---


set.seed(2026)
rf_up <- train(
  Pobre ~ .,
  data = db_train,
  method = "ranger",
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_up_f1,
  tuneGrid = rf_grid,
  num.trees = 500
  # Sin importance: se calcula aparte, una sola vez, sobre el modelo ganador
)

rf_up
rf_up$bestTune

pred_cv_rf_up <- rf_up$pred |>
  filter(mtry == rf_up$bestTune$mtry,
         min.node.size == rf_up$bestTune$min.node.size,
         splitrule == rf_up$bestTune$splitrule)

# Umbral fijo 0.5 (para comparar contra el umbral calibrado)
cm_rfup05 <- confusionMatrix(
  data = factor(ifelse(pred_cv_rf_up$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_rf_up$obs)),
  reference = pred_cv_rf_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_rfup05

# Calibración del umbral maximizando F1 en la curva PR
roc_rfupf1 <- pROC::roc(
  response  = pred_cv_rf_up$obs,
  predictor = pred_cv_rf_up$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_rfupf1 <- data.frame(pROC::coords(
  roc = roc_rfupf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_rfupf1 <- pr_cutoffs_rfupf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_rfupf1

cm_rfupf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_rf_up$Si >= best_cutoff_rfupf1$threshold, "Si", "No"),
                levels = levels(pred_cv_rf_up$obs)),
  reference = pred_cv_rf_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_rfupf1_pr

# Importancia de variables: se recalcula una sola vez, fijando los
# hiperparámetros ganadores y los pesos de clase, sobre todo db_train.
set.seed(2026)
rf_up_importancia <- ranger(
  formula = Pobre ~ .,
  data = db_train,
  probability = TRUE,
  splitrule = as.character(rf_up$bestTune$splitrule),
  num.trees = 500,
  mtry = rf_up$bestTune$mtry,
  min.node.size = rf_up$bestTune$min.node.size,
  importance = "permutation",
  case.weights = pesos_bag_up
)

importancia_rf_up <- sort(rf_up_importancia$variable.importance, decreasing = TRUE)
importancia_rf_up

umbral_rfupf1 <- best_cutoff_rfupf1$threshold
# Comparación: con/sin rebalanceo, umbral 0.5 vs umbral óptimo PR
model_comparison_rf_up <- bind_rows(
  tibble(strategy = "RF - ROC - umbral 0.5", threshold = 0.5,
         precision = unname(cm_rf_roc$byClass["Precision"]),
         recall = unname(cm_rf_roc$byClass["Recall"]),
         F1 = unname(cm_rf_roc$byClass["F1"])),
  tibble(strategy = "RF - F1 - umbral óptimo PR", threshold = umbral_rff1,
         precision = unname(cm_rff1_pr$byClass["Precision"]),
         recall = unname(cm_rff1_pr$byClass["Recall"]),
         F1 = unname(cm_rff1_pr$byClass["F1"])),
  tibble(strategy = "RF - Oversampling - umbral 0.5", threshold = 0.5,
         precision = unname(cm_rfup05$byClass["Precision"]),
         recall = unname(cm_rfup05$byClass["Recall"]),
         F1 = unname(cm_rfup05$byClass["F1"])),
  tibble(strategy = "RF - Oversampling - umbral óptimo PR", threshold = umbral_rfupf1,
         precision = unname(cm_rfupf1_pr$byClass["Precision"]),
         recall = unname(cm_rfupf1_pr$byClass["Recall"]),
         F1 = unname(cm_rfupf1_pr$byClass["F1"]))
) |> arrange(desc(F1))

model_comparison_rf_up



predictSample_rfupf1 <- db_test |>
  mutate(pobre_prob = predict(rf_up, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_rfupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_rfupf1 <- paste0("RandomForest_up_F1_prob_", round(umbral_rfupf1, 3), ".csv")
write.csv(predictSample_rfupf1, name_rfupf1, row.names = FALSE)

# MODELO 16: ADABOOST.M1 - SELECCION POR ROC, UMBRAL 0.5------------------------

ada_grid <- expand.grid(
  mfinal = c(100, 200, 500),
  maxdepth = 1,
  coeflearn = c("Breiman", "Freund")
)

predictores_ada <- setdiff(names(db_train), "Pobre")

set.seed(2026)
ada_roc <- train(
  x = as.data.frame(db_train[, predictores_ada]),
  y = db_train$Pobre,
  method = "AdaBoost.M1",
  trControl = ctrl_roc,
  metric = "ROC",
  maximize = TRUE,
  tuneGrid = ada_grid
)

ada_roc
ada_roc$bestTune

pred_cv_ada_roc <- ada_roc$pred |>
  dplyr::filter(mfinal == ada_roc$bestTune$mfinal,
                maxdepth == ada_roc$bestTune$maxdepth,
                coeflearn == ada_roc$bestTune$coeflearn)

cm_ada_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_ada_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_ada_roc$obs)),
  reference = pred_cv_ada_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_ada_roc

predictSample_ada_roc <- db_test |>
  mutate(pobre_prob = predict(ada_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

write.csv(predictSample_ada_roc, "AdaBoost.csv", row.names = FALSE)


# MODELO 17: ADABOOST.M1 - SELECCION POR F, UMBRAL OPTIMO EN CURVA PR ----------
set.seed(2026)
ada_f1 <- train(
  x = as.data.frame(db_train[, predictores_ada]),
  y = db_train$Pobre,
  method = "AdaBoost.M1",
  trControl = ctrl_f1,
  metric = "F",
  maximize = TRUE,
  tuneGrid = ada_grid
)

ada_f1
ada_f1$bestTune

pred_cv_ada_f1 <- ada_f1$pred |>
  dplyr::filter(mfinal == ada_f1$bestTune$mfinal,
                maxdepth == ada_f1$bestTune$maxdepth,
                coeflearn == ada_f1$bestTune$coeflearn)

roc_adaf1 <- pROC::roc(
  response  = pred_cv_ada_f1$obs,
  predictor = pred_cv_ada_f1$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_adaf1 <- data.frame(pROC::coords(
  roc = roc_adaf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_adaf1 <- pr_cutoffs_adaf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_adaf1

cm_adaf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_ada_f1$Si >= best_cutoff_adaf1$threshold, "Si", "No"),
                levels = levels(pred_cv_ada_f1$obs)),
  reference = pred_cv_ada_f1$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_adaf1_pr

umbral_adaf1 <- best_cutoff_adaf1$threshold

predictSample_adaf1 <- db_test |>
  mutate(pobre_prob = predict(ada_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_adaf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_adaf1 <- paste0("AdaBoost_F1_prob_", round(umbral_adaf1, 3), ".csv")
write.csv(predictSample_adaf1, name_adaf1, row.names = FALSE)

# MODELO 18: ADABOOST.M1 CON OVERSAMPLING SELECCION POR F, UMBRAL OPTIMO EN CURVA PR

set.seed(2026)
ada_up <- train(
  x = as.data.frame(db_train[, predictores_ada]),
  y = db_train$Pobre,
  method = "AdaBoost.M1",
  trControl = ctrl_up_f1,
  metric = "F",
  maximize = TRUE,
  tuneGrid = ada_grid
)

ada_up
ada_up$bestTune

pred_cv_ada_up <- ada_up$pred |>
  dplyr::filter(mfinal == ada_up$bestTune$mfinal,
                maxdepth == ada_up$bestTune$maxdepth,
                coeflearn == ada_up$bestTune$coeflearn)

# Umbral fijo 0.5
cm_adaup05 <- confusionMatrix(
  data = factor(ifelse(pred_cv_ada_up$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_ada_up$obs)),
  reference = pred_cv_ada_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_adaup05

# Umbral óptimo en la curva PR
roc_adaupf1 <- pROC::roc(
  response  = pred_cv_ada_up$obs,
  predictor = pred_cv_ada_up$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_adaupf1 <- data.frame(pROC::coords(
  roc = roc_adaupf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_adaupf1 <- pr_cutoffs_adaupf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_adaupf1

umbral_adaupf1 <- best_cutoff_adaupf1$threshold

cm_adaupf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_ada_up$Si >= umbral_adaupf1, "Si", "No"),
                levels = levels(pred_cv_ada_up$obs)),
  reference = pred_cv_ada_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_adaupf1_pr

# Comparación con y sin rebalanceo
model_comparison_ada_up <- bind_rows(
  tibble(strategy = "Ada - ROC - umbral 0.5", threshold = 0.5,
         precision = unname(cm_ada_roc$byClass["Precision"]),
         recall = unname(cm_ada_roc$byClass["Recall"]),
         F1 = unname(cm_ada_roc$byClass["F1"])),
  tibble(strategy = "Ada - F1 - umbral óptimo PR", threshold = umbral_adaf1,
         precision = unname(cm_adaf1_pr$byClass["Precision"]),
         recall = unname(cm_adaf1_pr$byClass["Recall"]),
         F1 = unname(cm_adaf1_pr$byClass["F1"])),
  tibble(strategy = "Ada - Oversampling - umbral 0.5", threshold = 0.5,
         precision = unname(cm_adaup05$byClass["Precision"]),
         recall = unname(cm_adaup05$byClass["Recall"]),
         F1 = unname(cm_adaup05$byClass["F1"])),
  tibble(strategy = "Ada - Oversampling - umbral óptimo PR", threshold = umbral_adaupf1,
         precision = unname(cm_adaupf1_pr$byClass["Precision"]),
         recall = unname(cm_adaupf1_pr$byClass["Recall"]),
         F1 = unname(cm_adaupf1_pr$byClass["F1"]))
) |> arrange(desc(F1))

model_comparison_ada_up

predictSample_adaupf1 <- db_test |>
  mutate(pobre_prob = predict(ada_up, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_adaupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_adaupf1 <- paste0("AdaBoost_up_F1_prob_", round(umbral_adaupf1, 3), ".csv")
write.csv(predictSample_adaupf1, name_adaupf1, row.names = FALSE)

# MODELO 19: GRADIENT BOOSTING - SELECCION POR ROC, UMBRAL FIJO 0.5 -------------

grid_gbm <- expand.grid(n.trees = c(100, 200, 500),
                        interaction.depth = c(1, 3),
                        shrinkage = c(0.001, 0.01),
                        n.minobsinnode = c(20, 50))

set.seed(2026)
gbm_roc <- train(
  Pobre ~ .,
  data = db_train,
  method = "gbm",
  metric = "ROC",
  maximize = TRUE,
  trControl = ctrl_roc,
  tuneGrid = grid_gbm,
  verbose = FALSE
)

gbm_roc
gbm_roc$bestTune

pred_cv_gbm_roc <- gbm_roc$pred |>
  filter(n.trees == gbm_roc$bestTune$n.trees,
         interaction.depth == gbm_roc$bestTune$interaction.depth,
         shrinkage == gbm_roc$bestTune$shrinkage,
         n.minobsinnode == gbm_roc$bestTune$n.minobsinnode)

cm_gbm_roc <- confusionMatrix(
  data = factor(ifelse(pred_cv_gbm_roc$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_gbm_roc$obs)),
  reference = pred_cv_gbm_roc$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_gbm_roc

predictSample_gbm_roc <- db_test |>
  mutate(pobre_prob = predict(gbm_roc, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_gbm_roc <- paste0("GBM_trees_", gbm_roc$bestTune$n.trees,
                       "_depth_", gbm_roc$bestTune$interaction.depth,
                       "_shrink_", gbm_roc$bestTune$shrinkage,
                       "_minobs_", gbm_roc$bestTune$n.minobsinnode, ".csv")
write.csv(predictSample_gbm_roc, name_gbm_roc, row.names = FALSE)


# MODELO 20: GRADIENT BOOSTING - SELECCION POR F, UMBRAL OPTIMO EN CURVA PR -----

set.seed(2026)
gbm_f1 <- train(
  Pobre ~ .,
  data = db_train,
  method = "gbm",
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_f1,
  tuneGrid = grid_gbm,
  verbose = FALSE
)

gbm_f1
gbm_f1$bestTune

pred_cv_gbm_f1 <- gbm_f1$pred |>
  filter(n.trees == gbm_f1$bestTune$n.trees,
         interaction.depth == gbm_f1$bestTune$interaction.depth,
         shrinkage == gbm_f1$bestTune$shrinkage,
         n.minobsinnode == gbm_f1$bestTune$n.minobsinnode)

# Calibramos el umbral de clasificación maximizando F1 en la curva PR.

roc_gbmf1 <- pROC::roc(
  response  = pred_cv_gbm_f1$obs,
  predictor = pred_cv_gbm_f1$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_gbmf1 <- data.frame(pROC::coords(
  roc = roc_gbmf1,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_gbmf1 <- pr_cutoffs_gbmf1 |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_gbmf1

cm_gbmf1_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_gbm_f1$Si >= best_cutoff_gbmf1$threshold, "Si", "No"),
                levels = levels(pred_cv_gbm_f1$obs)),
  reference = pred_cv_gbm_f1$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_gbmf1_pr

umbral_gbmf1 <- best_cutoff_gbmf1$threshold

predictSample_gbmf1 <- db_test |>
  mutate(pobre_prob = predict(gbm_f1, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_gbmf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_gbmf1 <- paste0("GBM_F1_trees_", gbm_f1$bestTune$n.trees,
                     "_depth_", gbm_f1$bestTune$interaction.depth,
                     "_shrink_", gbm_f1$bestTune$shrinkage,
                     "_prob_", round(umbral_gbmf1, 3), ".csv")
write.csv(predictSample_gbmf1, name_gbmf1, row.names = FALSE)


# MODELO 21: GRADIENT BOOSTING CON OVERSAMPLING - SELECCION POR F, UMBRAL PR ----

set.seed(2026)
gbm_upF <- train(
  Pobre ~ .,
  data = db_train,
  method = "gbm",
  metric = "F",
  maximize = TRUE,
  trControl = ctrl_up_f1,
  tuneGrid = grid_gbm,
  verbose = FALSE
)

gbm_upF
gbm_upF$bestTune

pred_cv_gbm_up <- gbm_upF$pred |>
  filter(n.trees == gbm_upF$bestTune$n.trees,
         interaction.depth == gbm_upF$bestTune$interaction.depth,
         shrinkage == gbm_upF$bestTune$shrinkage,
         n.minobsinnode == gbm_upF$bestTune$n.minobsinnode)

# Verificación de colapso: sd(Si) debe ser claramente mayor que 0
summary(pred_cv_gbm_up$Si)
sd(pred_cv_gbm_up$Si)

roc_gbmupF <- pROC::roc(
  response  = pred_cv_gbm_up$obs,
  predictor = pred_cv_gbm_up$Si,
  levels = c("No", "Si"),
  direction = "<"
)

pr_cutoffs_gbmupF <- data.frame(pROC::coords(
  roc = roc_gbmupF,
  x = seq(0, 1, length = 100),
  ret = c('threshold', 'precision', 'recall')
)) |>
  mutate(F1 = (2 * precision * recall) / (precision + recall))

best_cutoff_gbmupF <- pr_cutoffs_gbmupF |> slice_max(F1, n = 1, with_ties = FALSE)
best_cutoff_gbmupF

cm_gbmup05 <- confusionMatrix(
  data = factor(ifelse(pred_cv_gbm_up$Si >= 0.5, "Si", "No"),
                levels = levels(pred_cv_gbm_up$obs)),
  reference = pred_cv_gbm_up$obs,
  positive = "Si",
  mode = "prec_recall"
)

cm_gbmupF_pr <- confusionMatrix(
  data = factor(ifelse(pred_cv_gbm_up$Si >= best_cutoff_gbmupF$threshold, "Si", "No"),
                levels = levels(pred_cv_gbm_up$obs)),
  reference = pred_cv_gbm_up$obs,
  positive = "Si",
  mode = "prec_recall"
)
cm_gbmupF_pr

umbral_gbmupF <- best_cutoff_gbmupF$threshold

predictSample_gbmupF <- db_test |>
  mutate(pobre_prob = predict(gbm_upF, newdata = db_test, type = "prob")[["Si"]],
         pobre_lab = ifelse(pobre_prob >= umbral_gbmupF, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_gbmupF <- paste0("GBM_up_F1_trees_", gbm_upF$bestTune$n.trees,
                      "_depth_", gbm_upF$bestTune$interaction.depth,
                      "_shrink_", gbm_upF$bestTune$shrinkage,
                      "_prob_", round(umbral_gbmupF, 3), ".csv")
write.csv(predictSample_gbmupF, name_gbmupF, row.names = FALSE)


# MODELOS 22 A 27: XGBOOST Y LIGHTGBM ------------------------------------------
#
# Variantes de cada algoritmo, igual que en los demas modelos:
#   - Seleccion por ROC, umbral fijo 0.5            (modelos 22 y 25)
#   - Seleccion por F, umbral optimo en curva PR     (modelos 23 y 26)
#   - Oversampling, seleccion por F, umbral 0.5 y PR (modelos 24 y 27)
#
# La validacion cruzada se programa directamente con xgboost y lightgbm (sin
# caret) para que funcione con cualquier version de xgboost: method =
# "xgbTree" de caret falla con xgboost 3.x, que es la version actual de CRAN
# y la unica disponible para R 4.6. El protocolo es el mismo de caret:
# mismos folds, F de la clase "Si" (f1_si, umbral 0.5) y ROC calculados por
# fold y promediados, y predicciones fuera de muestra para calibrar el umbral.
#
# Las variantes por ROC y por F usan la misma grilla y los mismos folds, asi
# que comparten una sola validacion cruzada y solo difieren en la metrica con
# la que se elige la configuracion. El oversampling (como sampling = "up" de
# caret) repite al azar hogares pobres solo en los folds de entrenamiento,
# hasta igualar a los no pobres; la validacion se hace sobre los hogares
# originales.
#


# 22-27.1 MATRIZ DE PREDICTORES Y FUNCIONES COMUNES ---------------------------------

# xgboost y lightgbm exigen una matriz numerica: los factores se codifican
# como indicadoras. La formula no incluye Pobre para aplicarla tambien a test.
predictores_boost <- setdiff(names(db_train), "Pobre")
fml_boost <- reformulate(predictores_boost)

x_train_boost <- model.matrix(fml_boost, data = db_train)[, -1, drop = FALSE]
x_test_boost  <- model.matrix(fml_boost, data = db_test)[, -1, drop = FALSE]
colnames(x_train_boost) <- make.names(colnames(x_train_boost), unique = TRUE)
colnames(x_test_boost)  <- make.names(colnames(x_test_boost), unique = TRUE)

stopifnot(identical(colnames(x_train_boost), colnames(x_test_boost)),
          nrow(x_train_boost) == nrow(db_train),
          nrow(x_test_boost) == nrow(db_test))

# Respuesta 0/1 (1 = "Si")
y_train_boost <- as.integer(db_train$Pobre == "Si")


fun_boost_ajustar <- function(.x, .y, .par, .algoritmo) {
  #' Ajustar un modelo XGBoost o LightGBM con una combinacion de hiperparametros
  #'
  #' @param .x \code{matrix}. Matriz de predictores.
  #' @param .y \code{integer}. Respuesta 0/1 (1 = "Si").
  #' @param .par \code{data.frame} de una fila con los hiperparametros.
  #' @param .algoritmo \code{character}. "xgb" o "lgbm".
  #'
  #' @returns Objeto \code{xgb.Booster} o \code{lgb.Booster}.
  set.seed(2026)
  if (.algoritmo == "xgb") {
    xgb.train(
      params = list(objective = "binary:logistic",
                    eval_metric = "logloss",
                    eta = .par$eta,
                    max_depth = .par$max_depth,
                    min_child_weight = .par$min_child_weight,
                    colsample_bytree = .par$colsample_bytree,
                    subsample = .par$subsample,
                    gamma = 0,
                    tree_method = "hist",
                    nthread = 2),
      data = xgb.DMatrix(.x, label = .y),
      nrounds = .par$nrounds,
      verbose = 0
    )
  } else {
    lgb.train(
      params = list(objective = "binary",
                    learning_rate = .par$learning_rate,
                    num_leaves = .par$num_leaves,
                    min_data_in_leaf = .par$min_data_in_leaf,
                    feature_fraction = .par$feature_fraction,
                    bagging_fraction = 0.8,
                    bagging_freq = 1,
                    lambda_l2 = .par$lambda_l2,
                    num_threads = 2,
                    seed = 2026,
                    verbose = -1),
      data = lgb.Dataset(.x, label = .y),
      nrounds = .par$nrounds,
      verbose = -1
    )
  }
}


fun_boost_predecir <- function(.modelo, .x, .algoritmo) {
  #' Probabilidad de la clase "Si"
  if (.algoritmo == "xgb") predict(.modelo, xgb.DMatrix(.x)) else predict(.modelo, .x)
}


fun_oversampling <- function(.idx, .y) {
  #' Indices con la clase "Si" sobremuestreada hasta igualar a la clase "No"
  #'
  #' @description Equivale a sampling = "up" de caret: se conservan todos los
  #' hogares y se agregan copias al azar (con reemplazo) de hogares pobres.
  idx_si <- .idx[.y[.idx] == 1]
  idx_no <- .idx[.y[.idx] == 0]
  set.seed(2026)
  c(idx_no, idx_si, sample(idx_si, length(idx_no) - length(idx_si), replace = TRUE))
}


fun_cv_boost <- function(.grid, .algoritmo, .oversampling = FALSE) {
  #' Validacion cruzada con los folds comunes
  #'
  #' @param .grid \code{data.frame}. Una fila por combinacion de hiperparametros.
  #' @param .algoritmo \code{character}. "xgb" o "lgbm".
  #' @param .oversampling \code{logical}. Sobremuestrear "Si" en entrenamiento.
  #'
  #' @returns \code{list} con \code{pred} (predicciones fuera de muestra de
  #' cada configuracion: config, Resample, rowIndex, obs, Si) y
  #' \code{resultados} (ROC y F de "Si" con umbral 0.5, promedio de los
  #' folds, por configuracion).
  grid <- .grid |> mutate(config = row_number())
  lista_pred <- list()
  for (i in seq_len(nrow(grid))) {
    for (k in seq_along(folds)) {
      idx_tr <- folds[[k]]
      idx_va <- setdiff(seq_len(nrow(x_train_boost)), idx_tr)
      if (.oversampling) idx_tr <- fun_oversampling(idx_tr, y_train_boost)
      modelo_ik <- fun_boost_ajustar(x_train_boost[idx_tr, , drop = FALSE],
                                     y_train_boost[idx_tr], grid[i, ], .algoritmo)
      lista_pred[[length(lista_pred) + 1]] <- tibble(
        config   = grid$config[i],
        Resample = names(folds)[k],
        rowIndex = idx_va,
        obs      = db_train$Pobre[idx_va],
        Si       = fun_boost_predecir(modelo_ik, x_train_boost[idx_va, , drop = FALSE],
                                      .algoritmo)
      )
    }
    cat(.algoritmo, if (.oversampling) "oversampling" else "", ": configuracion",
        i, "de", nrow(grid), "lista\n")
  }
  pred <- bind_rows(lista_pred)

  resultados <- pred |>
    mutate(pred = factor(ifelse(Si >= 0.5, "Si", "No"), levels = c("No", "Si"))) |>
    group_by(config, Resample) |>
    summarize(
      ROC = as.numeric(pROC::auc(pROC::roc(obs, Si, levels = c("No", "Si"),
                                           direction = "<", quiet = TRUE))),
      F = unname(f1_si(data.frame(obs = obs, pred = pred))["F"]),
      .groups = "drop"
    ) |>
    group_by(config) |>
    summarize(ROC = mean(ROC), F = mean(F), .groups = "drop") |>
    left_join(grid, by = "config")

  list(pred = pred, resultados = resultados)
}


fun_umbral_pr <- function(.pred_cv) {
  #' Umbral que maximiza el F1 en la curva PR (predicciones fuera de muestra)
  roc_obj <- pROC::roc(
    response  = .pred_cv$obs,
    predictor = .pred_cv$Si,
    levels = c("No", "Si"),
    direction = "<"
  )
  data.frame(pROC::coords(
    roc = roc_obj,
    x = seq(0, 1, length = 100),
    ret = c('threshold', 'precision', 'recall')
  )) |>
    mutate(F1 = (2 * precision * recall) / (precision + recall)) |>
    slice_max(F1, n = 1, with_ties = FALSE)
}


fun_cm <- function(.pred_cv, .umbral) {
  #' Matriz de confusion de las predicciones fuera de muestra con un umbral
  confusionMatrix(
    data = factor(ifelse(.pred_cv$Si >= .umbral, "Si", "No"),
                  levels = levels(.pred_cv$obs)),
    reference = .pred_cv$obs,
    positive = "Si",
    mode = "prec_recall"
  )
}


fun_fila_cm <- function(.cm, .strategy, .threshold) {
  #' Fila de la tabla de comparacion
  tibble(strategy = .strategy,
         threshold = .threshold,
         precision = unname(.cm$byClass["Precision"]),
         recall = unname(.cm$byClass["Recall"]),
         F1 = unname(.cm$byClass["F1"]))
}


# 22-27.2 GRILLAS -------------------------------------------------------------------

grid_xgb <- expand.grid(nrounds = 700,
                        max_depth = c(4, 6),
                        eta = 0.05,
                        min_child_weight = c(10, 25),
                        colsample_bytree = c(0.6, 0.8),
                        subsample = 0.8)

grid_lgbm <- tibble(
  num_leaves       = c(15, 31, 63),
  learning_rate    = c(0.05, 0.02, 0.03),
  nrounds          = c(600, 1500, 900),
  min_data_in_leaf = c(20, 50, 30),
  feature_fraction = c(0.8, 0.5, 0.6),
  lambda_l2        = c(0, 5, 0)
)


# 22-27.3 VALIDACION CRUZADA --------------------------------------------------------

# Sin rebalanceo: comun a los modelos por ROC y por F
cv_xgb  <- fun_cv_boost(grid_xgb,  "xgb")
cv_lgbm <- fun_cv_boost(grid_lgbm, "lgbm")

# Con oversampling
cv_xgb_up  <- fun_cv_boost(grid_xgb,  "xgb",  .oversampling = TRUE)
cv_lgbm_up <- fun_cv_boost(grid_lgbm, "lgbm", .oversampling = TRUE)

cv_xgb$resultados
cv_lgbm$resultados
cv_xgb_up$resultados
cv_lgbm_up$resultados


# MODELO 22: XGBOOST - SELECCION POR ROC, UMBRAL FIJO 0.5 ---------------------

best_tune_xgb_roc <- cv_xgb$resultados |> slice_max(ROC, n = 1, with_ties = FALSE)
best_tune_xgb_roc

pred_cv_xgb_roc <- cv_xgb$pred |> filter(config == best_tune_xgb_roc$config)

cm_xgb_roc <- fun_cm(pred_cv_xgb_roc, 0.5)
cm_xgb_roc

# Modelo final con la configuracion ganadora sobre todo train
xgb_roc <- fun_boost_ajustar(x_train_boost, y_train_boost, best_tune_xgb_roc, "xgb")

predictSample_xgb_roc <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(xgb_roc, x_test_boost, "xgb"),
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_xgb_roc <- paste0("XGB_depth_", best_tune_xgb_roc$max_depth,
                       "_mcw_", best_tune_xgb_roc$min_child_weight,
                       "_colsample_", best_tune_xgb_roc$colsample_bytree, ".csv")
write.csv(predictSample_xgb_roc, name_xgb_roc, row.names = FALSE)


# MODELO 23: XGBOOST - SELECCION POR F, UMBRAL OPTIMO EN CURVA PR -------------

best_tune_xgb_f1 <- cv_xgb$resultados |> slice_max(F, n = 1, with_ties = FALSE)
best_tune_xgb_f1

pred_cv_xgb_f1 <- cv_xgb$pred |> filter(config == best_tune_xgb_f1$config)

# Calibramos el umbral de clasificación maximizando F1 en la curva PR.
best_cutoff_xgbf1 <- fun_umbral_pr(pred_cv_xgb_f1)
best_cutoff_xgbf1

umbral_xgbf1 <- best_cutoff_xgbf1$threshold

cm_xgbf1_pr <- fun_cm(pred_cv_xgb_f1, umbral_xgbf1)
cm_xgbf1_pr

xgb_f1 <- fun_boost_ajustar(x_train_boost, y_train_boost, best_tune_xgb_f1, "xgb")

predictSample_xgbf1 <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(xgb_f1, x_test_boost, "xgb"),
         pobre_lab = ifelse(pobre_prob >= umbral_xgbf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_xgbf1 <- paste0("XGB_F1_depth_", best_tune_xgb_f1$max_depth,
                     "_mcw_", best_tune_xgb_f1$min_child_weight,
                     "_prob_", round(umbral_xgbf1, 3), ".csv")
write.csv(predictSample_xgbf1, name_xgbf1, row.names = FALSE)


# MODELO 24: XGBOOST CON OVERSAMPLING - SELECCION POR F, UMBRAL PR -----------

best_tune_xgb_up <- cv_xgb_up$resultados |> slice_max(F, n = 1, with_ties = FALSE)
best_tune_xgb_up

pred_cv_xgb_up <- cv_xgb_up$pred |> filter(config == best_tune_xgb_up$config)

# Verificación de colapso: sd(Si) debe ser claramente mayor que 0
summary(pred_cv_xgb_up$Si)
sd(pred_cv_xgb_up$Si)

# Umbral fijo 0.5 (para comparar contra el umbral calibrado)
cm_xgbup05 <- fun_cm(pred_cv_xgb_up, 0.5)
cm_xgbup05

best_cutoff_xgbupf1 <- fun_umbral_pr(pred_cv_xgb_up)
best_cutoff_xgbupf1

umbral_xgbupf1 <- best_cutoff_xgbupf1$threshold

cm_xgbupf1_pr <- fun_cm(pred_cv_xgb_up, umbral_xgbupf1)
cm_xgbupf1_pr

# Modelo final: oversampling sobre todo train
idx_up_full <- fun_oversampling(seq_len(nrow(x_train_boost)), y_train_boost)
xgb_up <- fun_boost_ajustar(x_train_boost[idx_up_full, , drop = FALSE],
                            y_train_boost[idx_up_full], best_tune_xgb_up, "xgb")

predictSample_xgbupf1 <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(xgb_up, x_test_boost, "xgb"),
         pobre_lab = ifelse(pobre_prob >= umbral_xgbupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_xgbupf1 <- paste0("XGB_up_F1_depth_", best_tune_xgb_up$max_depth,
                       "_mcw_", best_tune_xgb_up$min_child_weight,
                       "_prob_", round(umbral_xgbupf1, 3), ".csv")
write.csv(predictSample_xgbupf1, name_xgbupf1, row.names = FALSE)

# Comparación: con/sin rebalanceo, umbral 0.5 vs umbral óptimo PR
model_comparison_xgb <- bind_rows(
  fun_fila_cm(cm_xgb_roc,    "XGB - ROC - umbral 0.5",                0.5),
  fun_fila_cm(cm_xgbf1_pr,   "XGB - F1 - umbral óptimo PR",           umbral_xgbf1),
  fun_fila_cm(cm_xgbup05,    "XGB - Oversampling - umbral 0.5",       0.5),
  fun_fila_cm(cm_xgbupf1_pr, "XGB - Oversampling - umbral óptimo PR", umbral_xgbupf1)
) |> arrange(desc(F1))

model_comparison_xgb


# MODELO 25: LIGHTGBM - SELECCION POR ROC, UMBRAL FIJO 0.5 --------------------

best_tune_lgbm_roc <- cv_lgbm$resultados |> slice_max(ROC, n = 1, with_ties = FALSE)
best_tune_lgbm_roc

pred_cv_lgbm_roc <- cv_lgbm$pred |> filter(config == best_tune_lgbm_roc$config)

cm_lgbm_roc <- fun_cm(pred_cv_lgbm_roc, 0.5)
cm_lgbm_roc

lgbm_roc <- fun_boost_ajustar(x_train_boost, y_train_boost, best_tune_lgbm_roc, "lgbm")

predictSample_lgbm_roc <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(lgbm_roc, x_test_boost, "lgbm"),
         pobre_lab = ifelse(pobre_prob >= 0.5, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_lgbm_roc <- paste0("LGBM_leaves_", best_tune_lgbm_roc$num_leaves,
                        "_lr_", best_tune_lgbm_roc$learning_rate,
                        "_trees_", best_tune_lgbm_roc$nrounds, ".csv")
write.csv(predictSample_lgbm_roc, name_lgbm_roc, row.names = FALSE)


# MODELO 26: LIGHTGBM - SELECCION POR F, UMBRAL OPTIMO EN CURVA PR ------------

best_tune_lgbm_f1 <- cv_lgbm$resultados |> slice_max(F, n = 1, with_ties = FALSE)
best_tune_lgbm_f1

pred_cv_lgbm_f1 <- cv_lgbm$pred |> filter(config == best_tune_lgbm_f1$config)

# Calibramos el umbral de clasificación maximizando F1 en la curva PR.
best_cutoff_lgbmf1 <- fun_umbral_pr(pred_cv_lgbm_f1)
best_cutoff_lgbmf1

umbral_lgbmf1 <- best_cutoff_lgbmf1$threshold

cm_lgbmf1_pr <- fun_cm(pred_cv_lgbm_f1, umbral_lgbmf1)
cm_lgbmf1_pr

lgbm_f1 <- fun_boost_ajustar(x_train_boost, y_train_boost, best_tune_lgbm_f1, "lgbm")

predictSample_lgbmf1 <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(lgbm_f1, x_test_boost, "lgbm"),
         pobre_lab = ifelse(pobre_prob >= umbral_lgbmf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_lgbmf1 <- paste0("LGBM_F1_leaves_", best_tune_lgbm_f1$num_leaves,
                      "_lr_", best_tune_lgbm_f1$learning_rate,
                      "_prob_", round(umbral_lgbmf1, 3), ".csv")
write.csv(predictSample_lgbmf1, name_lgbmf1, row.names = FALSE)


# MODELO 27: LIGHTGBM CON OVERSAMPLING - SELECCION POR F, UMBRAL PR ----------

best_tune_lgbm_up <- cv_lgbm_up$resultados |> slice_max(F, n = 1, with_ties = FALSE)
best_tune_lgbm_up

pred_cv_lgbm_up <- cv_lgbm_up$pred |> filter(config == best_tune_lgbm_up$config)

# Verificación de colapso: sd(Si) debe ser claramente mayor que 0
summary(pred_cv_lgbm_up$Si)
sd(pred_cv_lgbm_up$Si)

# Umbral fijo 0.5 (para comparar contra el umbral calibrado)
cm_lgbmup05 <- fun_cm(pred_cv_lgbm_up, 0.5)
cm_lgbmup05

best_cutoff_lgbmupf1 <- fun_umbral_pr(pred_cv_lgbm_up)
best_cutoff_lgbmupf1

umbral_lgbmupf1 <- best_cutoff_lgbmupf1$threshold

cm_lgbmupf1_pr <- fun_cm(pred_cv_lgbm_up, umbral_lgbmupf1)
cm_lgbmupf1_pr

# Modelo final: oversampling sobre todo train
lgbm_up <- fun_boost_ajustar(x_train_boost[idx_up_full, , drop = FALSE],
                             y_train_boost[idx_up_full], best_tune_lgbm_up, "lgbm")

predictSample_lgbmupf1 <- db_test |>
  mutate(pobre_prob = fun_boost_predecir(lgbm_up, x_test_boost, "lgbm"),
         pobre_lab = ifelse(pobre_prob >= umbral_lgbmupf1, "Si", "No"),
         pobre = if_else(pobre_lab == "Si", 1L, 0L)) |>
  select(id, pobre)

name_lgbmupf1 <- paste0("LGBM_up_F1_leaves_", best_tune_lgbm_up$num_leaves,
                        "_lr_", best_tune_lgbm_up$learning_rate,
                        "_prob_", round(umbral_lgbmupf1, 3), ".csv")
write.csv(predictSample_lgbmupf1, name_lgbmupf1, row.names = FALSE)

# Comparación: con/sin rebalanceo, umbral 0.5 vs umbral óptimo PR
model_comparison_lgbm <- bind_rows(
  fun_fila_cm(cm_lgbm_roc,    "LGBM - ROC - umbral 0.5",                0.5),
  fun_fila_cm(cm_lgbmf1_pr,   "LGBM - F1 - umbral óptimo PR",           umbral_lgbmf1),
  fun_fila_cm(cm_lgbmup05,    "LGBM - Oversampling - umbral 0.5",       0.5),
  fun_fila_cm(cm_lgbmupf1_pr, "LGBM - Oversampling - umbral óptimo PR", umbral_lgbmupf1)
) |> arrange(desc(F1))

model_comparison_lgbm
