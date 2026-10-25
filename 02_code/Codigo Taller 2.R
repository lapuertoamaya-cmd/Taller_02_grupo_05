################################################################################
# TALLER 2 - PREDICCION DE POBREZA EN HOGARES (GEIH)
# Big Data y Machine Learning

################################################################################

# 0. LIBRERIAS Y LECTURA DE DATOS ---------------------------------------------

#' Carga de paquetes
#'
#' @description Se cargan los paquetes necesarios para todo el script:
#' manejo de datos (tidyverse), modelos de regularizacion (glmnet), entrenamiento
#' y validacion cruzada (caret), metricas adicionales (MLmetrics), Naive Bayes
#' (naivebayes), curvas ROC/PR (pROC) y documentacion de funciones (docstring).
# 0. LIBRERIAS Y LECTURA DE DATOS ---------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(glmnet, MLmetrics, naivebayes, pROC, docstring,ranger,# Para bagging y random forest.
               randomForest, # Para random forest.
               tidyverse,    # Manipulación de datos.
               caret,        # Entrenamiento del modelo y selección de hiperparámetros.
               Metrics,      # Métricas de evaluación de los problemas de clasificación.
               adabag,       # AdaBoost 
               gbm,          # Gradient Boosting Models. 
               lightgbm,     # LightGBM
               xgboost,      # XGBoost
               conflicted, adabag, fastAdaboost)
conflicted::conflicts_prefer(dplyr::filter)

setwd("C:/Users/aeste/OneDrive/Documentos/10 Semestre/Big Data ML/Taller 2")

# Bases originales.
raw_train_hogares  <- read.csv("train_hogares.csv")
raw_train_personas <- read.csv("train_personas.csv")
raw_test_hogares   <- read.csv("test_hogares.csv")
raw_test_personas  <- read.csv("test_personas.csv")

db_train_hogares  <- raw_train_hogares
db_train_personas <- raw_train_personas
db_test_hogares   <- raw_test_hogares
db_test_personas  <- raw_test_personas

# Se excluyen las variables de ingreso (filtran la respuesta) y los factores
# de expansion. Lp y Dominio se conservan porque estan en test y definen la
# linea de pobreza de cada hogar.
db_train_hogares <- db_train_hogares |>
  select(-Ingtotug, -Ingtotugarr, -Ingpcug, -Li,
         -Indigente, -Npobres, -Nindigentes, -Fex_c, -Fex_dpto, -Depto)


# 1. FUNCIONES DE PREPROCESAMIENTO --------------------------------------------

fun_preprocess_personas <- function(.db) {
  #' Preprocesar base de datos de personas
  #'
  #' @description \code{fun_preprocess_personas()} retorna una base de datos con
  #' nombres descriptivos, variables binarias y etiquetas en las variables
  #' categóricas, lista para colapsar a nivel de hogar. La población en edad de
  #' trabajar se toma de la variable \code{Pet} del DANE.
  #'
  #' @param .db \code{tibble}. Base de datos descargada de Kaggle a nivel de
  #' personas (train o test).
  #'
  #' @returns \code{tibble}. Base de datos a nivel de persona. Incluye el id del
  #' hogar, \code{id}, el id de la persona, \code{id_persona}, la edad,
  #' \code{num_age}, indicadoras de sexo, jefatura, pertenencia a la PET,
  #' ocupación, desocupación e inactividad, el nivel educativo,
  #' \code{cat_educ}, los años de educación, \code{num_educ_anios}, e
  #' indicadoras de formalidad, subempleo, beneficios laborales, protección
  #' social y recepción de ingresos no laborales.
  text_vars <- c('id', 'id_persona' = 'Orden', 'num_age' = 'P6040',
                 'cat_gender' = 'P6020', 'cat_relationship' = 'P6050',
                 'bin_pet' = 'Pet', 'bin_employed' = 'Oc',
                 'bin_desocupado' = 'Des', 'bin_inactivo' = 'Ina',
                 'cat_educ' = 'P6210', 'num_grado' = 'P6210s1',
                 'cat_salud_afil' = 'P6090', 'cat_regimen' = 'P6100',
                 'cat_posicion' = 'P6430', 'bin_cotiza' = 'P6920',
                 'num_horas' = 'P6800', 'bin_subempleo' = 'P7090',
                 'bin_pension' = 'P7495', 'bin_ayuda_hogar' = 'P7500s2',
                 'bin_ayuda_inst' = 'P7500s3', 'bin_remesa' = 'P7505',
                 'ben_extras' = 'P6510', 'ben_primas' = 'P6545',
                 'ben_alim' = 'P6580', 'ben_transp' = 'P6585s1',
                 'ben_familiar' = 'P6585s2', 'ben_subsidio' = 'P6585s3',
                 'ben_educ' = 'P6585s4')
  
  db_result <- .db |>
    select(all_of(text_vars)) |>
    mutate(
      # Sexo, jefatura y estructura por edad
      bin_male = ifelse(test = cat_gender == 1, yes = 1, no = 0),
      bin_head = ifelse(test = cat_relationship == 1, yes = 1, no = 0),
      bin_menor6 = ifelse(test = num_age < 6, yes = 1, no = 0),
      bin_minor = ifelse(test = num_age < 18, yes = 1, no = 0),
      bin_old = ifelse(test = num_age > 65, yes = 1, no = 0),
      
      # Condición de actividad (NA = no aplica)
      bin_pet = ifelse(test = is.na(bin_pet), yes = 0, no = 1),
      bin_occup = ifelse(test = is.na(bin_employed), yes = 0, no = 1),
      bin_desocupado = ifelse(test = is.na(bin_desocupado), yes = 0, no = 1),
      bin_inactivo = ifelse(test = is.na(bin_inactivo), yes = 0, no = 1),
      
      # Educación: NA y 'no sabe' se imputan como 'Ninguno'
      cat_educ = case_when(is.na(cat_educ) ~ 1,
                           cat_educ == 9 ~ 1,
                           TRUE ~ cat_educ),
      # Años de educación aproximados: años acumulados por nivel + grado
      num_educ_anios = case_when(
        cat_educ == 1 ~ 0,
        cat_educ == 2 ~ 0,
        cat_educ == 3 ~ pmin(replace_na(num_grado, 0), 5),
        cat_educ == 4 ~ 5 + pmin(replace_na(num_grado, 0), 4),
        cat_educ == 5 ~ 9 + pmin(replace_na(num_grado, 0), 2),
        cat_educ == 6 ~ 11 + pmin(replace_na(num_grado, 0), 6),
        TRUE ~ 0),
      cat_educ = factor(x = cat_educ, levels = 1:6, ordered = TRUE,
                        labels = c("Ninguno", "Preescolar", "Primaria",
                                   "Secundaria", "Media", "Universitaria")),
      bin_superior = ifelse(test = cat_educ == "Universitaria", yes = 1, no = 0),
      
      # Calidad del empleo (solo aplica a ocupados)
      bin_cotiza = ifelse(test = bin_cotiza %in% 1 & bin_occup == 1, yes = 1, no = 0),
      bin_asalariado = ifelse(test = cat_posicion %in% c(1, 2) & bin_occup == 1,
                              yes = 1, no = 0),
      bin_domestico = ifelse(test = cat_posicion %in% 3 & bin_occup == 1,
                             yes = 1, no = 0),
      bin_jornalero = ifelse(test = cat_posicion %in% 8 & bin_occup == 1,
                             yes = 1, no = 0),
      bin_cuenta_propia = ifelse(test = cat_posicion %in% c(4, 6) & bin_occup == 1,
                                 yes = 1, no = 0),
      bin_subempleo = ifelse(test = bin_subempleo %in% 1 & bin_occup == 1,
                             yes = 1, no = 0),
      num_horas = ifelse(test = bin_occup == 1, yes = replace_na(num_horas, 0), no = 0),
      
      # Índice de beneficios laborales (cada columna vale 1 si recibe)
      num_beneficios = (ben_extras %in% 1) + (ben_primas %in% 1) + (ben_alim %in% 1) +
        (ben_transp %in% 1) + (ben_familiar %in% 1) + (ben_subsidio %in% 1) +
        (ben_educ %in% 1),
      num_beneficios = ifelse(test = bin_occup == 1, yes = num_beneficios, no = 0),
      
      # Protección social
      bin_subsidiado = ifelse(test = cat_regimen %in% 3, yes = 1, no = 0),
      bin_sin_salud = ifelse(test = cat_salud_afil %in% 2, yes = 1, no = 0),
      
      # Ingresos no laborales (indicadoras)
      bin_pension = ifelse(test = bin_pension %in% 1, yes = 1, no = 0),
      bin_ayuda = ifelse(test = bin_ayuda_hogar %in% 1 | bin_ayuda_inst %in% 1 |
                           bin_remesa %in% 1, yes = 1, no = 0)
    ) |>
    select(id, id_persona, num_age, bin_male, bin_head, bin_pet, bin_menor6,
           bin_minor, bin_old, bin_occup, bin_desocupado, bin_inactivo,
           cat_educ, num_educ_anios, bin_superior, bin_cotiza, bin_asalariado,
           bin_domestico, bin_jornalero, bin_cuenta_propia, bin_subempleo,
           num_horas, num_beneficios, bin_subsidiado, bin_sin_salud,
           bin_pension, bin_ayuda)
  
  return(db_result)
}


fun_colapse_personas <- function(.db) {
  #' Colapsar a nivel hogar las observaciones de personas
  #'
  #' @description \code{fun_colapse_personas()} retorna una base de datos con
  #' variables a nivel de hogar construidas a partir de las observaciones de
  #' personas, agrupadas en cinco bloques: ocupación, capital humano,
  #' demografía y dependencia, calidad del empleo, y protección social. Los
  #' indicadores de actividad y capital humano se calculan sobre la población en
  #' edad de trabajar (PET) definida por el DANE.
  #'
  #' @param .db \code{tibble}. Base de datos a nivel de personas preprocesada
  #' por \code{fun_preprocess_personas()}.
  #'
  #' @returns \code{tibble}. Una fila por hogar. Incluye el id del hogar,
  #' \code{id}, y las variables: \code{num_personas}, \code{num_pet},
  #' \code{prop_ocupados_pet}, \code{bin_sin_ocupados}, \code{num_desocupados},
  #' \code{prop_inactividad}, \code{cat_maxEduc}, \code{num_educ_max},
  #' \code{num_educ_prom_pet}, \code{prop_superior}, \code{num_educ_jefe},
  #' \code{num_minors}, \code{num_menores6}, \code{num_old},
  #' \code{prop_dependencia}, \code{tasa_dependencia_ocup},
  #' \code{num_edad_prom_pet}, \code{bin_head_male}, \code{prop_mujeres_ocup},
  #' \code{prop_cotiza}, \code{prop_asalariados},
  #' \code{prop_cuenta_propia}, \code{bin_hay_subempleo}, \code{num_horas_prom},
  #' \code{num_beneficios_prom}, \code{prop_subsidiado}, \code{prop_sin_salud},
  #' \code{bin_recibe_pension} y \code{bin_recibe_ayuda}.
  db_result <- .db |>
    group_by(id) |>
    summarize(
      # Tamaño y ocupación
      num_personas = n(),
      num_pet = sum(bin_pet, na.rm = TRUE),
      num_no_pet = sum(1 - bin_pet, na.rm = TRUE),
      num_ocupados = sum(bin_occup, na.rm = TRUE),
      num_desocupados = sum(bin_desocupado, na.rm = TRUE),
      num_inactivos = sum(bin_inactivo, na.rm = TRUE),
      
      # Capital humano
      cat_maxEduc = max(cat_educ, na.rm = TRUE),
      num_educ_max = max(num_educ_anios, na.rm = TRUE),
      num_educ_prom_pet = mean(num_educ_anios[bin_pet == 1], na.rm = TRUE),
      num_superior_pet = sum(bin_superior * bin_pet, na.rm = TRUE),
      num_educ_jefe = max(num_educ_anios * bin_head, na.rm = TRUE),
      
      # Demografía
      num_minors = sum(bin_minor, na.rm = TRUE),
      num_menores6 = sum(bin_menor6, na.rm = TRUE),
      num_old = sum(bin_old, na.rm = TRUE),
      num_edad_prom_pet = mean(num_age[bin_pet == 1], na.rm = TRUE),
      
      # Género y jefatura
      bin_head_male = max(bin_head * bin_male, na.rm = TRUE),
      num_mujeres_ocup = sum((1 - bin_male) * bin_occup, na.rm = TRUE),
      num_jefes = sum(bin_head, na.rm = TRUE),
      
      # Calidad del empleo
      num_cotizan = sum(bin_cotiza, na.rm = TRUE),
      num_asalariados = sum(bin_asalariado, na.rm = TRUE),
      num_domesticos = sum(bin_domestico, na.rm = TRUE),
      num_jornaleros = sum(bin_jornalero, na.rm = TRUE),
      num_cuenta_propia = sum(bin_cuenta_propia, na.rm = TRUE),
      bin_hay_subempleo = max(bin_subempleo, na.rm = TRUE),
      num_horas_total = sum(num_horas, na.rm = TRUE),
      num_beneficios_total = sum(num_beneficios, na.rm = TRUE),
      
      # Protección social
      prop_subsidiado = mean(bin_subsidiado, na.rm = TRUE),
      prop_sin_salud = mean(bin_sin_salud, na.rm = TRUE),
      bin_recibe_pension = max(bin_pension, na.rm = TRUE),
      bin_recibe_ayuda = max(bin_ayuda, na.rm = TRUE)
    ) |>
    mutate(
      # Ocupación (tasa de ocupación de la PET, como la publica el DANE)
      prop_ocupados_pet = replace_na(num_ocupados / na_if(num_pet, 0), 0),
      bin_sin_ocupados = as.integer(num_ocupados == 0),
      prop_inactividad = replace_na(num_inactivos / na_if(num_pet, 0), 0),
      prop_subsidiado = replace_na(prop_subsidiado, 0),
      prop_sin_salud = replace_na(prop_sin_salud, 0),
      
      # Capital humano
      prop_superior = replace_na(num_superior_pet / na_if(num_pet, 0), 0),
      num_educ_prom_pet = replace_na(num_educ_prom_pet, 0),
      num_edad_prom_pet = replace_na(num_edad_prom_pet, 0),
      
      # Dependencia: personas fuera de la PET por cada persona en la PET,
      # y personas del hogar por cada ocupado
      prop_dependencia = coalesce(num_no_pet / na_if(num_pet, 0), num_no_pet),
      tasa_dependencia_ocup = num_personas / pmax(num_ocupados, 1),
      
      # Género
      prop_mujeres_ocup = replace_na(num_mujeres_ocup / na_if(num_ocupados, 0), 0),
      
      # Calidad del empleo (proporciones sobre ocupados; 0 si no hay ocupados)
      prop_cotiza = replace_na(num_cotizan / na_if(num_ocupados, 0), 0),
      prop_asalariados = replace_na(num_asalariados / na_if(num_ocupados, 0), 0),
      prop_domestico = replace_na(num_domesticos / na_if(num_ocupados, 0), 0),
      prop_jornalero = replace_na(num_jornaleros / na_if(num_ocupados, 0), 0),
      prop_cuenta_propia = replace_na(num_cuenta_propia / na_if(num_ocupados, 0), 0),
      num_horas_prom = replace_na(num_horas_total / na_if(num_ocupados, 0), 0),
      num_beneficios_prom = replace_na(num_beneficios_total / na_if(num_ocupados, 0), 0)
    ) |>
    select(id, num_personas, num_pet, prop_ocupados_pet, bin_sin_ocupados,
           num_desocupados, prop_inactividad, cat_maxEduc, num_educ_max,
           num_educ_prom_pet, prop_superior, num_educ_jefe, num_minors,
           num_menores6, num_old, prop_dependencia, tasa_dependencia_ocup,
           num_edad_prom_pet, bin_head_male, prop_mujeres_ocup,
           prop_cotiza, prop_asalariados,
           prop_cuenta_propia, bin_hay_subempleo, num_horas_prom,
           num_beneficios_prom, prop_subsidiado, prop_sin_salud,
           bin_recibe_pension, bin_recibe_ayuda, prop_domestico, prop_jornalero) |>
    ungroup()
  
  return(db_result)
}


fun_preprocess_hogares <- function(.db) {
  #' Preprocesar variables de vivienda a nivel hogar
  #'
  #' @description \code{fun_preprocess_hogares()} convierte los NA estructurales
  #' de las variables de vivienda en indicadoras de tenencia e imputa con cero
  #' el monto cuando no aplica.
  #'
  #' @param .db \code{tibble}. Base de hogares (train o test) con las variables
  #' \code{P5100}, \code{P5130} y \code{P5140}.
  #'
  #' @returns \code{tibble}. Base con las indicadoras
  #' \code{bin_arriendo_imput} y \code{bin_paga_cuota}, y los montos sin NA.
  db_result <- .db |>
    mutate(bin_arriendo_imput = as.integer(!is.na(P5130)),
           bin_paga_cuota     = as.integer(!is.na(P5100)),
           # P5130 y P5140 son mutuamente excluyentes (cada hogar responde
           # solo una, según su tenencia), por lo que coalesce() nunca suma
           # dos valores reales a la vez.
           num_arriendo = coalesce(P5140, P5130, 0),
           num_log_arriendo = log1p(num_arriendo),
           P5100 = replace_na(P5100, 0)) |>
    rename(num_amortizacion = P5100) |>
    mutate(num_log_amortizacion = log1p(num_amortizacion)) |>
    select(-P5130, -P5140, -num_amortizacion, -num_arriendo)
  
  return(db_result)
}


fun_preprocess_privaciones <- function(.db_hogares, .db_personas_ok) {
  #' Construir proxies de privaciones del IPM como predictores
  #'
  #' @description \code{fun_preprocess_privaciones()} construye cinco
  #' indicadoras de privación inspiradas en la metodología del IPM del DANE
  #' (DANE, 2014), para usarlas como predictores adicionales de pobreza
  #' monetaria, no como un índice oficial. Cubre hacinamiento crítico, bajo
  #' logro educativo, rezago escolar, trabajo infantil y empleo informal.
  #' No se construyen los diez indicadores restantes del IPM oficial
  #' (material de pisos y paredes, acceso a agua y saneamiento,
  #' analfabetismo, inasistencia escolar, cuidado de primera infancia,
  #' desempleo de larga duración y barreras de acceso a salud) por no
  #' contar con esas variables en esta base.
  #'
  #' @param .db_hogares \code{tibble}. Base de hogares con \code{id},
  #' \code{Nper}, \code{P5010} y \code{Clase}.
  #' @param .db_personas_ok \code{tibble}. Base de personas preprocesada por
  #' \code{fun_preprocess_personas()}, con \code{id}, \code{num_age},
  #' \code{num_educ_anios}, \code{bin_occup} y \code{bin_cotiza}.
  #'
  #' @returns \code{tibble}. Una fila por hogar con las indicadoras
  #' \code{bin_hacinamiento}, \code{bin_bajo_logro_educ},
  #' \code{bin_rezago_escolar}, \code{bin_trabajo_infantil},
  #' \code{bin_empleo_informal_ipm}.
  
  # --- Hacinamiento (DANE, 2014): >= 3 personas por cuarto urbano,
  # > 3 rural.
  hacinamiento <- .db_hogares |>
    mutate(
      num_pers_cuarto = Nper / P5010,
      bin_hacinamiento = case_when(
        Clase == 1 & num_pers_cuarto >= 3 ~ 1,
        Clase == 2 & num_pers_cuarto > 3  ~ 1,
        TRUE ~ 0
      )
    ) |>
    select(id, bin_hacinamiento)
  
  # --- Bajo logro educativo (DANE, 2014): promedio de años de educación
  # de mayores de 15 años menor a 9. Hogares sin mayores de 15 se privan.
  logro_educ <- .db_personas_ok |>
    group_by(id) |>
    summarize(
      n_mayores15 = sum(num_age >= 15, na.rm = TRUE),
      prom_educ_mayores15 = mean(num_educ_anios[num_age >= 15], na.rm = TRUE)
    ) |>
    mutate(
      bin_bajo_logro_educ = case_when(
        n_mayores15 == 0 ~ 1,
        prom_educ_mayores15 < 9 ~ 1,
        TRUE ~ 0
      )
    ) |>
    select(id, bin_bajo_logro_educ)
  
  # --- Rezago escolar (DANE, 2014): niños de 7 a 17 años con años de
  # educación por debajo de la norma (edad - 6), aproximando la regla
  # exacta del DANE.
  rezago <- .db_personas_ok |>
    mutate(
      bin_en_rezago = as.integer(
        between(num_age, 7, 17) & num_educ_anios < (num_age - 6)
      )
    ) |>
    group_by(id) |>
    summarize(
      n_7_17 = sum(between(num_age, 7, 17), na.rm = TRUE),
      n_rezago = sum(bin_en_rezago, na.rm = TRUE)
    ) |>
    mutate(bin_rezago_escolar = as.integer(n_7_17 > 0 & n_rezago > 0)) |>
    select(id, bin_rezago_escolar)
  
  # --- Trabajo infantil (DANE, 2014): al menos un ocupado entre 12 y
  # 17 años.
  trabajo_infantil <- .db_personas_ok |>
    mutate(bin_nino_ocupado = as.integer(between(num_age, 12, 17) & bin_occup == 1)) |>
    group_by(id) |>
    summarize(bin_trabajo_infantil = max(bin_nino_ocupado, na.rm = TRUE)) |>
    select(id, bin_trabajo_infantil)
  
  # --- Unión y conteo de privaciones
  db_result <- hacinamiento |>
    left_join(logro_educ, by = "id") |>
    left_join(rezago, by = "id") |>
    left_join(trabajo_infantil, by = "id")
  
  return(db_result)
}


fun_privaciones_adicionales <- function(.db_personas_raw, .db_privaciones) {
  #' Agregar empleo informal y conteo de privaciones
  #'
  #' @description \code{fun_privaciones_adicionales()} construye la privacion
  #' de empleo informal del IPM (DANE, 2014), donde el hogar esta privado si
  #' al menos un ocupado no cotiza a pension. Los pensionados (P6920 = 3) no
  #' se cuentan como privados. Luego suma las cinco indicadoras en
  #' \code{num_privaciones}.
  #'
  #' @param .db_personas_raw \code{data.frame}. Base original de personas.
  #' @param .db_privaciones \code{tibble}. Salida de
  #' \code{fun_preprocess_privaciones()}.
  #'
  #' @returns \code{tibble}. Una fila por hogar con \code{id},
  #' \code{bin_empleo_informal_ipm} y \code{num_privaciones}.
  informal <- .db_personas_raw |>
    mutate(bin_ocup_informal = as.integer(Oc %in% 1 & !(P6920 %in% c(1, 3)))) |>
    group_by(id) |>
    summarize(bin_empleo_informal_ipm = max(bin_ocup_informal), .groups = "drop")
  
  .db_privaciones |>
    left_join(informal, by = "id") |>
    mutate(bin_empleo_informal_ipm = replace_na(bin_empleo_informal_ipm, 0L),
           num_privaciones = bin_hacinamiento + bin_bajo_logro_educ +
             bin_rezago_escolar + bin_trabajo_infantil +
             bin_empleo_informal_ipm) |>
    select(id, bin_empleo_informal_ipm, num_privaciones)
}


# --- Ecuacion de ingreso individual (etapa 1) ---

# Predictores: todas las variables de personas que existen en test, salvo
# identificadores, factores de expansion y Depto (redundante con Dominio).
vars_etapa1 <- setdiff(names(raw_test_personas),
                       c("id", "Orden", "Fex_c", "Fex_dpto", "Depto", "Pet"))

# Variables categoricas sin orden natural
vars_factor_etapa1 <- c("Dominio", "Oficio", "P6050", "P6100", "P6240",
                        "P6430", "P7050", "P7350")

formula_etapa1 <- as.formula(paste("y_log_ingtot ~",
                                   paste(vars_etapa1, collapse = " + ")))


fun_preprocess_etapa1 <- function(.db_train, .db_test) {
  #' Preparar las bases de personas para la ecuacion de ingreso
  #'
  #' @description \code{fun_preprocess_etapa1()} conserva solo la poblacion
  #' en edad de trabajar (Ingtot es NA exactamente para quienes no estan en
  #' la PET, y su ingreso se fija en 0 al agregar). Los NA estructurales
  #' (no aplica) se codifican como -1 en las numericas y como "NA" en las
  #' categoricas, para que el arbol los trate como una categoria propia.
  #' Los niveles de los factores se fijan con la union de train y test.
  #'
  #' @param .db_train,.db_test \code{data.frame}. Bases originales de
  #' personas.
  #'
  #' @returns \code{list} con \code{train} y \code{test}. Cada una incluye
  #' \code{id}, \code{cod_parentesco} (P6050 original, para definir la
  #' unidad de gasto) y los predictores. \code{train} incluye ademas
  #' \code{y_log_ingtot = log1p(Ingtot)}.
  prep <- function(.db) {
    .db |>
      filter(!is.na(Pet)) |>
      mutate(cod_parentesco = P6050) |>
      select(id, cod_parentesco, any_of("Ingtot"), all_of(vars_etapa1)) |>
      mutate(across(all_of(vars_factor_etapa1),
                    ~ replace_na(as.character(.x), "NA")),
             across(setdiff(vars_etapa1, vars_factor_etapa1),
                    ~ replace_na(as.numeric(.x), -1)))
  }
  
  db_tr <- prep(.db_train) |> mutate(y_log_ingtot = log1p(Ingtot)) |> select(-Ingtot)
  db_te <- prep(.db_test)
  
  for (v in vars_factor_etapa1) {
    niveles <- sort(union(unique(db_tr[[v]]), unique(db_te[[v]])))
    db_tr[[v]] <- factor(db_tr[[v]], levels = niveles)
    db_te[[v]] <- factor(db_te[[v]], levels = niveles)
  }
  
  list(train = db_tr, test = db_te)
}


fun_ranger_etapa1 <- function(.data) {
  #' Ajustar la ecuacion de ingreso individual con ranger
  #'
  #' @description Hiperparametros fijos (sin tuning).
  #' \code{respect.unordered.factors = "order"} ordena las categorias por su
  #' ingreso medio, lo que permite usar Oficio (82 categorias) y Dominio sin
  #' crear dummies.
  #'
  #' @param .data \code{tibble}. Personas con \code{y_log_ingtot} y los
  #' predictores de \code{vars_etapa1}.
  #'
  #' @returns Objeto \code{ranger}.
  ranger(
    formula = formula_etapa1,
    data = .data,
    num.trees = 200,
    mtry = 15,
    min.node.size = 50,
    respect.unordered.factors = "order",
    seed = 2026
  )
}


fun_agregar_ingreso <- function(.db_personas_pred, .db_hogares_raw) {
  #' Construir el ingreso per capita predicho de la unidad de gasto (etapa 2)
  #'
  #' @description \code{fun_agregar_ingreso()} replica la construccion de
  #' Ingpcug. Suma el ingreso predicho de las personas de la unidad de gasto
  #' (excluye P6050 = 6, 7 y 8: servicio domestico, pensionistas y
  #' trabajadores), suma el arriendo imputado solo cuando P5090 es 1 o 4 y
  #' P5130 no es 98 ni 99, y divide por Npersug. Con el ingreso observado,
  #' esta regla reproduce Pobre en el 99,9% de los hogares de train.
  #'
  #' @param .db_personas_pred \code{tibble}. Personas de la etapa 1 con
  #' \code{id}, \code{cod_parentesco} y \code{y_log_hat}.
  #' @param .db_hogares_raw \code{data.frame}. Base original de hogares.
  #'
  #' @returns \code{tibble}. Una fila por hogar con \code{id},
  #' \code{num_ing_ug_hat}, \code{num_ing_pc_hat} y \code{num_ratio_lp_hat}.
  ing_ug <- .db_personas_pred |>
    filter(!cod_parentesco %in% c(6, 7, 8)) |>
    group_by(id) |>
    summarize(num_ing_ug_hat = sum(pmax(expm1(y_log_hat), 0)), .groups = "drop")
  
  .db_hogares_raw |>
    select(id, Lp, Npersug, P5090, P5130) |>
    left_join(ing_ug, by = "id") |>
    mutate(
      num_ing_ug_hat = replace_na(num_ing_ug_hat, 0),
      num_arriendo_imp_dane = ifelse(P5090 %in% c(1, 4) & !(P5130 %in% c(98, 99)),
                                     replace_na(P5130, 0), 0),
      num_ing_pc_hat = (num_ing_ug_hat + num_arriendo_imp_dane) / pmax(Npersug, 1),
      num_ratio_lp_hat = num_ing_pc_hat / Lp
    ) |>
    select(id, num_ing_ug_hat, num_ing_pc_hat, num_ratio_lp_hat)
}


# 2. CONSTRUCCION DE LAS BASES FINALES ----------------------------------------

db_train_personas <- fun_preprocess_personas(db_train_personas)
db_test_personas  <- fun_preprocess_personas(db_test_personas)

db_train_personas_nivel_hogar <- fun_colapse_personas(db_train_personas)
db_test_personas_nivel_hogar  <- fun_colapse_personas(db_test_personas)

db_train_privaciones <- fun_preprocess_privaciones(db_train_hogares, db_train_personas)
db_test_privaciones  <- fun_preprocess_privaciones(db_test_hogares, db_test_personas)

db_train_ipm_extra <- fun_privaciones_adicionales(raw_train_personas, db_train_privaciones)
db_test_ipm_extra  <- fun_privaciones_adicionales(raw_test_personas,  db_test_privaciones)

db_train <- db_train_hogares |>
  left_join(db_train_personas_nivel_hogar, by = 'id') |>
  left_join(db_train_privaciones, by = 'id') |>
  left_join(db_train_ipm_extra, by = 'id') |>
  fun_preprocess_hogares() |>
  select(-id)

db_test <- db_test_hogares |>
  left_join(db_test_personas_nivel_hogar, by = 'id') |>
  left_join(db_test_privaciones, by = 'id') |>
  left_join(db_test_ipm_extra, by = 'id') |>
  fun_preprocess_hogares()

# 1. Preparar la base: caret exige que los niveles de la respuesta sean
#    nombres válidos de R (no 0 y 1)
db_train <- db_train |>
  mutate(Pobre = factor(Pobre, levels = c(0, 1), labels = c("No", "Si")))

# Dominio como factor, con los mismos niveles en train y test
niveles_dominio <- sort(unique(raw_train_hogares$Dominio))
db_train <- db_train |> mutate(Dominio = factor(Dominio, levels = niveles_dominio))
db_test  <- db_test  |> mutate(Dominio = factor(Dominio, levels = niveles_dominio))

# Eliminamos variables repetidas. Npersug se conserva porque es el
# denominador del ingreso per capita de la unidad de gasto (DANE).
db_train <- db_train |>
  select(-Nper, -P5000, -P5010, -P5090)

# Verificación: no deben quedar NA
sum(is.na(db_train))
sum(is.na(db_test))


# 3. FUNCION AUXILIAR DE METRICAS (usada por todos los modelos) --------------

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


# 4. FOLDS Y CONTROLES COMUNES -------------------------------------------------

set.seed(2026)
folds <- createFolds(db_train$Pobre, k = 5, returnTrain = TRUE)

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


# 5. VARIABLES DE INGRESO PREDICHO (DOS ETAPAS) --------------------------------

# Esta seccion va despues de los folds porque la etapa 1 se estima fuera de
# muestra con los mismos folds de los modelos, para evitar fuga de la
# respuesta hacia los predictores.

id_train <- db_train_hogares$id
stopifnot(length(id_train) == nrow(db_train),
          identical(id_train, raw_train_hogares$id))

# --- Etapa 1: ecuacion de ingreso individual (ranger, fuera de muestra) ---

etapa1 <- fun_preprocess_etapa1(raw_train_personas, raw_test_personas)
db_train_etapa1 <- etapa1$train
db_test_etapa1  <- etapa1$test
rm(etapa1)

# Fold de cada hogar: el hogar pertenece al fold k cuando queda fuera de
# folds[[k]] (returnTrain = TRUE). Cada persona hereda el fold de su hogar.
fold_hogar <- integer(length(id_train))
for (k in seq_along(folds)) fold_hogar[-folds[[k]]] <- k

db_train_etapa1 <- db_train_etapa1 |>
  left_join(tibble(id = id_train, fold = fold_hogar), by = "id")
stopifnot(!anyNA(db_train_etapa1$fold))

# Predicciones fuera de muestra para train: el modelo del fold k se entrena
# con las personas de los otros cuatro folds y predice las del fold k.
db_train_etapa1$y_log_hat <- NA_real_
for (k in seq_along(folds)) {
  en_k <- db_train_etapa1$fold == k
  modelo_k <- fun_ranger_etapa1(db_train_etapa1[!en_k, ])
  db_train_etapa1$y_log_hat[en_k] <-
    predict(modelo_k, data = db_train_etapa1[en_k, ])$predictions
  rm(modelo_k); gc()
  cat("Etapa 1, fold", k, "listo\n")
}

# Calidad de la etapa 1. El R2 total esta inflado por los ingresos en cero
# (inactivos, faciles de predecir). El relevante es el de ingreso > 0.
r2 <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
con_ingreso <- db_train_etapa1$y_log_ingtot > 0
r2_etapa1_total     <- r2(db_train_etapa1$y_log_ingtot, db_train_etapa1$y_log_hat)
r2_etapa1_positivos <- r2(db_train_etapa1$y_log_ingtot[con_ingreso],
                          db_train_etapa1$y_log_hat[con_ingreso])
r2_etapa1_total
r2_etapa1_positivos

# Test: modelo entrenado con todas las personas de train
ranger_etapa1_full <- fun_ranger_etapa1(db_train_etapa1)
db_test_etapa1$y_log_hat <- predict(ranger_etapa1_full, data = db_test_etapa1)$predictions
rm(ranger_etapa1_full); gc()

# --- Etapa 2: ingreso del hogar con la formula del DANE ---

db_train_ingreso <- fun_agregar_ingreso(db_train_etapa1, raw_train_hogares)
db_test_ingreso  <- fun_agregar_ingreso(db_test_etapa1,  raw_test_hogares)

# --- Bases finales: todos los modelos se estiman sobre estas bases ---

db_train <- db_train |>
  mutate(id = id_train) |>
  left_join(db_train_ingreso, by = "id") |>
  select(-id)

db_test <- db_test |>
  left_join(db_test_ingreso, by = "id")

# Verificación: no deben quedar NA en los predictores
sum(is.na(db_train))
sum(is.na(db_test[, setdiff(names(db_train), "Pobre")]))


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

best_cutoff_logitf1 <- pr_cutoffs_logitf1 |> slice_max(F1, n = 1)
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

best_cutoff_ldaf1 <- pr_cutoffs_ldaf1 |> slice_max(F1, n = 1)
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


qda_roc <- train(
  Pobre ~ .,
  data = db_train |> select(-num_privaciones, -Clase),
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
  data = db_train |> select(-num_privaciones, -Clase),
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

best_cutoff_qdaf1 <- pr_cutoffs_qdaf1 |> slice_max(F1, n = 1)
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

best_cutoff_logitupf1 <- pr_cutoffs_logitupf1 |> slice_max(F1, n = 1)
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

best_cutoff_bagf1 <- pr_cutoffs_bagf1 |> slice_max(F1, n = 1)
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

best_cutoff_bagupf1 <- pr_cutoffs_bagupf1 |> slice_max(F1, n = 1)
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

best_cutoff_rff1 <- pr_cutoffs_rff1 |> slice_max(F1, n = 1)
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

best_cutoff_rfupf1 <- pr_cutoffs_rfupf1 |> slice_max(F1, n = 1)
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

best_cutoff_adaf1 <- pr_cutoffs_adaf1 |> slice_max(F1, n = 1)
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

best_cutoff_adaupf1 <- pr_cutoffs_adaupf1 |> slice_max(F1, n = 1)
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

best_cutoff_gbmf1 <- pr_cutoffs_gbmf1 |> slice_max(F1, n = 1)
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

best_cutoff_gbmupF <- pr_cutoffs_gbmupF |> slice_max(F1, n = 1)
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