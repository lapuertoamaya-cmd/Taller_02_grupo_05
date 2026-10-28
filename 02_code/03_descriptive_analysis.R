# ------------------------------------------------------------------------------
# Archivo: 03_descriptive_analysis.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Explorar las dimensiones, el desbalance y las distribuciones de los datos.
# Input:
#   Bases originales, features de 02_creation_income_var.R y diccionario
#   04_documentation/diccionario_variables.xlsx (hoja Variables hogares).
#   Paquetes del setup y rutas definidas en main.R.
# Output:
#   Cinco CSV en 03_outputs/descriptivas/tables/ y cuatro PDF vectoriales
#   en 03_outputs/descriptivas/figures/, numerados segun el orden del analisis.
# ------------------------------------------------------------------------------

# Reutilizar las rutas definidas en main.R.
dir_raw_completo  <- dir_datos_raw
dir_repo_features <- dir_features
dir_salida        <- file.path(dir_resultados, "descriptivas")

for (carpeta in c("tables", "figures")) {
  dir.create(file.path(dir_salida, carpeta),
             recursive = TRUE, showWarnings = FALSE)
}

# ------------------------------------------------------------------------------
# (1) Cantidad de variables y filas de cada base cruda
# ------------------------------------------------------------------------------

contar_dimensiones <- function(nombre, ruta) {
  # Solo la primera fila de cada base: mas rapido que leerla completa.
  primera_fila <- read_csv(ruta, n_max = 1, show_col_types = FALSE)
  n_filas <- length(count.fields(ruta, sep = ",")) - 1  # -1 por el encabezado
  tibble(base = nombre,
         filas = n_filas,
         columnas = ncol(primera_fila))
}

dimensiones_bases <- bind_rows(
  contar_dimensiones(
    "Hogares de entrenamiento",
    file.path(dir_raw_completo, "train_hogares.csv")
  ),
  contar_dimensiones(
    "Hogares de prueba",
    file.path(dir_raw_completo, "test_hogares.csv")
  ),
  contar_dimensiones(
    "Personas de entrenamiento",
    file.path(dir_raw_completo, "train_personas.csv")
  ),
  contar_dimensiones(
    "Personas de prueba",
    file.path(dir_raw_completo, "test_personas.csv")
  )
)

cat("Dimensiones de las cuatro bases:\n")
print(dimensiones_bases)
write_csv(dimensiones_bases,
          file.path(dir_salida, "tables", "01_dimensiones_bases.csv"))


# ------------------------------------------------------------------------------
# (2) Desbalance de clases (Pobre), sin ponderar y ponderado
# ------------------------------------------------------------------------------

train_hogares <- read_csv(file.path(dir_raw_completo, "train_hogares.csv"),
                          show_col_types = FALSE)

desbalance <- train_hogares |>
  summarise(
    n_hogares = n(),
    n_pobres = sum(Pobre),
    tasa_pobreza_sin_ponderar = mean(Pobre),
    tasa_pobreza_ponderada = weighted.mean(Pobre, Fex_c),
    ratio_no_pobre_pobre = sum(Pobre == 0) / sum(Pobre == 1)
  )

cat("\nDesbalance de clases:\n")
print(desbalance)
write_csv(desbalance,
          file.path(dir_salida, "tables", "02_desbalance_clases.csv"))


# ------------------------------------------------------------------------------
# (3) Distribucion del ingreso per capita: observado vs. predicho
# ------------------------------------------------------------------------------

archivo_db_train <- file.path(dir_repo_features, "db_train.rds")
archivo_db_test  <- file.path(dir_repo_features, "db_test.rds")
if (!file.exists(archivo_db_train) ||
    !file.exists(archivo_db_test)) {
  stop(
    "Faltan ",
    dir_repo_features,
    "/db_train.rds y db_test.rds. ",
    "Se generan corriendo 02_code/01_cleaning.R y 02_code/02_creation_income_var.R ",
    "en el orden definido por main.R."
  )
}
db_train <- readRDS(archivo_db_train)
db_test  <- readRDS(archivo_db_test)

stopifnot("num_ing_pc_hat" %in% names(db_train),
          "num_ing_pc_hat" %in% names(db_test))

cat("\nNA en num_ing_pc_hat -> train:",
    sum(is.na(db_train$num_ing_pc_hat)),
    "| test:",
    sum(is.na(db_test$num_ing_pc_hat)),
    "\n")

ingreso_pc <- bind_rows(
  tibble(
    muestra = "Entrenamiento: observado",
    log_ingreso_pc = log1p(train_hogares$Ingpcug)
  ),
  tibble(muestra = "Entrenamiento: predicho", log_ingreso_pc = log1p(pmax(
    db_train$num_ing_pc_hat, 0
  ))),
  tibble(muestra = "Prueba: predicho", log_ingreso_pc = log1p(pmax(
    db_test$num_ing_pc_hat, 0
  )))
) |>
  mutate(muestra = factor(
    muestra,
    levels = c(
      "Entrenamiento: observado",
      "Entrenamiento: predicho",
      "Prueba: predicho"
    )
  ))

# Linea de pobreza mediana, como referencia (Lp varia por dominio y mes).
lp_mediana <- median(train_hogares$Lp, na.rm = TRUE)

fig_ingreso <- ggplot(ingreso_pc,
                      aes(x = log_ingreso_pc, fill = muestra, color = muestra)) +
  geom_density(alpha = 0.3, linewidth = 0.8) +
  geom_vline(
    xintercept = log1p(lp_mediana),
    linetype = "dashed",
    color = "black"
  ) +
  annotate(
    "text",
    x = log1p(lp_mediana),
    y = Inf,
    label = "Línea de pobreza mediana",
    vjust = 1.5,
    hjust = -0.1,
    size = 3
  ) +
  labs(
    title = "Ingreso por persona: observado y predicho",
    subtitle = "Distribuciones sin ponderar; ingreso mensual por persona en pesos.",
    x = "Logaritmo de uno más el ingreso mensual por persona",
    y = "Densidad",
    fill = NULL,
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", legend.justification = "center",
        plot.title = element_text(face = "bold")) +
  guides(fill = guide_legend(nrow = 1, ncol = 3, byrow = TRUE),
         color = guide_legend(nrow = 1, ncol = 3, byrow = TRUE))

ggsave(
  file.path(dir_salida, "figures", "01_distribucion_ingreso_por_persona.pdf"),
  fig_ingreso,
  device = cairo_pdf,
  width = 10,
  height = 6.5
)


# ------------------------------------------------------------------------------
# (4) Descriptivas y correlacion simple de las variables con Pobre
# ------------------------------------------------------------------------------
# Para cada variable numerica (incluye las binarias 0/1, que ya vienen asi en
# la base limpia): media, desviacion estandar y correlacion simple con Pobre.
# Es una exploracion univariada -- da una primera idea de que variables estan
# mas asociadas con la pobreza, pero no reemplaza la importancia de variables
# que calculan los modelos.

pobre_num <- as.numeric(db_train$Pobre == "Si")

vars_numericas <- db_train |>
  select(where(is.numeric)) |>
  select(-any_of(c("id", "Pobre"))) |>
  names()

# Nombres para lectura; conservar el codigo original en una columna separada.
etiquetas <- read_excel(file.path(dir_documentacion, "diccionario_variables.xlsx"),
                        sheet = "Variables hogares") |>
  transmute(variable = Variable, nombre = Nombre) |>
  distinct(variable, .keep_all = TRUE)
etiquetas_extra <- tribble(
  ~variable, ~nombre,
  "lp", "Línea de pobreza monetaria por persona",
  "bin_arriendo_nsnr", "Arriendo: monto no informado",
  "bin_cuota_nsnr", "Amortización: monto no informado",
  "bin_hay_rentas_pensiones", "Hogar con rentas o pensiones",
  "bin_empleo_informal_ipm", "Empleo informal: aproximación de privación",
  "personas_por_dormitorio", "Personas por dormitorio",
  "personas_por_cuarto", "Personas por cuarto de la vivienda",
  "pct_personas_ingreso", "Ocupados y pensionados (% de la unidad de gasto)",
  "pct_ocupados", "Personas ocupadas (% de la unidad de gasto)",
  "pct_cotizantes", "Cotizantes a pensión (% de ocupados)",
  "num_privaciones", "Número de privaciones aproximadas",
  "arriendo_pc_lp", "Arriendo por persona respecto de la línea de pobreza",
  "log_arriendo_cuarto", "Logaritmo del arriendo por cuarto",
  "num_ing_ug_hat", "Ingreso predicho de la unidad de gasto",
  "num_ing_pc_hat", "Ingreso predicho por persona",
  "num_ratio_lp_hat", "Ingreso predicho por persona respecto de la línea de pobreza",
  "num_ratio_lp_sin_arr_hat", "Ingreso predicho respecto de la línea de pobreza, sin arriendo imputado",
  "num_ing_max_hat", "Mayor ingreso individual predicho del hogar",
  "num_perceptores_hat", "Número de perceptores de ingreso predicho",
  "prop_ing_jefe_hat", "Participación del jefe en el ingreso predicho del hogar"
)
etiquetas <- bind_rows(etiquetas_extra, etiquetas) |>
  distinct(variable, .keep_all = TRUE)
# Las indicadoras na_* describen desconocimiento, no ausencia del beneficio.
etiquetas_faltantes <- tibble(variable = vars_numericas[str_starts(vars_numericas, "na_")]) |>
  mutate(base = str_remove(variable, "^na_")) |>
  left_join(etiquetas, by = c("base" = "variable")) |>
  transmute(variable, nombre = paste("Información faltante:", nombre))
etiquetas <- bind_rows(etiquetas_faltantes, etiquetas)

descriptivas_vars <- map_dfr(vars_numericas, function(v) {
  x <- db_train[[v]]
  x[!is.finite(x)] <- NA_real_
  completos <- complete.cases(x, pobre_num)
  correlacion <- if (sum(completos) > 1 && sd(x[completos]) > 0 &&
                    sd(pobre_num[completos]) > 0) cor(x[completos], pobre_num[completos]) else NA_real_
  tibble(
    variable = v,
    media = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    correlacion_pobre = correlacion
  )
}) |>
  left_join(etiquetas, by = "variable") |>
  mutate(nombre = coalesce(nombre, str_to_sentence(str_replace_all(variable, "_", " ")))) |>
  select(variable, nombre, everything()) |>
  arrange(desc(abs(correlacion_pobre)))

write_csv(descriptivas_vars,
          file.path(dir_salida, "tables", "03_descriptivas_variables.csv"))

cat("\nTop 10 variables mas correlacionadas con Pobre:\n")
print(head(descriptivas_vars, 10))

top_vars <- descriptivas_vars |>
  filter(is.finite(correlacion_pobre)) |>
  slice_head(n = 15) |>
  mutate(etiqueta = str_wrap(nombre, width = 48))

fig_top_vars <- ggplot(top_vars, aes(x = reorder(etiqueta, abs(correlacion_pobre)), y = correlacion_pobre)) +
  geom_col(fill = "#2c3e50") +
  coord_flip() +
  labs(
    title = "Variables más asociadas con la pobreza",
    subtitle = "Quince mayores correlaciones absolutas en entrenamiento; asociación sin controles.",
    x = NULL,
    y = "Correlación con la condición de pobreza"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_salida, "figures", "02_variables_asociadas_pobreza.pdf"),
  fig_top_vars,
  device = cairo_pdf,
  width = 12,
  height = 9
)

# ------------------------------------------------------------------------------
# (5) Magnitud absoluta y signo de la asociacion con pobreza
# ------------------------------------------------------------------------------

top_vars_absolutas <- top_vars |>
  mutate(magnitud = abs(correlacion_pobre),
         signo = factor(if_else(correlacion_pobre >= 0, "Positiva", "Negativa"),
                        levels = c("Positiva", "Negativa")))

fig_top_absolutas <- ggplot(top_vars_absolutas,
    aes(x = magnitud, y = reorder(etiqueta, magnitud), fill = signo)) +
  geom_col(width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", magnitud)), hjust = -0.15, size = 3.5) +
  scale_fill_manual(values = c("Positiva" = "#2563EB", "Negativa" = "#DC2626"),
                    drop = FALSE) +
  # Todas las barras parten de cero; el color conserva el signo original.
  scale_x_continuous(limits = c(0, NA), breaks = seq(0, 1, by = 0.05),
                     expand = expansion(mult = c(0, 0.14))) +
  labs(title = "Fuerza de asociación de las variables con la pobreza",
       subtitle = "Quince mayores correlaciones absolutas; el color indica el signo original.",
       x = "Valor absoluto de la correlación con la condición de pobreza",
       y = NULL, fill = NULL) +
  guides(fill = guide_legend(nrow = 1, ncol = 2, byrow = TRUE)) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", legend.justification = "center",
        panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(dir_salida, "figures", "03_asociacion_absoluta_pobreza.pdf"),
       fig_top_absolutas, device = cairo_pdf, width = 12, height = 9)

# ------------------------------------------------------------------------------
# (6) Comparar medias de entrenamiento y prueba: t de Welch, bilateral
# ------------------------------------------------------------------------------

# Mismas variables numericas, incluidas binarias (su media es una proporcion).
# No ponderar; excluir identificadores y la respuesta, que no existe en test.
variables_comunes <- intersect(vars_numericas,
  names(db_test)[vapply(db_test, is.numeric, logical(1))])
variables_comunes <- setdiff(variables_comunes, c("id", "Pobre", "pobre"))

comparacion_train_test <- map_dfr(variables_comunes, function(v) {
  x <- db_train[[v]]; x <- x[is.finite(x)]
  y <- db_test[[v]]; y <- y[is.finite(y)]
  # Welch no supone varianzas iguales. Sin variacion o sin observaciones
  # suficientes, el contraste no es estimable: conservar NA, no imputar p=1.
  prueba <- if (length(x) >= 2 && length(y) >= 2 && (var(x) > 0 || var(y) > 0))
    tryCatch(t.test(x, y, var.equal = FALSE, alternative = "two.sided"),
             error = function(e) NULL) else NULL
  tibble(variable = v, n_train = length(x), n_test = length(y),
    media_train = if (length(x)) mean(x) else NA_real_,
    media_test = if (length(y)) mean(y) else NA_real_,
    diferencia_train_test = media_train - media_test,
    p_value = if (is.null(prueba)) NA_real_ else prueba$p.value,
    nota = if (is.null(prueba)) "Contraste no estimable: datos insuficientes o sin variacion" else "")
}) |>
  left_join(etiquetas, by = "variable") |>
  mutate(nombre = coalesce(nombre, str_to_sentence(str_replace_all(variable, "_", " ")))) |>
  select(variable, nombre, n_train, n_test, media_train, media_test,
         diferencia_train_test, p_value, nota)

write_csv(comparacion_train_test,
          file.path(dir_salida, "tables", "04_comparacion_train_test.csv"))

# ------------------------------------------------------------------------------
# (7) Distribucion de la linea de pobreza: entrenamiento frente a prueba
# ------------------------------------------------------------------------------

test_hogares_lp <- read_csv(file.path(dir_raw_completo, "test_hogares.csv"),
                            col_select = Lp, show_col_types = FALSE)
lineas_pobreza <- bind_rows(
  tibble(muestra = "Entrenamiento", linea_pobreza = train_hogares$Lp),
  tibble(muestra = "Prueba", linea_pobreza = test_hogares_lp$Lp)
) |>
  filter(is.finite(linea_pobreza), linea_pobreza > 0) |>
  mutate(muestra = factor(muestra, levels = c("Entrenamiento", "Prueba")))

# Densidades sin ponderar; mismo suavizado para comparar ambas muestras.
# La linea es mensual por persona, no ingreso observado ni predicho.
ancho_lp <- bw.nrd0(lineas_pobreza$linea_pobreza)
fig_lineas_pobreza <- ggplot(lineas_pobreza,
    aes(x = linea_pobreza, color = muestra, fill = muestra)) +
  geom_density(alpha = 0.2, linewidth = 0.9, bw = ancho_lp) +
  scale_color_manual(values = c("Entrenamiento" = "#2563EB", "Prueba" = "#DC2626")) +
  scale_fill_manual(values = c("Entrenamiento" = "#2563EB", "Prueba" = "#DC2626")) +
  scale_x_continuous(labels = function(x)
    format(x, big.mark = ".", decimal.mark = ",", scientific = FALSE, trim = TRUE)) +
  labs(title = "Distribución de la línea de pobreza",
       subtitle = "Comparación sin ponderar entre hogares de entrenamiento y prueba.",
       x = "Línea de pobreza mensual por persona (pesos)", y = "Densidad",
       color = NULL, fill = NULL) +
  guides(color = guide_legend(nrow = 1, ncol = 2),
         fill = guide_legend(nrow = 1, ncol = 2)) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", legend.justification = "center",
        plot.title = element_text(face = "bold"))

ggsave(file.path(dir_salida, "figures", "04_distribucion_linea_pobreza.pdf"),
       fig_lineas_pobreza, device = cairo_pdf, width = 10, height = 6.5)

# ------------------------------------------------------------------------------
# (8) Comparar pobres y no pobres dentro de entrenamiento
# ------------------------------------------------------------------------------

# Pobreza observada, no predicha. Medias sin ponderar; binarias como proporciones.
# La diferencia se expresa como pobres menos no pobres, en la unidad original.
comparacion_pobreza <- map_dfr(setdiff(vars_numericas, c("id", "Pobre", "pobre")), function(v) {
  x <- db_train[[v]][db_train$Pobre == "Si"]
  y <- db_train[[v]][db_train$Pobre == "No"]
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  prueba <- if (length(x) >= 2 && length(y) >= 2 && (var(x) > 0 || var(y) > 0))
    tryCatch(t.test(x, y, var.equal = FALSE, alternative = "two.sided"),
             error = function(e) NULL) else NULL
  tibble(variable = v, n_pobres = length(x), n_no_pobres = length(y),
    media_pobres = if (length(x)) mean(x) else NA_real_,
    media_no_pobres = if (length(y)) mean(y) else NA_real_,
    diferencia_pobres_no_pobres = media_pobres - media_no_pobres,
    p_value = if (is.null(prueba)) NA_real_ else prueba$p.value,
    nota = if (is.null(prueba)) "Contraste no estimable: datos insuficientes o sin variacion" else "")
}) |>
  left_join(etiquetas, by = "variable") |>
  mutate(nombre = coalesce(nombre, str_to_sentence(str_replace_all(variable, "_", " ")))) |>
  select(variable, nombre, n_pobres, n_no_pobres, media_pobres, media_no_pobres,
         diferencia_pobres_no_pobres, p_value, nota)

write_csv(comparacion_pobreza,
          file.path(dir_salida, "tables", "05_comparacion_pobres_no_pobres_train.csv"))

cat("\nListo. Tablas y figura guardadas en",
    normalizePath(dir_salida),
    "\n")