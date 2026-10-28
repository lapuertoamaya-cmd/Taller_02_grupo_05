# ------------------------------------------------------------------------------
# Archivo: 03_descriptive_analysis.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Explorar las dimensiones, el desbalance y las distribuciones de los datos.
# Input:
#   Bases originales y features de 02_creation_income_var.R.
#   Paquetes del setup y rutas definidas en main.R.
# Output:
#   Tablas y figuras descriptivas.
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
    "train_hogares",
    file.path(dir_raw_completo, "train_hogares.csv")
  ),
  contar_dimensiones(
    "test_hogares",
    file.path(dir_raw_completo, "test_hogares.csv")
  ),
  contar_dimensiones(
    "train_personas",
    file.path(dir_raw_completo, "train_personas.csv")
  ),
  contar_dimensiones(
    "test_personas",
    file.path(dir_raw_completo, "test_personas.csv")
  )
)

cat("Dimensiones de las cuatro bases:\n")
print(dimensiones_bases)
write_csv(dimensiones_bases,
          file.path(dir_salida, "tables", "dimensiones_bases.csv"))


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
          file.path(dir_salida, "tables", "desbalance_clases.csv"))


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
    "Se generan corriendo 02_code/01_cleaning.R y 02_code/02_ingreso.R ",
    "(o main.R completo) una vez en el repo clonado de GitHub."
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
    muestra = "Train - observado",
    log_ingreso_pc = log1p(train_hogares$Ingpcug)
  ),
  tibble(muestra = "Train - predicho (XGBoost)", log_ingreso_pc = log1p(pmax(
    db_train$num_ing_pc_hat, 0
  ))),
  tibble(muestra = "Test - predicho (XGBoost)", log_ingreso_pc = log1p(pmax(
    db_test$num_ing_pc_hat, 0
  )))
) |>
  mutate(muestra = factor(
    muestra,
    levels = c(
      "Train - observado",
      "Train - predicho (XGBoost)",
      "Test - predicho (XGBoost)"
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
    label = "Lp mediana",
    vjust = 1.5,
    hjust = -0.1,
    size = 3
  ) +
  labs(
    title = "Distribución del ingreso per cápita del hogar: observado vs. predicho",
    subtitle = "Escala log(1 + ingreso). El ingreso predicho sale del modelo de ingreso potencial (XGBoost, 02_ingreso.R)",
    x = "log(1 + ingreso per cápita)",
    y = "Densidad",
    fill = NULL,
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  file.path(dir_salida, "figures", "distribucion_ingreso_pc.pdf"),
  fig_ingreso,
  width = 8,
  height = 5.5
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
  select(-any_of("Pobre")) |>
  names()

descriptivas_vars <- map_dfr(vars_numericas, function(v) {
  x <- db_train[[v]]
  tibble(
    variable = v,
    media = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    correlacion_pobre = suppressWarnings(cor(x, pobre_num, use = "complete.obs"))
  )
}) |>
  arrange(desc(abs(correlacion_pobre)))

write_csv(descriptivas_vars,
          file.path(dir_salida, "tables", "descriptivas_variables.csv"))

cat("\nTop 10 variables mas correlacionadas con Pobre:\n")
print(head(descriptivas_vars, 10))

top_vars <- descriptivas_vars |> slice_head(n = 15)

fig_top_vars <- ggplot(top_vars, aes(x = reorder(variable, abs(correlacion_pobre)), y = correlacion_pobre)) +
  geom_col(fill = "#2c3e50") +
  coord_flip() +
  labs(
    title = "Variables mas asociadas con Pobre (correlación simple)",
    subtitle = "Top 15 por valor absoluto de correlación, base de entrenamiento",
    x = NULL,
    y = "Correlación con Pobre"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(dir_salida, "figures", "top_variables_pobre.pdf"),
  fig_top_vars,
  width = 7,
  height = 6
)

cat("\nListo. Tablas y figura guardadas en",
    normalizePath(dir_salida),
    "\n")
