# ------------------------------------------------------------------------------
# Archivo: 01_cleaning.R
# Proyecto: Taller 2 - Big Data y Machine Learning
#
# Descripcion:
#   Importar las cuatro bases y procesar hogares y personas.
#   Construir variables individuales, agregar personas, unir y procesar hogares.
#   Version unificada: integra la construccion de 01_cleaning.R con las
#   variables de vivienda y privaciones del script de modelos, y corrige los
#   codigos 98/99 (no sabe / no responde) en los montos de la vivienda.
# Input:
#   train_hogares.csv, train_personas.csv, test_hogares.csv y test_personas.csv
#   en 01_data/01_raw/. Paquetes y dir_datos_raw preparados por main.R.
# Output:
#   Cuatro bases originales y sus versiones procesadas con sufijo _clean.
#   Hogares: tamanio, vivienda, arriendo, amortizacion y pobre (solo en train).
#   Personas: identificadores, educacion, actividad, condiciones laborales,
#   proteccion social y recepcion de pagos.
#   train_clean y test_clean listas para modelar (tipos definidos, sin NA y
#   Pobre como factor No/Si en train); archivos CSV en 01_data/02_clean/.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Importar las bases originales
# ------------------------------------------------------------------------------

# read_csv adivina el tipo de cada columna con las primeras filas. Columnas
# casi vacias al inicio (Imdies, Iof1es) se leian como logicas y sus montos
# quedaban en NA; guess_max = Inf obliga a revisar todas las filas.

# (1.1) Entrenamiento
db_train_hogares <- read_csv(
  file.path(dir_datos_raw, "train_hogares.csv"), show_col_types = FALSE,
  guess_max = Inf
)
db_train_personas <- read_csv(
  file.path(dir_datos_raw, "train_personas.csv"), show_col_types = FALSE,
  guess_max = Inf
)

# (1.2) Test
db_test_hogares <- read_csv(
  file.path(dir_datos_raw, "test_hogares.csv"), show_col_types = FALSE,
  guess_max = Inf
)
db_test_personas <- read_csv(
  file.path(dir_datos_raw, "test_personas.csv"), show_col_types = FALSE,
  guess_max = Inf
)


# ------------------------------------------------------------------------------
# (2) Definir las funciones de procesamiento
# ------------------------------------------------------------------------------

# (2.1) Procesar hogares
fun_preprocess_hogares <- function(.db) {
  .db |>
    # (2.1.1) Pasar nombres a minusculas y reunir los renombrados
    rename_with(tolower) |>
    rename(
      num_personas = nper,
      num_personas_ug = npersug,
      num_cuartos = p5000,
      num_cuartos_dormir = p5010,
      tenencia_vivienda = p5090,
      cuota_amortizacion = p5100
    ) |>
    # (2.1.2) Construir indicadores y montos
    mutate(
      # p5090: 1 = propia totalmente pagada; 2 = propia que estan pagando.
      bin_casa_propia_pagada = as.integer(tenencia_vivienda == 1),
      bin_paga_amortizacion = as.integer(tenencia_vivienda == 2),
      # Tenencia como categoria (seis opciones de la encuesta).
      cat_tenencia = factor(tenencia_vivienda, levels = 1:6,
                            labels = c("Propia_pagada", "Propia_pagando",
                                       "Arriendo", "Usufructo",
                                       "Posesion_sin_titulo", "Otra")),
      # Distingue arriendo estimado del pago efectivo; no indica propiedad.
      bin_arriendo_imput = as.integer(!is.na(p5130)),
      # p5130 y p5140 son excluyentes: cada hogar responde solo una.
      # No sustituir por cero si, inesperadamente, faltan ambas respuestas.
      arriendo = coalesce(p5140, p5130),
      # 98 y 99 son codigos de no sabe / no responde, no montos en pesos.
      bin_arriendo_nsnr = as.integer(arriendo %in% c(98, 99)),
      arriendo = if_else(arriendo %in% c(98, 99), NA_real_, arriendo),
      log_arriendo = log1p(arriendo),
      # Quien no paga cuota aporta cero. Si esta pagando y falta el monto,
      # conservar NA: monto desconocido no equivale a ausencia de pago.
      # Los codigos 98 y 99 tambien se tratan como monto desconocido.
      bin_cuota_nsnr = as.integer(cuota_amortizacion %in% c(98, 99)),
      cuota_amortizacion = case_when(
        cuota_amortizacion %in% c(98, 99) ~ NA_real_,
        tenencia_vivienda != 2 & is.na(cuota_amortizacion) ~ 0,
        TRUE ~ cuota_amortizacion
      ),
      log_amortizacion = log1p(cuota_amortizacion)
    ) |>
    # (2.1.3) Seleccionar explicitamente las variables que conservamos
    # Dominio/depto son categorias geograficas; lp es la linea de pobreza.
    # Las tres estan disponibles tambien en test; no son ingresos del hogar.
    # num_personas_ug es el tamanio pertinente para la unidad de gasto.
    # pobre existe solo en train; se conserva como variable objetivo.
    select(
      id, clase, dominio, depto, lp, num_personas, num_personas_ug,
      num_cuartos, num_cuartos_dormir, tenencia_vivienda, cat_tenencia,
      bin_casa_propia_pagada, bin_paga_amortizacion, bin_arriendo_imput,
      arriendo, log_arriendo, bin_arriendo_nsnr,
      cuota_amortizacion, log_amortizacion, bin_cuota_nsnr,
      any_of("pobre")
    )
}


# (2.2) Procesar personas
procesar_personas <- function(base) {
  base |>
    # (2.2.1) Pasar nombres a minusculas y reunir los renombrados
    rename_with(tolower) |>
    rename(
      id_persona = orden,
      edad = p6040,
      sexo = p6020,
      parentesco_hogar = p6050,
      afiliacion_salud = p6090,
      regimen_salud = p6100,
      tipo_ocupacion = p6430,
      cotiza_pension = p6920,
      horas_laborales = p6800,
      tamanio_firma = p6870,
      segundo_trabajo = p7040,
      horas_segundo_trabajo = p7045,
      tipo_seg_ocupacion = p7050,
      mas_horas_trabajo = p7090,
      pago_horas_extra = p6510,
      pago_primas = p6545,
      pago_bonificaciones = p6580,
      subsidio_alimentos = p6585s1,
      subsidio_transporte = p6585s2,
      subsidio_familiar = p6585s3,
      subsidio_educativo = p6585s4,
      pago_especie_alimento = p6590,
      pago_especie_vivienda = p6600,
      transporte_empresa = p6610,
      pago_especie_otros = p6620,
      prima_servicios = p6630s1,
      prima_navidad = p6630s2,
      prima_vacaciones = p6630s3,
      viaticos_permanentes = p6630s4,
      bonificacion_anual = p6630s6,
      rentas_pensiones = p7495,
      pensionista = p7500s2,
      pension_alimenticia = p7500s3,
      filtro_ingresos_no_laborales = p7505,
      remesas_nacionales = p7510s1,
      remesas_internacionales = p7510s2,
      ayuda_instituciones = p7510s3,
      ingresos_capital = p7510s5
    ) |>
    # (2.2.2) Crear variables
    # Secundaria/media reportan grados acumulados; superior reporta anios.
    # Secundaria con grado cero equivale a primaria completa (cinco anios).
    mutate(
      anios_educ = case_when(
        p6210 %in% c(1, 2) ~ 0,
        p6210 == 3 ~ p6210s1,
        p6210 == 4 & p6210s1 == 0 ~ 5,
        p6210 == 4 & p6210s1 > 0 ~ p6210s1,
        p6210 == 5 ~ p6210s1,
        p6210 == 6 ~ 11 + p6210s1,
        TRUE ~ NA_real_
      ),
      # Conservar el nivel como categoria para posibles usos posteriores.
      nivel_educativo = factor(
        p6210, levels = 1:6, ordered = TRUE,
        labels = c("Ninguno", "Preescolar", "Primaria", "Secundaria",
                   "Media", "Superior")
      ),
      bin_educacion_superior = case_when(
        p6210 == 6 ~ 1L,
        p6210 %in% 1:5 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Primaria como maximo nivel alcanzado, no "al menos primaria".
      bin_educacion_primaria = case_when(
        p6210 == 3 ~ 1L,
        p6210 %in% c(1, 2, 4, 5, 6) ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Jefatura y actividad: Pet/Oc/Des/Ina contienen solo 1 o NA.
      bin_jefe_hogar = as.integer(parentesco_hogar == 1),
      bin_pet = as.integer(!is.na(pet)),
      bin_ocupado = as.integer(!is.na(oc)),
      bin_desocupado = as.integer(!is.na(des)),
      bin_inactivo = as.integer(!is.na(ina)),
      bin_menor6 = as.integer(edad < 6),
      bin_menor18 = as.integer(edad < 18),
      bin_mayor65 = as.integer(edad > 65),
      # No sabe/no informa no equivale a una respuesta negativa.
      bin_afiliado_salud = case_when(
        afiliacion_salud == 1 ~ 1L,
        afiliacion_salud == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_salud_subsidiada = case_when(
        regimen_salud == 3 ~ 1L,
        regimen_salud %in% c(1, 2) | afiliacion_salud == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Salud no contributiva: sin afiliacion O regimen subsidiado.
      # Regimen especial se mantiene separado de esta precariedad.
      bin_salud_no_contributiva = case_when(
        afiliacion_salud == 2 | regimen_salud == 3 ~ 1L,
        regimen_salud %in% c(1, 2) ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_cotiza_pension = case_when(
        cotiza_pension == 1 ~ 1L,
        cotiza_pension %in% c(2, 3) ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Privacion de empleo informal del IPM (DANE, 2014): ocupado que no
      # cotiza a pension. Los pensionados (cotiza_pension = 3) no se privan.
      bin_informal_ipm = as.integer(bin_ocupado == 1 & !(cotiza_pension %in% c(1, 3))),
      # Proxy acordada, solo para ocupados: sin cotizacion a pension Y
      # (sin afiliacion a salud O en regimen subsidiado).
      # Excluir regimen especial y pensionados; no imputar desconocidos.
      bin_informal = case_when(
        bin_ocupado == 0 ~ 0L,
        regimen_salud %in% 2 | cotiza_pension %in% c(1, 3) |
          pensionista %in% 1 ~ 0L,
        bin_cotiza_pension == 0 & bin_salud_no_contributiva == 1 ~ 1L,
        regimen_salud %in% 1 & bin_afiliado_salud == 1 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Posiciones: domestico, sin remuneracion (familia/otros), jornalero.
      # Cuenta propia solo es vulnerable si ademas cumple informalidad.
      bin_trabajador_vulnerable = case_when(
        bin_ocupado == 0 ~ 0L,
        tipo_ocupacion %in% c(3, 6, 7, 8) ~ 1L,
        tipo_ocupacion == 4 ~ bin_informal,
        tipo_ocupacion %in% c(1, 2, 5, 9) ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_segundo_trabajo = case_when(
        segundo_trabajo == 1 ~ 1L,
        segundo_trabajo == 2 | bin_ocupado == 0 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # 98/99 indican desconocimiento, no horas efectivas. Sin segundo
      # trabajo se asigna cero; con segundo trabajo y monto desconocido, NA.
      horas_segundo_trabajo = case_when(
        horas_segundo_trabajo %in% c(98, 99) ~ NA_real_,
        bin_segundo_trabajo == 0 ~ 0,
        TRUE ~ horas_segundo_trabajo
      ),
      # Agrupar alimentos, familiar, educativo y ayudas institucionales.
      # Transporte va aparte. El filtro negativo implica no recibir ayudas;
      # si no hay informacion suficiente, no convertir desconocimiento en cero.
      bin_ayuda_instituciones = case_when(
        ayuda_instituciones == 1 ~ 1L,
        ayuda_instituciones == 2 | filtro_ingresos_no_laborales == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_subsidios = case_when(
        if_any(c(subsidio_alimentos, subsidio_familiar,
                 subsidio_educativo), ~ .x == 1) |
          bin_ayuda_instituciones == 1 ~ 1L,
        if_all(c(subsidio_alimentos, subsidio_familiar, subsidio_educativo),
               ~ is.na(.x) | .x == 2) & bin_ayuda_instituciones == 0 ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_subsidio_transporte = case_when(
        subsidio_transporte == 1 ~ 1L,
        subsidio_transporte == 2 | is.na(subsidio_transporte) ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_rentas_pensiones = case_when(
        rentas_pensiones == 1 ~ 1L,
        rentas_pensiones == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_pensionista = case_when(
        pensionista == 1 ~ 1L,
        pensionista == 2 | rentas_pensiones == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_ingresos_capital = case_when(
        ingresos_capital == 1 ~ 1L,
        ingresos_capital == 2 | filtro_ingresos_no_laborales == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Rentista: filtro de rentas positivo excluyendo AMBAS pensiones,
      # O recepcion de ingresos de capital. Respuestas 9/NA no son negativas.
      bin_rentista = case_when(
        bin_ingresos_capital == 1 |
          (rentas_pensiones == 1 & pensionista == 2 &
             pension_alimenticia == 2) ~ 1L,
        bin_ingresos_capital == 0 &
          (rentas_pensiones == 2 | pensionista %in% 1 |
             pension_alimenticia %in% 1) ~ 0L,
        TRUE ~ NA_integer_
      ),
      bin_remesas = case_when(
        remesas_nacionales == 1 | remesas_internacionales == 1 ~ 1L,
        (remesas_nacionales == 2 & remesas_internacionales == 2) |
          filtro_ingresos_no_laborales == 2 ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Bonificacion mensual O anual; no confundir con primas.
      bin_bonificaciones = case_when(
        pago_bonificaciones == 1 | bonificacion_anual == 1 ~ 1L,
        (pago_bonificaciones == 2 | is.na(pago_bonificaciones)) &
          (bonificacion_anual == 2 | is.na(bonificacion_anual)) ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Indice del borrador: numero de siete pagos laborales recibidos.
      # No es su monto; no sabe/no informa permanece desconocido.
      num_beneficios_laborales = case_when(
        if_any(c(pago_horas_extra, pago_primas, pago_bonificaciones,
                 subsidio_alimentos, subsidio_transporte, subsidio_familiar,
                 subsidio_educativo), ~ .x == 9) ~ NA_real_,
        TRUE ~ (pago_horas_extra %in% 1) + (pago_primas %in% 1) +
          (pago_bonificaciones %in% 1) + (subsidio_alimentos %in% 1) +
          (subsidio_transporte %in% 1) + (subsidio_familiar %in% 1) +
          (subsidio_educativo %in% 1)
      )
    ) |>
    # (2.2.3) Seleccionar las variables que se conservaran
    # Los campos sin prefijo bin_ conservan los codigos de la encuesta.
    select(
      id, id_persona, edad, sexo, parentesco_hogar, anios_educ,
      nivel_educativo, bin_educacion_superior, bin_educacion_primaria,
      bin_jefe_hogar, bin_pet, bin_ocupado, bin_desocupado, bin_inactivo,
      bin_menor6, bin_menor18, bin_mayor65,
      bin_afiliado_salud, regimen_salud, bin_salud_subsidiada,
      bin_salud_no_contributiva,
      tipo_ocupacion, cotiza_pension, bin_cotiza_pension,
      bin_informal, bin_informal_ipm, bin_trabajador_vulnerable,
      horas_laborales, tamanio_firma,
      bin_segundo_trabajo, horas_segundo_trabajo, tipo_seg_ocupacion,
      mas_horas_trabajo,
      pago_horas_extra, pago_primas, pago_bonificaciones,
      bin_bonificaciones, num_beneficios_laborales,
      bin_subsidios, bin_subsidio_transporte,
      pago_especie_alimento, pago_especie_vivienda, transporte_empresa,
      pago_especie_otros, prima_servicios, prima_navidad, prima_vacaciones,
      viaticos_permanentes, bonificacion_anual,
      bin_rentas_pensiones, bin_pensionista, pension_alimenticia,
      bin_ingresos_capital, bin_rentista, bin_remesas,
      bin_ayuda_instituciones
    )
}


# (2.3) Resumenes que conservan desconocidos en vez de inventar ceros
promedio_observado <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}
maximo_observado <- function(x) {
  if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}
hay_alguno <- function(x) {
  if (any(x == 1, na.rm = TRUE)) 1L else if (anyNA(x)) NA_integer_ else 0L
}


# (2.4) Agregar personas: resumenes y conteos, aun sin denominadores del hogar
agregar_personas_hogar <- function(personas) {
  personas <- rename_with(personas, tolower)
  # La edad promedio incluye a TODOS los integrantes, como se solicito.
  edad_hogar <- personas |>
    group_by(id) |>
    summarise(edad_promedio = promedio_observado(edad), .groups = "drop")

  personas |>
    # Las demas agregaciones usan integrantes de la unidad de gasto.
    # Parentesco 7 (pensionista del hogar) no significa jubilacion.
    filter(parentesco_hogar %in% c(1:5, 9)) |>
    group_by(id) |>
    summarise(
      # (2.4.1) Composicion y capacidad laboral
      num_pet_ug = sum(bin_pet),
      num_ocupados_ug = sum(bin_ocupado),
      num_inactivos_ug = sum(bin_inactivo),
      num_desocupados_ug = sum(bin_desocupado),
      num_pensionistas_ug = sum(bin_pensionista[edad >= 10], na.rm = TRUE),
      num_menores6 = sum(edad < 6),
      num_menores10 = sum(edad < 10),
      num_menores18 = sum(edad < 18),
      num_mayores65 = sum(edad > 65),
      edad_promedio_pet = promedio_observado(edad[bin_pet == 1]),
      bin_sin_ocupados = as.integer(num_ocupados_ug == 0),
      # (2.4.2) Jefatura y educacion
      bin_jefe_mujer = as.integer(sexo[bin_jefe_hogar == 1][1] == 2),
      anios_educ_jefe = anios_educ[bin_jefe_hogar == 1][1],
      # Pareja = suma de jefe y conyuge. Sin conyuge, sumar cero.
      # Con educacion desconocida de alguno, conservar NA.
      anios_educ_pareja = anios_educ_jefe + sum(anios_educ[parentesco_hogar == 2]),
      anios_educ_max = maximo_observado(anios_educ),
      anios_educ_promedio_pet = promedio_observado(anios_educ[bin_pet == 1]),
      nivel_educativo_max = maximo_observado(as.integer(nivel_educativo)),
      num_superior_pet = sum(bin_educacion_superior[bin_pet == 1], na.rm = TRUE),
      bin_jefe_educacion_superior = bin_educacion_superior[bin_jefe_hogar == 1][1],
      # (2.4.3) Calidad del empleo: denominadores se aplican postunion
      bin_jefe_informal = bin_informal[bin_jefe_hogar == 1][1],
      bin_jefe_vulnerable = bin_trabajador_vulnerable[bin_jefe_hogar == 1][1],
      bin_jefe_segundo_trabajo = bin_segundo_trabajo[bin_jefe_hogar == 1][1],
      bin_jefe_subsidio_transporte = bin_subsidio_transporte[bin_jefe_hogar == 1][1],
      num_informales = sum(bin_informal[bin_ocupado == 1], na.rm = TRUE),
      num_vulnerables = sum(bin_trabajador_vulnerable[bin_ocupado == 1], na.rm = TRUE),
      num_segundo_trabajo = sum(bin_segundo_trabajo[bin_ocupado == 1], na.rm = TRUE),
      num_cotizantes = sum(bin_cotiza_pension[bin_ocupado == 1], na.rm = TRUE),
      num_asalariados = sum(tipo_ocupacion[bin_ocupado == 1] %in% c(1, 2)),
      num_cuenta_propia = sum(tipo_ocupacion[bin_ocupado == 1] %in% 4),
      num_domesticos = sum(tipo_ocupacion[bin_ocupado == 1] %in% 3),
      num_jornaleros = sum(tipo_ocupacion[bin_ocupado == 1] %in% 8),
      num_mujeres_ocupadas = sum(sexo[bin_ocupado == 1] == 2),
      horas_promedio_ocupados = promedio_observado(horas_laborales[bin_ocupado == 1]),
      beneficios_promedio_ocupados = promedio_observado(num_beneficios_laborales[bin_ocupado == 1]),
      # Deseo de mas horas no es toda la definicion oficial de subempleo.
      bin_desea_mas_horas = hay_alguno(case_when(
        mas_horas_trabajo[bin_ocupado == 1] == 1 ~ 1L,
        mas_horas_trabajo[bin_ocupado == 1] == 2 ~ 0L,
        TRUE ~ NA_integer_
      )),
      # (2.4.4) Al menos un receptor; menores de 10 no responden estos modulos
      bin_hay_bonificaciones = hay_alguno(bin_bonificaciones[edad >= 10]),
      bin_hay_rentista = hay_alguno(bin_rentista[edad >= 10]),
      bin_hay_remesas = hay_alguno(bin_remesas[edad >= 10]),
      bin_hay_subsidios = hay_alguno(bin_subsidios[edad >= 10]),
      bin_hay_ayuda = hay_alguno(bin_ayuda_instituciones[edad >= 10]),
      bin_hay_pension = hay_alguno(bin_pensionista[edad >= 10]),
      bin_hay_rentas_pensiones = hay_alguno(bin_rentas_pensiones[edad >= 10]),
      # Salud: proporciones entre respuestas observadas, no NA como negativos
      pct_salud_subsidiada = 100 * promedio_observado(bin_salud_subsidiada),
      pct_salud_no_contributiva = 100 * promedio_observado(bin_salud_no_contributiva),
      pct_sin_salud = 100 * promedio_observado(1 - bin_afiliado_salud),
      # (2.4.5) Proxies de privaciones del IPM (DANE, 2014); no es el IPM oficial
      bin_bajo_logro_educ = if (sum(edad >= 15) == 0) 1L else
        as.integer(promedio_observado(anios_educ[edad >= 15]) < 9),
      bin_rezago_escolar = hay_alguno(as.integer(
        anios_educ[edad >= 7 & edad <= 17] < edad[edad >= 7 & edad <= 17] - 6
      )),
      bin_trabajo_infantil = hay_alguno(bin_ocupado[edad >= 12 & edad <= 17]),
      bin_empleo_informal_ipm = as.integer(any(bin_informal_ipm == 1)),
      .groups = "drop"
    ) |>
    left_join(edad_hogar, by = "id")
}


# (2.5) Union: una fila de personas agregadas por cada hogar
unir_bases <- function(hogares, personas_hogar) {
  hogares |>
    rename_with(tolower) |>
    left_join(rename_with(personas_hogar, tolower), by = "id")
}


# (2.6) Postunion: porcentajes, dependencia, vivienda y privaciones
procesar_postunion <- function(.db) {
  .db |>
    rename_with(tolower) |>
    mutate(
      # Porcentajes 0-100 sobre la unidad de gasto, no sobre PET ni PEA.
      pct_pet = 100 * num_pet_ug / na_if(num_personas_ug, 0),
      pct_desocupados = 100 * num_desocupados_ug / na_if(num_personas_ug, 0),
      pct_inactivos = 100 * num_inactivos_ug / na_if(num_personas_ug, 0),
      pct_ocupados = 100 * num_ocupados_ug / na_if(num_personas_ug, 0),
      carga_cuidado = num_menores10 + num_mayores65,
      prop_dependientes_potenciales = carga_cuidado / na_if(num_personas_ug, 0),
      # Suma literal solicitada: ocupado y pensionado puede contar dos veces.
      # Es una proxy de fuentes de ingreso, no un porcentaje de personas unicas.
      pct_personas_ingreso = 100 * (num_ocupados_ug + num_pensionistas_ug) /
        na_if(num_personas_ug, 0),
      # Sobre ocupados; sin ocupados, porcentaje no definido (NA), no cero.
      pct_segundo_trabajo = 100 * num_segundo_trabajo / na_if(num_ocupados_ug, 0),
      pct_trabajadores_vulnerables = 100 * num_vulnerables / na_if(num_ocupados_ug, 0),
      pct_informales = 100 * num_informales / na_if(num_ocupados_ug, 0),
      pct_cotizantes = 100 * num_cotizantes / na_if(num_ocupados_ug, 0),
      pct_asalariados = 100 * num_asalariados / na_if(num_ocupados_ug, 0),
      pct_cuenta_propia = 100 * num_cuenta_propia / na_if(num_ocupados_ug, 0),
      pct_domesticos = 100 * num_domesticos / na_if(num_ocupados_ug, 0),
      pct_jornaleros = 100 * num_jornaleros / na_if(num_ocupados_ug, 0),
      pct_mujeres_ocupadas = 100 * num_mujeres_ocupadas / na_if(num_ocupados_ug, 0),
      pct_superior_pet = 100 * num_superior_pet / na_if(num_pet_ug, 0),
      prop_dependencia_pet = (num_personas_ug - num_pet_ug) / na_if(num_pet_ug, 0),
      personas_por_ocupado = num_personas_ug / na_if(num_ocupados_ug, 0),
      nivel_educativo_max = factor(
        nivel_educativo_max, levels = 1:6, ordered = TRUE,
        labels = c("Ninguno", "Preescolar", "Primaria", "Secundaria", "Media", "Superior")
      ),
      # Hacinamiento usa todos los residentes del hogar y sus cuartos para dormir.
      personas_por_dormitorio = num_personas / na_if(num_cuartos_dormir, 0),
      personas_por_cuarto = num_personas / na_if(num_cuartos, 0),
      bin_hacinamiento = case_when(
        clase == 1 & personas_por_dormitorio >= 3 ~ 1L,
        clase == 2 & personas_por_dormitorio > 3 ~ 1L,
        !is.na(personas_por_dormitorio) & clase %in% c(1, 2) ~ 0L,
        TRUE ~ NA_integer_
      ),
      # Conteo de privaciones observadas (las desconocidas no suman)
      num_privaciones = coalesce(bin_hacinamiento, 0L) +
        coalesce(bin_bajo_logro_educ, 0L) + coalesce(bin_rezago_escolar, 0L) +
        coalesce(bin_trabajo_infantil, 0L) + coalesce(bin_empleo_informal_ipm, 0L),
      # Arriendo por persona de la unidad de gasto, en unidades de la linea
      # de pobreza (comparable entre dominios), y arriendo por cuarto
      arriendo_pc_lp = arriendo / na_if(num_personas_ug, 0) / lp,
      log_arriendo_cuarto = log1p(arriendo / na_if(num_cuartos, 0))
    )
}


# (2.7) Preparar las bases para modelar
fun_preparar_modelo <- function(.db, .niveles_dominio, .niveles_depto) {
  #' Tipos y faltantes para modelar
  #'
  #' @description Convierte dominio y depto en factores con los mismos
  #' niveles en train y test (los niveles se pasan como argumentos). Los NA se reemplazan para que todos los
  #' modelos corran (caret y model.matrix no aceptan NA):
  #' - Variables definidas sobre ocupados (pct_*, promedios): 0 cuando el
  #'   hogar no tiene ocupados; bin_sin_ocupados ya identifica esos hogares.
  #' - personas_por_ocupado sin ocupados: num_personas_ug.
  #' - Indicadoras bin_hay_* desconocidas: 0, con una indicadora na_* que
  #'   marca el desconocimiento.
  #' - Arriendo y cuota con codigos 98/99: 0; bin_arriendo_nsnr y
  #'   bin_cuota_nsnr ya los marcan.
  #' - Resto (menos de 0,1% de los hogares): 0 en numericas y "Ninguno" en
  #'   nivel_educativo_max.
  vars_bin_hay <- c("bin_hay_rentista", "bin_hay_pension", "bin_hay_rentas_pensiones",
                    "bin_hay_ayuda", "bin_hay_remesas", "bin_hay_subsidios",
                    "bin_hay_bonificaciones")
  .db |>
    mutate(
      dominio = factor(dominio, levels = .niveles_dominio),
      depto = factor(depto, levels = .niveles_depto),
      across(all_of(vars_bin_hay), ~ as.integer(is.na(.x)), .names = "na_{.col}"),
      personas_por_ocupado = coalesce(personas_por_ocupado, as.numeric(num_personas_ug)),
      # Nivel educativo maximo desconocido (menos de 5 hogares): "Ninguno"
      nivel_educativo_max = replace(nivel_educativo_max, is.na(nivel_educativo_max), "Ninguno"),
      across(where(is.numeric), ~ replace_na(.x, 0))
    ) |>
    # tenencia_vivienda queda representada por cat_tenencia
    select(-tenencia_vivienda)
}


# ------------------------------------------------------------------------------
# (3) Aplicar el procesamiento a entrenamiento y test
# ------------------------------------------------------------------------------

# (3.1) Hogares: misma funcion para ambas muestras; conservar las bases originales
db_train_hogares_clean <- fun_preprocess_hogares(db_train_hogares)
db_test_hogares_clean <- fun_preprocess_hogares(db_test_hogares)

# (3.2) Personas
db_train_personas_clean <- procesar_personas(db_train_personas)
db_test_personas_clean <- procesar_personas(db_test_personas)

# ------------------------------------------------------------------------------
# (4) Union: preparar conteos y pegar horizontalmente una fila por hogar
# ------------------------------------------------------------------------------

db_train_personas_hogar <- agregar_personas_hogar(db_train_personas_clean)
db_test_personas_hogar <- agregar_personas_hogar(db_test_personas_clean)
db_train_hogares_clean <- unir_bases(db_train_hogares_clean, db_train_personas_hogar)
db_test_hogares_clean <- unir_bases(db_test_hogares_clean, db_test_personas_hogar)


# ------------------------------------------------------------------------------
# (5) Procesamiento postunion: proporciones, carga de cuidado y vivienda
# ------------------------------------------------------------------------------

db_train_hogares_clean <- procesar_postunion(db_train_hogares_clean)
db_test_hogares_clean <- procesar_postunion(db_test_hogares_clean)


# ------------------------------------------------------------------------------
# (6) Preparar las bases para modelar: tipos, faltantes y variable objetivo
# ------------------------------------------------------------------------------

# Los niveles de los factores geograficos se fijan con train, para que train
# y test tengan las mismas categorias.
niveles_dominio <- sort(unique(db_train_hogares_clean$dominio))
niveles_depto   <- sort(unique(db_train_hogares_clean$depto))

# caret exige que los niveles de la respuesta sean nombres validos de R.
train_clean <- db_train_hogares_clean |>
  fun_preparar_modelo(niveles_dominio, niveles_depto) |>
  mutate(Pobre = factor(pobre, levels = c(0, 1), labels = c("No", "Si"))) |>
  select(-pobre)
test_clean <- db_test_hogares_clean |>
  fun_preparar_modelo(niveles_dominio, niveles_depto)

# Verificacion: mismo orden de hogares que la base original y sin NA
stopifnot(identical(train_clean$id, db_train_hogares$id),
          identical(test_clean$id, db_test_hogares$id),
          sum(is.na(train_clean)) == 0, sum(is.na(test_clean)) == 0)


# ------------------------------------------------------------------------------
# (7) Exportar las bases finales: una fila por hogar
# ------------------------------------------------------------------------------

# CSV para inspeccion. Al leerlos de nuevo los factores quedan como texto;
# los scripts siguientes usan los objetos en memoria creados por main.R.
dir.create(dir_clean, recursive = TRUE, showWarnings = FALSE)
write_csv(train_clean, file.path(dir_clean, "train_clean.csv"))
write_csv(test_clean, file.path(dir_clean, "test_clean.csv"))
