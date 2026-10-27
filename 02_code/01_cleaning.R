# ------------------------------------------------------------------------------
# Archivo: 01_cleaning.R
# Proyecto: Taller 2 - Big Data y Machine Learning
#
# Descripcion:
#   Importar las cuatro bases y procesar hogares y personas.
#   Construir variables individuales, agregar personas, unir y procesar hogares.
#   Estimar ingreso individual con random forest sobre las variables originales,
#   y construir ingreso UG, per capita y relativo a la linea de pobreza.
# Input:
#   train_hogares.csv, train_personas.csv, test_hogares.csv y test_personas.csv
#   en 01_data/01_raw/. Paquetes y dir_datos_raw preparados por main.R.
# Output:
#   Cuatro bases originales y sus versiones procesadas con sufijo _clean.
#   Hogares: tamanio, vivienda, arriendo, amortizacion y pobre (solo en train).
#   Personas: identificadores, educacion, actividad, condiciones laborales,
#   proteccion social y recepcion de pagos.
#   train_clean y test_clean; archivos CSV en 01_data/02_clean/.
#   03_outputs/ingreso_personas/modelo_ingreso.rds: bosque, esquema de entradas,
#   folds por hogar y predicciones de train.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# (1) Importar las bases originales
# ------------------------------------------------------------------------------

# (1.1) Entrenamiento
db_train_hogares <- read_csv(
  file.path(dir_datos_raw, "train_hogares.csv"), show_col_types = FALSE
)
db_train_personas <- read_csv(
  file.path(dir_datos_raw, "train_personas.csv"), show_col_types = FALSE
)

# (1.2) Test
db_test_hogares <- read_csv(
  file.path(dir_datos_raw, "test_hogares.csv"), show_col_types = FALSE
)
db_test_personas <- read_csv(
  file.path(dir_datos_raw, "test_personas.csv"), show_col_types = FALSE
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
      # Distingue arriendo estimado del pago efectivo; no indica propiedad.
      bin_arriendo_imput = as.integer(!is.na(p5130)),
      # p5130 y p5140 son excluyentes: cada hogar responde solo una.
      # No sustituir por cero si, inesperadamente, faltan ambas respuestas.
      arriendo = coalesce(p5140, p5130),
      log_arriendo = log1p(arriendo),
      # Quien no paga cuota aporta cero. Si esta pagando y falta el monto,
      # conservar NA: monto desconocido no equivale a ausencia de pago.
      cuota_amortizacion = if_else(
        tenencia_vivienda != 2 & is.na(cuota_amortizacion), 0, cuota_amortizacion
      ),
      log_amortizacion = log1p(cuota_amortizacion),
      # Componente del ingreso, NO pago efectivo de arriendo: se agrega una
      # sola vez por hogar tras sumar personas. Regla del Codigo Taller 2:
      # 1 = propia pagada, 4 = usufructo; 98/99 no son montos utilizables.
      # Es una aproximacion de Ingpcug, no una replica exacta de todos sus ajustes.
      arriendo_imputado_ingreso = if_else(
        tenencia_vivienda %in% c(1, 4) & !(p5130 %in% c(98, 99)),
        replace_na(p5130, 0), 0
      )
    ) |>
    # (2.1.3) Seleccionar explicitamente las variables que conservamos
    # Dominio/depto son categorias geograficas; lp es la linea de pobreza.
    # Las tres estan disponibles tambien en test; no son ingresos del hogar.
    # num_personas_ug es el tamanio pertinente para la unidad de gasto.
    # pobre existe solo en train; se conserva como variable objetivo.
    select(
      id, clase, dominio, depto, lp, num_personas, num_personas_ug,
      num_cuartos, num_cuartos_dormir, tenencia_vivienda, bin_casa_propia_pagada,
      bin_paga_amortizacion, bin_arriendo_imput,
      arriendo, log_arriendo, cuota_amortizacion, log_amortizacion,
      arriendo_imputado_ingreso,
      any_of("pobre")
    )
}


# (2.2) Procesar personas
procesar_personas <- function(base, ingresos) {
  #' Procesar personas y adjuntar ingreso individual predicho.
  #'
  #' @description Los predictores laborales/educativos se procesan aqui, pero
  #' ingreso_predicho se estima ANTES y en otra rama, usando las columnas RAW.
  #' El bosque predice log(1 + Ingtot) en PET; su salida se transforma a pesos
  #' con max(exp(prediccion) - 1, 0). No PET recibe cero, como el codigo original.
  #' En train las personas heredan el fold del hogar: ningun miembro de ese
  #' hogar participa en el bosque que genera su propia prediccion de ingreso.
  #' Fuera de muestra no equivale a validacion anidada de toda la cadena.
  #'
  #' @param base Base original de personas, sin eliminar predictores de ingreso.
  #' @param ingresos Tabla id, id_persona, ingreso_predicho de generar_ingreso_personas.
  #' @returns Personas procesadas con ingreso_predicho mensual individual en pesos.
  base |>
    # (2.2.1) Pasar nombres a minusculas y reunir los renombrados
    rename_with(tolower) |>
    left_join(ingresos, by = c("id", "orden" = "id_persona")) |>
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
      # Version historica SOLO para comparar con el Codigo Taller 2.
      # No reemplaza anios_educ: aqui desconocido se vuelve "ninguno" y se
      # aplican los topes/sumas del borrador, aun cuando difieren de nuestra regla.
      nivel_educativo_viejo = factor(
        if_else(is.na(p6210) | p6210 == 9, 1, p6210), levels = 1:6,
        ordered = TRUE, labels = c("Ninguno", "Preescolar", "Primaria",
                                   "Secundaria", "Media", "Universitaria")
      ),
      anios_educ_viejo = case_when(
        is.na(p6210) | p6210 %in% c(1, 2, 9) ~ 0,
        p6210 == 3 ~ pmin(replace_na(p6210s1, 0), 5),
        p6210 == 4 ~ 5 + pmin(replace_na(p6210s1, 0), 4),
        p6210 == 5 ~ 9 + pmin(replace_na(p6210s1, 0), 2),
        p6210 == 6 ~ 11 + pmin(replace_na(p6210s1, 0), 6),
        TRUE ~ 0
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
      # PET oficial DANE: autoridad para este proyecto, no redefinir por edad.
      bin_pet = as.integer(!is.na(pet)),
      bin_ocupado = as.integer(!is.na(oc)),
      # Sin ocupacion actual, horas de ese empleo = cero. Si esta ocupado y
      # no informa horas, conservar NA: no equivale a trabajar cero horas.
      horas_laborales = if_else(bin_ocupado == 0, 0, horas_laborales),
      # Oficio es nominal. Separar no ocupado de ocupado sin respuesta.
      oficio = case_when(
        bin_ocupado == 0 ~ "No_ocupado",
        is.na(oficio) ~ "Desconocido",
        TRUE ~ as.character(oficio)
      ),
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
        bin_ocupado == 0 ~ 0,
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
    # Nuevos agregados y las demas decisiones pendientes se revisaran despues.
    select(
      id, id_persona, ingreso_predicho, edad, sexo, parentesco_hogar, oficio, anios_educ,
      nivel_educativo, nivel_educativo_viejo, anios_educ_viejo,
      bin_educacion_superior, bin_educacion_primaria,
      bin_jefe_hogar, bin_pet, bin_ocupado, bin_desocupado, bin_inactivo,
      bin_menor6, bin_menor18, bin_mayor65,
      bin_afiliado_salud, regimen_salud, bin_salud_subsidiada,
      bin_salud_no_contributiva,
      tipo_ocupacion, cotiza_pension, bin_cotiza_pension,
      bin_informal, bin_trabajador_vulnerable,
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
  #' Agregar integrantes de la unidad de gasto, incluido su ingreso predicho.
  #'
  #' @description num_ing_ug_hat suma ingresos individuales predichos en pesos.
  #' Se excluyen parentescos 6 (servicio domestico y parientes), 7 (pensionistas
  #' que pagan alojamiento, NO jubilados) y 8 (trabajadores del negocio del hogar
  #' que no comparten la unidad de gasto). NO se excluye a los ocupados en general:
  #' hijos, pareja o jefe que trabajan siguen incluidos si pertenecen a la UG.
  #' El arriendo imputado NO se suma por persona: se incorpora despues de unir
  #' hogares, una vez por hogar, al construir el ingreso per capita.
  #'
  #' @param personas Personas procesadas con ingreso_predicho generado previamente.
  #' @returns Una fila por hogar con num_ing_ug_hat y los otros agregados de personas.
  personas <- rename_with(personas, tolower)
  # La edad promedio incluye a TODOS los integrantes, como se solicito.
  edad_hogar <- personas |>
    group_by(id) |>
    summarise(edad_promedio = promedio_observado(edad), .groups = "drop")

  # Agregados educativos historicos sobre TODOS los residentes, como el original.
  # Versiones actuales mantienen la definicion y el universo de unidad de gasto.
  educacion_vieja_hogar <- personas |>
    group_by(id) |>
    summarise(
      nivel_educativo_max_viejo = max(nivel_educativo_viejo, na.rm = TRUE),
      anios_educ_max_viejo = max(anios_educ_viejo, na.rm = TRUE),
      anios_educ_jefe_viejo = max(anios_educ_viejo * bin_jefe_hogar, na.rm = TRUE),
      anios_educ_promedio_pet_viejo = replace_na(
        mean(anios_educ_viejo[bin_pet == 1], na.rm = TRUE), 0),
      prop_superior_pet_viejo = replace_na(
        sum((nivel_educativo_viejo == "Universitaria") * bin_pet, na.rm = TRUE) /
          na_if(sum(bin_pet, na.rm = TRUE), 0), 0),
      bin_bajo_logro_educ_viejo = if (sum(edad >= 15, na.rm = TRUE) == 0) 1L else
        as.integer(mean(anios_educ_viejo[edad >= 15], na.rm = TRUE) < 9),
      bin_rezago_escolar_viejo = as.integer(any(
        between(edad, 7, 17) & anios_educ_viejo < edad - 6, na.rm = TRUE)),
      .groups = "drop"
    )

  personas |>
    # Las demas agregaciones usan integrantes de la unidad de gasto.
    # Parentesco 7 (pensionista del hogar) no significa jubilacion.
    filter(parentesco_hogar %in% c(1:5, 9)) |>
    group_by(id) |>
    summarise(
      # (2.4.1) Composicion y capacidad laboral
      num_pet_ug = sum(bin_pet),
      num_ing_ug_hat = sum(ingreso_predicho),
      num_ocupados_ug = sum(bin_ocupado),
      num_inactivos_ug = sum(bin_inactivo),
      num_desocupados_ug = sum(bin_desocupado),
      num_pensionistas_ug = sum(bin_pensionista[edad >= 10]),
      num_menores6 = sum(edad < 6),
      num_menores10 = sum(edad < 10),
      num_menores18 = sum(edad < 18),
      num_mayores65 = sum(edad > 65),
      edad_promedio_pet = promedio_observado(edad[bin_pet == 1]),
      bin_sin_ocupados = as.integer(num_ocupados_ug == 0),
      # (2.4.2) Jefatura y educacion
      bin_jefe_mujer = as.integer(sexo[bin_jefe_hogar == 1][1] == 2),
      oficio_jefe = oficio[parentesco_hogar == 1][1],
      # Sin pareja no es no ocupado. Multiples parejas con oficios distintos
      # dejan NA; no seleccionar arbitrariamente una de las personas.
      oficio_pareja = if (!any(parentesco_hogar == 2)) "Sin_pareja" else
        if (n_distinct(oficio[parentesco_hogar == 2]) == 1)
          oficio[parentesco_hogar == 2][1] else NA_character_,
      anios_educ_jefe = anios_educ[bin_jefe_hogar == 1][1],
      # Pareja = suma de jefe y conyuge. Sin conyuge, sumar cero.
      # Con educacion desconocida de alguno, conservar NA.
      anios_educ_pareja = anios_educ_jefe + sum(anios_educ[parentesco_hogar == 2]),
      anios_educ_max = maximo_observado(anios_educ),
      anios_educ_promedio_pet = promedio_observado(anios_educ[bin_pet == 1]),
      nivel_educativo_max = maximo_observado(as.integer(nivel_educativo)),
      num_superior_pet = sum(bin_educacion_superior[bin_pet == 1]),
      bin_jefe_educacion_superior = bin_educacion_superior[bin_jefe_hogar == 1][1],
      # (2.4.3) Calidad del empleo: denominadores se aplican postunion
      bin_jefe_informal = bin_informal[bin_jefe_hogar == 1][1],
      bin_jefe_vulnerable = bin_trabajador_vulnerable[bin_jefe_hogar == 1][1],
      bin_jefe_segundo_trabajo = bin_segundo_trabajo[bin_jefe_hogar == 1][1],
      bin_jefe_subsidio_transporte = bin_subsidio_transporte[bin_jefe_hogar == 1][1],
      num_informales = sum(bin_informal[bin_ocupado == 1]),
      num_vulnerables = sum(bin_trabajador_vulnerable[bin_ocupado == 1]),
      num_segundo_trabajo = sum(bin_segundo_trabajo[bin_ocupado == 1]),
      num_cotizantes = sum(bin_cotiza_pension[bin_ocupado == 1]),
      num_asalariados = sum(tipo_ocupacion[bin_ocupado == 1] %in% c(1, 2)),
      num_cuenta_propia = sum(tipo_ocupacion[bin_ocupado == 1] %in% 4),
      num_domesticos = sum(tipo_ocupacion[bin_ocupado == 1] %in% 3),
      num_jornaleros = sum(tipo_ocupacion[bin_ocupado == 1] %in% 8),
      num_mujeres_ocupadas = sum(sexo[bin_ocupado == 1] == 2),
      # Convencion aprobada: sin ocupados, promedio cero y bin_sin_ocupados=1.
      # Con ocupados pero todos sus datos desconocidos, conservar NA.
      horas_promedio_ocupados = if (num_ocupados_ug == 0) 0 else
        promedio_observado(horas_laborales[bin_ocupado == 1]),
      beneficios_promedio_ocupados = if (num_ocupados_ug == 0) 0 else
        promedio_observado(num_beneficios_laborales[bin_ocupado == 1]),
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
      # Salud: proporciones entre respuestas observadas, no NA como negativos
      pct_salud_subsidiada = 100 * promedio_observado(bin_salud_subsidiada),
      pct_salud_no_contributiva = 100 * promedio_observado(bin_salud_no_contributiva),
      pct_sin_salud = 100 * promedio_observado(1 - bin_afiliado_salud),
      # (2.4.5) Proxies de privaciones del borrador; no constituyen el IPM oficial
      bin_bajo_logro_educ = if (sum(edad >= 15) == 0) 1L else
        as.integer(mean(anios_educ[edad >= 15]) < 9),
      bin_rezago_escolar = hay_alguno(as.integer(
        anios_educ[edad >= 7 & edad <= 17] < edad[edad >= 7 & edad <= 17] - 6
      )),
      bin_trabajo_infantil = hay_alguno(bin_ocupado[edad >= 12 & edad <= 17]),
      .groups = "drop"
    ) |>
    left_join(edad_hogar, by = "id") |>
    left_join(educacion_vieja_hogar, by = "id")
}


# (2.5) Union: una fila de personas agregadas por cada hogar
unir_bases <- function(hogares, personas_hogar) {
  hogares |>
    rename_with(tolower) |>
    left_join(rename_with(personas_hogar, tolower), by = "id")
}


# (2.6) Postunion: porcentajes, dependencia y vivienda
procesar_postunion <- function(.db) {
  #' Construir razones y el ingreso per capita predicho despues de unir hogares.
  #'
  #' @description num_ing_pc_hat = (num_ing_ug_hat + arriendo imputado) / Npersug.
  #' El arriendo se suma UNA vez, solo para tenencia 1/4 y sin codigos 98/99.
  #' Se replica el denominador pmax(Npersug, 1) del codigo de referencia.
  #' num_ratio_lp_hat = num_ing_pc_hat / lp: ingreso estimado relativo al umbral,
  #' no su inversa; valores menores de 1 sugieren ingreso inferior a la linea.
  #' Son predictores generados, no ingresos observados ni decisiones de pobreza.
  #'
  #' @param .db Hogares unidos con num_ing_ug_hat, lp, num_personas_ug y arriendo.
  #' @returns Hogares con las tres variables de ingreso; elimina el componente
  #' temporal arriendo_imputado_ingreso para no duplicarlo en la base final.
  .db |>
    rename_with(tolower) |>
    mutate(
      num_ing_pc_hat = (num_ing_ug_hat + arriendo_imputado_ingreso) /
        pmax(num_personas_ug, 1),
      num_ratio_lp_hat = num_ing_pc_hat / lp,
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
      # Sobre ocupados: cero si no los hay (convencion predictiva aprobada),
      # distinguiendo ese caso con bin_sin_ocupados. Otros desconocidos siguen NA.
      pct_segundo_trabajo = if_else(num_ocupados_ug == 0, 0,
        100 * num_segundo_trabajo / na_if(num_ocupados_ug, 0)),
      pct_trabajadores_vulnerables = if_else(num_ocupados_ug == 0, 0,
        100 * num_vulnerables / na_if(num_ocupados_ug, 0)),
      pct_informales = if_else(num_ocupados_ug == 0, 0,
        100 * num_informales / na_if(num_ocupados_ug, 0)),
      pct_cotizantes = if_else(num_ocupados_ug == 0, 0,
        100 * num_cotizantes / na_if(num_ocupados_ug, 0)),
      pct_asalariados = if_else(num_ocupados_ug == 0, 0,
        100 * num_asalariados / na_if(num_ocupados_ug, 0)),
      pct_cuenta_propia = if_else(num_ocupados_ug == 0, 0,
        100 * num_cuenta_propia / na_if(num_ocupados_ug, 0)),
      pct_domesticos = if_else(num_ocupados_ug == 0, 0,
        100 * num_domesticos / na_if(num_ocupados_ug, 0)),
      pct_jornaleros = if_else(num_ocupados_ug == 0, 0,
        100 * num_jornaleros / na_if(num_ocupados_ug, 0)),
      pct_mujeres_ocupadas = if_else(num_ocupados_ug == 0, 0,
        100 * num_mujeres_ocupadas / na_if(num_ocupados_ug, 0)),
      pct_superior_pet = 100 * num_superior_pet / na_if(num_pet_ug, 0),
      prop_dependencia_pet = (num_personas_ug - num_pet_ug) / na_if(num_pet_ug, 0),
      personas_por_ocupado = num_personas_ug / na_if(num_ocupados_ug, 0),
      nivel_educativo_max = factor(
        nivel_educativo_max, levels = 1:6, ordered = TRUE,
        labels = c("Ninguno", "Preescolar", "Primaria", "Secundaria", "Media", "Superior")
      ),
      # Hacinamiento usa todos los residentes del hogar y sus cuartos para dormir.
      personas_por_cuarto = num_personas / na_if(num_cuartos_dormir, 0),
      bin_hacinamiento = case_when(
        clase == 1 & personas_por_cuarto >= 3 ~ 1L,
        clase == 2 & personas_por_cuarto > 3 ~ 1L,
        !is.na(personas_por_cuarto) & clase %in% c(1, 2) ~ 0L,
        TRUE ~ NA_integer_
      )
    ) |>
    select(-arriendo_imputado_ingreso)
}


# (2.7) Funcion que procesa ingreso: train ajusta y guarda; test carga y aplica
generar_ingreso_personas <- function(base, entrenamiento, archivo_modelo,
                                     referencia_test = NULL, folds_hogar = NULL) {
  #' Estimar ingreso individual con las mismas entradas y parametros del codigo viejo.
  #'
  #' @description Usa personas RAW, antes de nuestro select y de crear anios_educ.
  #' Respuesta: log1p(ingtot); PET se identifica por pet no faltante. Predictores:
  #' columnas del test salvo id, orden, fex_c, fex_dpto, depto y pet. Educacion
  #' entra como p6210/p6210s1 originales. Factores nominales: dominio, oficio,
  #' p6050, p6100, p6240, p6430, p7050 y p7350; NA numericos = -1 y categoricos
  #' = "NA". Como la referencia, sus niveles se fijan con union de train y test.
  #' Bosque fijo: 200 arboles, mtry 15, nodo minimo 50, categorias ordenadas por
  #' ranger y semilla 2026. NO se calibran estos hiperparametros por CV.
  #' Train: predice cada fold de hogares con los otros cuatro y luego ajusta un
  #' bosque final con toda PET de train para test. Guarda bosque, niveles, folds
  #' y predicciones de train. Test solo carga ese archivo: no usa ingtot ni ajusta.
  #' La transformacion inversa exp(prediccion)-1 reproduce la referencia; no
  #' aplica correccion de retransformacion, por lo que no garantiza ingreso medio.
  #' Este OOF global replica el original. Para evaluar la cadena con CV posterior,
  #' regenerar la etapa de ingreso DENTRO del entrenamiento de cada fold externo:
  #' pegar estas columnas una vez no garantiza ausencia de filtracion entre etapas.
  #'
  #' @param base Personas originales de train o test, sin filtrar columnas.
  #' @param entrenamiento TRUE: ajustar y guardar; FALSE: cargar y predecir.
  #' @param archivo_modelo Ruta RDS del bosque y su esquema de entradas.
  #' @param referencia_test Personas originales de test, necesarias solo en train.
  #' @param folds_hogar Tabla id/fold con cinco folds de hogares, solo en train.
  #' @returns Tabla id, id_persona e ingreso_predicho mensual en pesos. No PET = 0.
  base <- rename_with(base, tolower)
  resultado <- transmute(base, id, id_persona = orden, ingreso_predicho = 0)
  # Usar la PET oficial; cualquier alternativa de edad requiere otro nombre
  # y aprobacion expresa. No sustituir ni recodificar la definicion del DANE.
  es_pet <- !is.na(base$pet)
  personas <- base[es_pet, ]

  if (entrenamiento) {
    referencia_test <- rename_with(referencia_test, tolower)
    variables <- setdiff(names(referencia_test),
                          c("id", "orden", "fex_c", "fex_dpto", "depto", "pet"))
    categoricas <- c("dominio", "oficio", "p6050", "p6100", "p6240",
                     "p6430", "p7050", "p7350")
    niveles <- list()
    for (v in categoricas) {
      valores <- c(as.character(personas[[v]]),
                    as.character(referencia_test[[v]][!is.na(referencia_test$pet)]))
      valores[is.na(valores)] <- "NA"
      niveles[[v]] <- sort(unique(valores))
    }
  } else {
    guardado <- readRDS(archivo_modelo)
    variables <- guardado$variables
    categoricas <- guardado$categoricas
    niveles <- guardado$niveles
  }

  datos <- as.data.frame(personas[variables])
  for (v in categoricas) {
    valor <- as.character(datos[[v]])
    valor[is.na(valor)] <- "NA"
    if (any(!valor %in% niveles[[v]])) stop("Categoria nueva no prevista en ", v)
    datos[[v]] <- factor(valor, levels = niveles[[v]])
  }
  for (v in setdiff(variables, categoricas)) {
    datos[[v]] <- replace_na(as.numeric(datos[[v]]), -1)
  }

  if (entrenamiento) {
    datos$y_log_ingtot <- log1p(personas$ingtot)
    fold_persona <- folds_hogar$fold[match(personas$id, folds_hogar$id)]
    stopifnot(!anyNA(fold_persona), setequal(unique(fold_persona), 1:5),
              !anyDuplicated(folds_hogar$id), all(is.finite(datos$y_log_ingtot)))
    formula_ingreso <- reformulate(variables, response = "y_log_ingtot")
    y_log_hat <- rep(NA_real_, nrow(datos))
    # k=0 ajusta el bosque definitivo; los otros cinco generan predicciones OOF.
    for (k in c(1:5, 0)) {
      ajustar <- if (k == 0) rep(TRUE, nrow(datos)) else fold_persona != k
      bosque <- ranger(
        formula = formula_ingreso, data = datos[ajustar, , drop = FALSE],
        num.trees = 200, mtry = 15, min.node.size = 50,
        respect.unordered.factors = "order", seed = 2026
      )
      if (k != 0) {
        y_log_hat[!ajustar] <- predict(bosque, data = datos[!ajustar, , drop = FALSE])$predictions
        message("Ingreso individual: fold ", k, "/5 terminado.")
      }
    }
    resultado$ingreso_predicho[es_pet] <- pmax(expm1(y_log_hat), 0)
    dir.create(dirname(archivo_modelo), recursive = TRUE, showWarnings = FALSE)
    saveRDS(list(modelo = bosque, variables = variables, categoricas = categoricas,
                 niveles = niveles, folds_hogar = folds_hogar,
                 predicciones_train = resultado, semilla = 2026), archivo_modelo)
  } else if (nrow(datos) > 0) {
    y_log_hat <- predict(guardado$modelo, data = datos)$predictions
    resultado$ingreso_predicho[es_pet] <- pmax(expm1(y_log_hat), 0)
  }
  resultado
}


# ------------------------------------------------------------------------------
# (3) Aplicar el procesamiento a entrenamiento y test
# ------------------------------------------------------------------------------

# (3.1) Hogares: misma funcion para ambas muestras; conservar las bases originales
db_train_hogares_clean <- fun_preprocess_hogares(db_train_hogares)
db_test_hogares_clean <- fun_preprocess_hogares(db_test_hogares)

# (3.2) Ingreso sobre RAW: mismos cinco folds de hogares que el codigo de referencia
set.seed(2026)
folds_ingreso <- createFolds(factor(db_train_hogares$Pobre, levels = 0:1),
                             k = 5, returnTrain = FALSE)
folds_hogar_ingreso <- data.frame(id = db_train_hogares$id, fold = NA_integer_)
for (k in 1:5) folds_hogar_ingreso$fold[folds_ingreso[[k]]] <- k
archivo_ingreso <- file.path(dir_resultados, "ingreso_personas", "modelo_ingreso.rds")
ingresos_train <- generar_ingreso_personas(
  db_train_personas, TRUE, archivo_ingreso,
  referencia_test = db_test_personas, folds_hogar = folds_hogar_ingreso)
ingresos_test <- generar_ingreso_personas(db_test_personas, FALSE, archivo_ingreso)

# (3.3) Procesamiento habitual: adjuntar por persona sin eliminar columnas de RAW
db_train_personas_clean <- procesar_personas(db_train_personas, ingresos_train)
db_test_personas_clean <- procesar_personas(db_test_personas, ingresos_test)

# ------------------------------------------------------------------------------
# (4) Union: preparar conteos y pegar horizontalmente una fila por hogar
# ------------------------------------------------------------------------------

db_train_personas_hogar <- agregar_personas_hogar(db_train_personas_clean)
db_test_personas_hogar <- agregar_personas_hogar(db_test_personas_clean)
db_train_hogares_clean <- unir_bases(db_train_hogares_clean, db_train_personas_hogar)
db_test_hogares_clean <- unir_bases(db_test_hogares_clean, db_test_personas_hogar)


# ------------------------------------------------------------------------------
# (5) Procesamiento postunion: proporciones y carga de cuidado
# ------------------------------------------------------------------------------

db_train_hogares_clean <- procesar_postunion(db_train_hogares_clean)
db_test_hogares_clean <- procesar_postunion(db_test_hogares_clean)


# ------------------------------------------------------------------------------
# (6) Exportar las bases finales: una fila por hogar
# ------------------------------------------------------------------------------

train_clean <- db_train_hogares_clean
test_clean <- db_test_hogares_clean
dir_clean <- file.path(root, "01_data", "02_clean")
dir.create(dir_clean, recursive = TRUE, showWarnings = FALSE)
write_csv(train_clean, file.path(dir_clean, "train_clean.csv"))
write_csv(test_clean, file.path(dir_clean, "test_clean.csv"))
