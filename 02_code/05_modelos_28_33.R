# ------------------------------------------------------------------------------
# Archivo: 05_modelos_28_33.R
# Proyecto: Taller 2 - Big Data y Machine Learning
# Descripcion:
#   Entrenar seis bloques de modelos, seleccionar sus ganadores y exportar resultados.
#   Conserva las especificaciones iniciales y las posteriores con ingreso RF.
# Input:
#   Bases originales en 01_data/01_raw/; rutas y paquetes de main.R.
#   Funciones comunes de exportacion definidas en 04_modelos_1_27.R.
# Output:
#   Modelos 28-33, CSV id/pobre, parametros, CV y matrices de confusion en
#   03_outputs/entrega_final/. Intermedios en cache/entrenamiento_28_33/.
#   No necesita modelos, predicciones, matrices ni resultados previamente guardados.
# ------------------------------------------------------------------------------

local({


# ------------------------------------------------------------------------------
# (1) Definir las transformaciones y exportaciones comunes
# ------------------------------------------------------------------------------


stopifnot(exists("registrar_modelo"), exists("metricas_entrega"))
# Aislar objetos del 04; ambos bloques alimentan los mismos registros y Excel.
dir_resultados <- file.path(dir_entrega, "cache", "entrenamiento_28_33")
dir.create(dir_resultados, recursive = TRUE, showWarnings = FALSE)
set.seed(2026)

fun_preprocess_hogares <- function(.db) {
  .db |>
    # Pasar nombres a minusculas y reunir los renombrados
    rename_with(tolower) |>
    rename(
      num_personas = nper,
      num_personas_ug = npersug,
      num_cuartos = p5000,
      num_cuartos_dormir = p5010,
      tenencia_vivienda = p5090,
      cuota_amortizacion = p5100
    ) |>
    # Construir indicadores y montos
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
    # Seleccionar explicitamente las variables que conservamos
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


# Procesar personas
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
    # Pasar nombres a minusculas y reunir los renombrados
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
    # Crear variables
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
    # Seleccionar las variables que se conservaran
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


# Resumenes que conservan desconocidos en vez de inventar ceros
promedio_observado <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}
maximo_observado <- function(x) {
  if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}
hay_alguno <- function(x) {
  if (any(x == 1, na.rm = TRUE)) 1L else if (anyNA(x)) NA_integer_ else 0L
}


# Agregar personas: resumenes y conteos, aun sin denominadores del hogar
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
      # Composicion y capacidad laboral
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
      # Jefatura y educacion
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
      # Calidad del empleo: denominadores se aplican postunion
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
      # Al menos un receptor; menores de 10 no responden estos modulos
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
      # Proxies de privaciones del borrador; no constituyen el IPM oficial
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


# Union: una fila de personas agregadas por cada hogar
unir_bases <- function(hogares, personas_hogar) {
  hogares |>
    rename_with(tolower) |>
    left_join(rename_with(personas_hogar, tolower), by = "id")
}


# Postunion: porcentajes, dependencia y vivienda
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


# Funcion que procesa ingreso: train ajusta y guarda; test carga y aplica
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
      niveles[[v]] <- sort(unique(c(valores, "NA")))
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
    # Un oficio ausente del entrenamiento externo se trata como desconocido.
    # No usar ingresos ni respuestas de validacion para ampliar el bosque.
    valor[!valor %in% niveles[[v]]] <- "NA"
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



especificaciones <- tribble(
  ~numero, ~nombre, ~carpeta, ~comparacion, ~archivo, ~tipo,
  28, "EN_logit_seleccion_lasso", "lasso_elastic_net", "comparacion_modelos.csv", "ganador.rds", "glmnet",
  29, "Random_forest_seleccion_lasso", "random_forest", "comparacion_rf.csv", "ganador_rf.rds", "ranger",
  30, "XGBoost_base", "xgboost", "comparacion_xgb.csv", "configuracion_ganador.rds", "xgb",
  31, "LightGBM_base", "lightgbm", "comparacion_lgb.csv", "configuracion_ganador.rds", "lgb",
  32, "EN_logit_ingreso_predicho", "logit_en_consolidado", "validacion.csv", "ganador.rds", "glmnet",
  33, "XGBoost_ingreso_predicho", "xgb_consolidado", "validacion.csv", "configuracion.rds", "xgb"
)

exportar_adicional <- function(numero) {
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
  fila$base <- if (h$numero <= 31) "Variables iniciales sin ingreso predicho" else
    "Base consolidada con ingreso RF y matrices propias por fold"
  fila$etapa_ingreso <- if (h$numero <= 31) "Sin ingreso predicho agregado de la consolidacion posterior" else
    "Ingreso RF recalculado dentro de la CV del clasificador; cache guardada"
  fila$F1_CV <- ganador$f1_cv
  fila$f1_cv <- NULL
  fila$F1_OOB <- NA_real_
  # Calcular componentes del mismo F1 CV; no sustituirlos por metricas de validacion.
  archivo_oof <- if (h$numero == 29) paste0("oof_", ganador$version, ".csv") else
    if (h$numero %in% 30:31) paste0("oof_", ganador$configuracion, ".csv") else
      if (h$numero %in% 32:33) "oof_ganador.csv" else "oof_ganador.csv"
  ruta_oof <- file.path(carpeta, archivo_oof)
  for (n in c("Precision", "Recall", "Especificidad", "Accuracy", "VP", "FP", "FN", "VN")) {
    fila[[paste0(n, "_CV")]] <- NA_real_
  }
  stopifnot(file.exists(ruta_oof))
  if (file.exists(ruta_oof)) {
    oof <- read_csv(ruta_oof, show_col_types = FALSE)
    fila <- agregar_metricas(fila, metricas_entrega(oof$pobre, oof$probabilidad, ganador$corte))
    fila$estado_metricas_CV <- "Completa: OOF de la ejecucion actual"
  } else fila$estado_metricas_CV <- "Solo F1 disponible: no se conservaron predicciones OOF; no se reestima"
  fila$punto_corte <- ganador$corte
  fila$corte <- NULL
  fila$criterio_hiperparametros <- "F1 OOF y corte conjuntamente; rondas con parada interna cuando aplica"
  fila$criterio_ganador_bloque <- if (h$numero %in% 31:33) "F1 CV" else
    "F1 de validacion interna entre candidatos; desempate por F1 CV"
  fila$metodo_corte <- "Maximo F1 sobre predicciones OOF; grilla 0.02 a 0.98, paso 0.02"
  fila$n_train <- nrow(train_completo)
  fila$n_CV <- if (!is.null(registro$grupos)) sum(registro$grupos$grupo %in% 2:4) else NA_integer_
  if (h$numero %in% 29:31) {
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
  fila$archivo_origen <- file.path("03_outputs", "entrega_final", "cache", "entrenamiento_28_33", h$carpeta, h$archivo)
  fila$huella_resultado <- unname(tools::md5sum(file.path(carpeta, h$archivo)))
  fila$comparabilidad <- "Entrenado en la ejecucion actual; base y particiones indicadas. Distintas bases no son comparacion controlada"
  paquete <- switch(h$tipo, xgb = "xgboost", lgb = "lightgbm", h$tipo)
  fila$version_paquete <- as.character(packageVersion(paquete))
  fila$version_R <- R.version.string
  fila$script <- file.path("02_code", "05_modelos_28_33.R")
  fila$huella_script <- unname(tools::md5sum(file.path(root, fila$script)))

  # Grilla realmente guardada, cuando existe; no reconstruir una grilla inexistente.
  grilla <- data.frame()
  busqueda <- file.path(carpeta, "busqueda_cv.csv")
  if (file.exists(busqueda)) {
    grilla <- read_csv(busqueda, show_col_types = FALSE)
    fila$origen_grilla <- file.path("03_outputs", "entrega_final", "cache", "entrenamiento_28_33", h$carpeta, "busqueda_cv.csv")
  } else {
    fila$origen_grilla <- "No se guardo la grilla completa; comparar candidatos en el archivo de origen"
    grilla <- tabla
  }
  if (h$numero == 28) {
    # El archivo de resultados solo guardo el ganador de cada candidato.
    # Completar la grilla DOCUMENTADA por el codigo conservado, distinguiendo
    # su procedencia de una captura inmutable de aquella ejecucion guardada.
    # Documentacion del diseno original; no depende de scripts fuera del pipeline.
    entorno <- list(alphas_en = seq(0.125, 0.875, by = 0.125),
                    lambdas = 10^seq(0, -4, length.out = 25),
                    configuracion = data.frame(
                      nombre = c("Lasso_MPL", "Lasso_logit", "EN_MPL_sel_lineal",
                                 "EN_logit_sel_lineal", "EN_MPL_sel_logit", "EN_logit_sel_logit"),
                      familia = rep(c("gaussian", "binomial"), 3),
                      seleccion = c("todas", "todas", "gaussian", "gaussian", "binomial", "binomial")))
    grilla <- bind_rows(lapply(seq_len(nrow(entorno$configuracion)), function(j) {
      candidato <- entorno$configuracion[j, ]
      expand.grid(candidato = candidato$nombre, familia = candidato$familia,
                   seleccion = candidato$seleccion,
                   alpha = if (candidato$seleccion == "todas") 1 else entorno$alphas_en,
                   lambda = entorno$lambdas, stringsAsFactors = FALSE)
    }))
    fila$origen_grilla <- "Diseno original documentado en 05_modelos_28_33.R; no captura inmutable de la ejecucion"
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

  modelo_exportar <- registro$modelo
  if (h$tipo == "xgb") {
    nombre_modelo <- if (h$numero == 30) "ganador_xgb.ubj" else "ganador.ubj"
    modelo_exportar <- xgb.load(file.path(carpeta, nombre_modelo))
  }
  if (h$tipo == "lgb") modelo_exportar <- lgb.load(file.path(carpeta, "ganador_lgb.txt"))
  registrar_modelo(h$numero, h$nombre, pred, fila, variables, grilla,
                    importancia, metrica, modelo = modelo_exportar)
  rm(registro, pred, importancia)
  gc(verbose = FALSE)
}


# ------------------------------------------------------------------------------
# (2) Construir las dos bases desde los archivos originales
# ------------------------------------------------------------------------------


# Transformaciones propias de estas especificaciones: las primeras no usan ingreso;
# las posteriores usan ingreso RF. No sustituirlo por el ingreso XGBoost del 04.
raw_train_hogares <- read_csv(file.path(dir_datos_raw, "train_hogares.csv"), show_col_types = FALSE, guess_max = Inf)
raw_test_hogares <- read_csv(file.path(dir_datos_raw, "test_hogares.csv"), show_col_types = FALSE, guess_max = Inf)
raw_train_personas <- read_csv(file.path(dir_datos_raw, "train_personas.csv"), show_col_types = FALSE, guess_max = Inf)
raw_test_personas <- read_csv(file.path(dir_datos_raw, "test_personas.csv"), show_col_types = FALSE, guess_max = Inf)
base_hogar_adicional <- function(hogares, personas) {
  # Cero solo permite usar la agregacion comun; estas columnas de ingreso se
  # excluyen de la base inicial y se reemplazan por estimaciones RF en 32-33.
  ceros <- personas |> rename_with(tolower) |>
    transmute(id, id_persona = orden, ingreso_predicho = 0)
  ph <- agregar_personas_hogar(procesar_personas(personas, ceros))
  procesar_postunion(unir_bases(fun_preprocess_hogares(hogares), ph))
}
train_completo <- base_hogar_adicional(raw_train_hogares, raw_train_personas)
test_completo <- base_hogar_adicional(raw_test_hogares, raw_test_personas)
# Predictores sustantivos de la especificacion inicial, antes de su ampliacion.
vars_base <- c("clase", "num_personas", "num_personas_ug", "num_cuartos",
  "num_cuartos_dormir", "tenencia_vivienda", "bin_casa_propia_pagada", "bin_paga_amortizacion",
  "bin_arriendo_imput", "arriendo", "log_arriendo", "cuota_amortizacion",
  "log_amortizacion", "num_pet_ug", "num_ocupados_ug", "num_inactivos_ug",
  "num_desocupados_ug", "num_pensionistas_ug", "num_menores6", "num_menores10",
  "num_menores18", "num_mayores65", "edad_promedio_pet", "bin_sin_ocupados",
  "bin_jefe_mujer", "anios_educ_jefe", "anios_educ_pareja", "anios_educ_max",
  "anios_educ_promedio_pet", "nivel_educativo_max", "num_superior_pet", "bin_jefe_educacion_superior",
  "bin_jefe_informal", "bin_jefe_vulnerable", "bin_jefe_segundo_trabajo", "bin_jefe_subsidio_transporte",
  "num_informales", "num_vulnerables", "num_segundo_trabajo", "num_cotizantes",
  "num_asalariados", "num_cuenta_propia", "num_domesticos", "num_jornaleros",
  "num_mujeres_ocupadas", "horas_promedio_ocupados", "beneficios_promedio_ocupados", "bin_desea_mas_horas",
  "bin_hay_bonificaciones", "bin_hay_rentista", "bin_hay_remesas", "bin_hay_subsidios",
  "bin_hay_ayuda", "bin_hay_pension", "pct_salud_subsidiada", "pct_salud_no_contributiva",
  "pct_sin_salud", "bin_bajo_logro_educ", "bin_rezago_escolar", "bin_trabajo_infantil",
  "edad_promedio", "pct_pet", "pct_desocupados", "pct_inactivos",
  "pct_ocupados", "carga_cuidado", "prop_dependientes_potenciales", "pct_personas_ingreso",
  "pct_segundo_trabajo", "pct_trabajadores_vulnerables", "pct_informales", "pct_cotizantes",
  "pct_asalariados", "pct_cuenta_propia", "pct_domesticos", "pct_jornaleros",
  "pct_mujeres_ocupadas", "pct_superior_pet", "prop_dependencia_pet", "personas_por_ocupado",
  "personas_por_cuarto", "bin_hacinamiento", "dominio", "lp",
  "depto")
train_base <- train_completo |> select(id, pobre, all_of(vars_base))
test_base <- test_completo |> select(id, all_of(vars_base))


# ------------------------------------------------------------------------------
# (3) Modelo 28: lasso y elastic net; seleccionar y entrenar el ganador
# ------------------------------------------------------------------------------


set.seed(2026)
inicio <- Sys.time()
train <- train_base
test <- test_base
y <- train$pobre
stopifnot(!anyNA(y), all(y %in% c(0, 1)))

numero_folds <- 5
alphas_en <- c(0.125,0.25,0.375, 0.50,0.625, 0.75,0.875)
lambdas <- 10^seq(0, -4, length.out = 25)
cortes <- seq(0.02, 0.98, by = 0.02)
# Sin rebalanceo inicial. F1 y el corte atienden el objetivo de clasificacion.

# Tres grupos estratificados e independientes: 20/60/20 aproximadamente
# Seleccion: elegir columnas de los EN. Ajuste: CV de los seis competidores.
# Validacion: comparar los seis con parametros y cortes ya congelados.
grupo <- integer(nrow(train))
for (clase in 0:1) {
  filas <- which(y == clase)
  grupo[filas] <- sample(rep(1:5, length.out = length(filas)))
}
i_seleccion <- which(grupo == 1)
i_ajuste <- which(grupo %in% 2:4)
i_validacion <- which(grupo == 5)

# ------------------------------------------------------------------------------
# Construir la matriz de predictores, sin usar pobre ni id
# ------------------------------------------------------------------------------

predictores <- setdiff(names(train), c("id", "pobre"))
categoricas <- intersect(c("clase", "dominio", "depto", "tenencia_vivienda",
                          "nivel_educativo_max", "nivel_educativo_max_viejo",
                          "oficio_jefe", "oficio_pareja"),
                        predictores)
numericas <- setdiff(predictores, categoricas)
datos_train <- as.data.frame(train[predictores])
datos_test <- as.data.frame(test[predictores])
niveles <- list()

# Codigos nominales se convierten en dummies. No usar los niveles de test
# para definir el modelo; una categoria nueva se trata como desconocida.
for (v in categoricas) {
  niveles[[v]] <- c(sort(unique(na.omit(as.character(datos_train[[v]])))),
                    "Desconocido")
  for (muestra in c("train", "test")) {
    datos <- if (muestra == "train") datos_train else datos_test
    valor <- as.character(datos[[v]])
    valor[is.na(valor) | !valor %in% niveles[[v]]] <- "Desconocido"
    datos[[v]] <- factor(valor, levels = niveles[[v]])
    if (muestra == "train") datos_train <- datos else datos_test <- datos
  }
}

# Conservar el patron de ausencia. La mediana se aprende DENTRO de cada fold.
for (v in numericas) {
  datos_train[[v]][!is.finite(datos_train[[v]])] <- NA_real_
  datos_test[[v]][!is.finite(datos_test[[v]])] <- NA_real_
  datos_train[[paste0(v, "_faltante")]] <- as.integer(is.na(datos_train[[v]]))
  datos_test[[paste0(v, "_faltante")]] <- as.integer(is.na(datos_test[[v]]))
}
formula_x <- ~ . - 1
x <- model.matrix(formula_x, model.frame(formula_x, datos_train, na.action = na.pass))
x_test <- model.matrix(formula_x, model.frame(formula_x, datos_test, na.action = na.pass))
x_test <- x_test[, colnames(x), drop = FALSE]

# ------------------------------------------------------------------------------
# Dos funciones para evitar repetir imputacion y CV en los seis modelos
# ------------------------------------------------------------------------------

# La referencia siempre es entrenamiento; nunca validacion ni test.
imputar_matriz <- function(matriz, medianas = NULL) {
  if (is.null(medianas)) {
    medianas <- apply(matriz, 2, median, na.rm = TRUE)
    medianas[!is.finite(medianas)] <- 0
    # Si una columna esta enteramente ausente en entrenamiento, usar cero;
    # su indicadora de faltante conserva esa informacion.
  }
  for (j in seq_len(ncol(matriz))) {
    matriz[is.na(matriz[, j]), j] <- medianas[j]
  }
  list(x = matriz, medianas = medianas)
}

ajustar_cv <- function(x, y, familia, alphas, folds, ponderar = FALSE) {
  mejor <- list(f1_cv = -Inf)
  for (a in alphas) {
    oof <- matrix(NA_real_, nrow(x), length(lambdas))
    for (k in seq_len(numero_folds)) {
      entrenamiento <- folds != k
      preparado <- imputar_matriz(x[entrenamiento, , drop = FALSE])
      evaluacion <- imputar_matriz(x[!entrenamiento, , drop = FALSE],
                                  preparado$medianas)$x
      # Balancear la contribucion de ambas clases solo con datos del fold.
      pesos <- rep(1, sum(entrenamiento))
      if (ponderar) {
        frecuencias <- table(factor(y[entrenamiento], levels = 0:1))
        pesos <- sum(entrenamiento) / (2 * frecuencias[y[entrenamiento] + 1])
      }
      modelo <- glmnet(preparado$x, y[entrenamiento], family = familia,
                       alpha = a, lambda = lambdas, standardize = TRUE,
                       weights = as.numeric(pesos))
      oof[!entrenamiento, ] <- predict(modelo, evaluacion, s = lambdas,
                                      type = "response")
    }
    # MPL puede salir de [0,1]: acotar solo para clasificar, no al estimar.
    oof[] <- pmin(1, pmax(0, oof))
    for (j in seq_along(lambdas)) {
      # Conteos acumulados: todos los cortes sin recorrer de nuevo cada fila.
      intervalo <- findInterval(oof[, j], cortes)
      tp <- rev(cumsum(rev(tabulate(intervalo[y == 1] + 1,
                                    nbins = length(cortes) + 1))))[-1]
      fp <- rev(cumsum(rev(tabulate(intervalo[y == 0] + 1,
                                    nbins = length(cortes) + 1))))[-1]
      fn <- sum(y == 1) - tp
      f1 <- 2 * tp / pmax(2 * tp + fp + fn, 1)
      t <- which.max(f1)
      # Empates: conservar la primera combinacion (penalizacion mayor).
      if (f1[t] > mejor$f1_cv) {
        mejor <- list(alpha = a, lambda = lambdas[j], corte = cortes[t],
                      f1_cv = f1[t], prob_oof = oof[, j])
      }
    }
  }
  preparado <- imputar_matriz(x)
  pesos <- rep(1, length(y))
  if (ponderar) {
    frecuencias <- table(factor(y, levels = 0:1))
    pesos <- length(y) / (2 * frecuencias[y + 1])
  }
  mejor$modelo <- glmnet(preparado$x, y, family = familia, alpha = mejor$alpha,
                         lambda = lambdas, standardize = TRUE,
                         weights = as.numeric(pesos))
  mejor$medianas <- preparado$medianas
  mejor$columnas <- colnames(x)
  mejor$familia <- familia
  mejor$ponderar <- ponderar
  mejor
}

# ------------------------------------------------------------------------------
# Seleccionar columnas por lasso, en la muestra independiente de seleccion
# ------------------------------------------------------------------------------

folds_seleccion <- integer(length(i_seleccion))
for (clase in 0:1) {
  filas <- which(y[i_seleccion] == clase)
  folds_seleccion[filas] <- sample(rep(1:numero_folds, length.out = length(filas)))
}
seleccion <- list()
columnas <- list()
for (familia in c("gaussian", "binomial")) {
  seleccion[[familia]] <- ajustar_cv(x[i_seleccion, , drop = FALSE],
                                    y[i_seleccion], familia, 1, folds_seleccion)
  resultado <- seleccion[[familia]]
  coeficientes <- as.matrix(coef(resultado$modelo, s = resultado$lambda))[, 1]
  columnas[[familia]] <- setdiff(names(coeficientes)[coeficientes != 0], "(Intercept)")
  if (!length(columnas[[familia]])) {
    stop("Lasso ", familia, " selecciono solo intercepto; revise antes de ajustar EN.")
  }
}

# ------------------------------------------------------------------------------
# Competir: dos lasso completos y cuatro EN con columnas seleccionadas
# ------------------------------------------------------------------------------

configuracion <- data.frame(
  nombre = c("Lasso_MPL", "Lasso_logit", "EN_MPL_sel_lineal", "EN_logit_sel_lineal",
             "EN_MPL_sel_logit", "EN_logit_sel_logit"),
  familia = rep(c("gaussian", "binomial"), 3),
  seleccion = c("todas", "todas", "gaussian", "gaussian", "binomial", "binomial")
)
folds_ajuste <- integer(length(i_ajuste))
for (clase in 0:1) {
  filas <- which(y[i_ajuste] == clase)
  folds_ajuste[filas] <- sample(rep(1:numero_folds, length.out = length(filas)))
}
ajustes <- list()
comparacion <- list()
for (m in seq_len(nrow(configuracion))) {
  nombre <- configuracion$nombre[m]
  origen <- configuracion$seleccion[m]
  vars <- if (origen == "todas") colnames(x) else columnas[[origen]]
  # glmnet exige al menos dos columnas; la constante adicional no es predictor.
  xm <- x[, vars, drop = FALSE]
  if (ncol(xm) == 1) xm <- cbind(xm, auxiliar_cero = 0)
  message("Ajustando ", nombre, " (", length(vars), " columnas).")
  ajuste <- ajustar_cv(xm[i_ajuste, , drop = FALSE], y[i_ajuste],
                        configuracion$familia[m],
                        if (origen == "todas") 1 else alphas_en, folds_ajuste)
  xv <- imputar_matriz(xm[i_validacion, , drop = FALSE], ajuste$medianas)$x
  prob <- as.numeric(predict(ajuste$modelo, xv, s = ajuste$lambda, type = "response"))
  pred <- as.integer(pmin(1, pmax(0, prob)) >= ajuste$corte)
  verdad <- y[i_validacion]
  tp <- sum(pred == 1 & verdad == 1)
  fp <- sum(pred == 1 & verdad == 0)
  fn <- sum(pred == 0 & verdad == 1)
  comparacion[[nombre]] <- data.frame(
    modelo = nombre, columnas = length(vars), alpha = ajuste$alpha,
    lambda = ajuste$lambda, corte = ajuste$corte, f1_cv = ajuste$f1_cv,
    f1_validacion = 2 * tp / max(2 * tp + fp + fn, 1),
    precision = if (tp + fp == 0) NA_real_ else tp / (tp + fp),
    sensibilidad = tp / (tp + fn), accuracy = mean(pred == verdad)
  )
  ajustes[[nombre]] <- ajuste
}
comparacion <- bind_rows(comparacion) |>
  arrange(desc(f1_validacion), desc(f1_cv))
print(comparacion)
ganador <- comparacion$modelo[1]
message("Ganador en validacion interna: ", ganador)
# Esta validacion sirve para elegir modelo; el F1 del ganador puede ser optimista
# por comparar seis alternativas. El test oficial no tiene etiquetas para medirlo.

# ------------------------------------------------------------------------------
# Reestimar el ganador con todo train y aplicar al test oficial
# ------------------------------------------------------------------------------

elegido <- ajustes[[ganador]]
x_final <- x
xt_final <- x_test
if ("auxiliar_cero" %in% elegido$columnas) {
  x_final <- cbind(x_final, auxiliar_cero = 0)
  xt_final <- cbind(xt_final, auxiliar_cero = 0)
}
x_final <- imputar_matriz(x_final[, elegido$columnas, drop = FALSE])
xt_final <- imputar_matriz(xt_final[, elegido$columnas, drop = FALSE],
                           x_final$medianas)$x
modelo_final <- glmnet(x_final$x, y, family = elegido$familia,
                       alpha = elegido$alpha, lambda = lambdas, standardize = TRUE)
prob_test <- as.numeric(predict(modelo_final, xt_final, s = elegido$lambda,
                                type = "response"))
predicciones <- data.frame(id = test$id, puntaje = pmin(1, pmax(0, prob_test)))
predicciones$pobre <- as.integer(predicciones$puntaje >= elegido$corte)
# Se conserva el corte OOF; no se vuelve a optimizar con predicciones in-sample.
# En MPL, puntaje es una prediccion acotada, no probabilidad calibrada garantizada.

salida <- file.path(dir_resultados, "lasso_elastic_net")
dir.create(salida, recursive = TRUE, showWarnings = FALSE)
write_csv(comparacion, file.path(salida, "comparacion_modelos.csv"))
write_csv(predicciones, file.path(salida, "predicciones_test.csv"))
write_csv(predicciones[c("id", "pobre")], file.path(salida, "clasificacion_test.csv"))
write_csv(data.frame(variable = columnas$gaussian),
          file.path(salida, "seleccion_lasso_lineal.csv"))
write_csv(data.frame(variable = columnas$binomial),
          file.path(salida, "seleccion_lasso_logit.csv"))
saveRDS(list(modelo = modelo_final, configuracion = elegido[c("alpha", "lambda",
            "corte", "familia", "columnas")], medianas = x_final$medianas,
             niveles = niveles, predictores = predictores,
             seleccion = columnas, comparacion = comparacion,
             grupos = data.frame(id = train$id, grupo = grupo), semilla = 2026),
        file.path(salida, "ganador.rds"))
message("Minutos transcurridos: ", round(as.numeric(difftime(Sys.time(), inicio,
                                                          units = "mins")), 1))


write_csv(data.frame(id = train$id[i_ajuste], fold = folds_ajuste,
  pobre = y[i_ajuste], probabilidad = ajustes[[ganador]]$prob_oof),
  file.path(salida, "oof_ganador.csv"))
exportar_adicional(28)


# ------------------------------------------------------------------------------
# (4) Modelo 29: random forest con todas las variables y con seleccion lasso
# ------------------------------------------------------------------------------


set.seed(2026)
inicio_rf <- Sys.time()
arboles_rf <- 1000
nodo_rf <- 20
cortes_rf <- seq(0.02, 0.98, by = 0.02)
numero_folds_rf <- 5
nucleos_rf <- parallel::detectCores(logical = FALSE)
if (is.na(nucleos_rf)) nucleos_rf <- 2L
hilos_rf <- max(1L, nucleos_rf - 1L)
# Paralelizar arboles; folds y versiones se ejecutan secuencialmente.
# Dejar un nucleo fisico libre. Reducir hilos_rf si falta memoria.

seleccion_rf <- read_csv(file.path(dir_resultados, "lasso_elastic_net",
                                   "seleccion_lasso_lineal.csv"),
                         show_col_types = FALSE)$variable
stopifnot(length(seleccion_rf) > 0,
          all(seleccion_rf %in% colnames(x)),
          length(folds_ajuste) == length(i_ajuste),
          setequal(unique(folds_ajuste), 1:numero_folds_rf))
versiones_rf <- list(todas = colnames(x), seleccion_lasso = seleccion_rf)
salida_rf <- file.path(dir_resultados, "random_forest")
dir.create(salida_rf, recursive = TRUE, showWarnings = FALSE)

# Guardar identificadores, no solo la semilla: permiten compartir los mismos folds.
particiones_rf <- data.frame(id = train$id, grupo = grupo, fold_cv = NA_integer_)
particiones_rf$fold_cv[i_ajuste] <- folds_ajuste
write_csv(particiones_rf, file.path(salida_rf, "particiones.csv"))
saveRDS(list(x = x, x_test = x_test, y = y, train_id = train$id,
             test_id = test$id, particiones = particiones_rf,
             seleccion = seleccion_rf, niveles = niveles, predictores = predictores),
        file.path(salida_rf, "insumos_compartidos.rds"))
# Este RDS conserva el diseno exacto y las particiones para otros computadores.

# ------------------------------------------------------------------------------
# CV: cinco folds, 1000 arboles en cada uno y corte por F1
# ------------------------------------------------------------------------------

comparacion_rf <- list()
tiempos_rf <- list()
for (version in names(versiones_rf)) {
  vars <- versiones_rf[[version]]
  mtry_rf <- max(1L, floor(sqrt(length(vars))))
  message("Version ", version, ": ", length(vars), " columnas; mtry=", mtry_rf,
           "; hilos=", hilos_rf)
  prob_oof_rf <- rep(NA_real_, length(i_ajuste))

  for (k in 1:numero_folds_rf) {
    inicio_fold <- Sys.time()
    dentro <- folds_ajuste != k
    filas_train <- i_ajuste[dentro]
    filas_eval <- i_ajuste[!dentro]
    preparado_rf <- imputar_matriz(x[filas_train, vars, drop = FALSE])
    evaluacion_rf <- imputar_matriz(x[filas_eval, vars, drop = FALSE],
                                     preparado_rf$medianas)$x
    bosque_rf <- ranger(
      x = preparado_rf$x, y = factor(y[filas_train], levels = 0:1),
      probability = TRUE, num.trees = arboles_rf, mtry = mtry_rf,
      min.node.size = nodo_rf, max.depth = 0, splitrule = "gini",
      replace = TRUE, sample.fraction = 1, importance = "none",
      oob.error = FALSE, num.threads = hilos_rf, seed = 2026
    )
    prob_oof_rf[!dentro] <- predict(bosque_rf, data = evaluacion_rf,
                                   num.threads = hilos_rf)$predictions[, "1"]
    minutos <- as.numeric(difftime(Sys.time(), inicio_fold, units = "mins"))
    tiempos_rf[[paste(version, k)]] <- data.frame(
      version = version, etapa = paste0("fold_", k), minutos = minutos)
    message("Fold ", k, "/5 terminado: ", round(minutos, 1), " minutos.")
    rm(bosque_rf)
    gc(verbose = FALSE)
  }

  # Solo se busca el corte; los parametros del bosque no se ajustan en una grilla.
  f1_cortes_rf <- numeric(length(cortes_rf))
  verdad_rf <- y[i_ajuste]
  for (j in seq_along(cortes_rf)) {
    pred_rf <- as.integer(prob_oof_rf >= cortes_rf[j])
    tp <- sum(pred_rf == 1 & verdad_rf == 1)
    fp <- sum(pred_rf == 1 & verdad_rf == 0)
    fn <- sum(pred_rf == 0 & verdad_rf == 1)
    f1_cortes_rf[j] <- 2 * tp / max(2 * tp + fp + fn, 1)
  }
  mejor_corte_rf <- which.max(f1_cortes_rf)
  corte_rf <- cortes_rf[mejor_corte_rf]
  write_csv(data.frame(id = train$id[i_ajuste], fold = folds_ajuste,
                       pobre = verdad_rf, probabilidad = prob_oof_rf),
            file.path(salida_rf, paste0("oof_", version, ".csv")))
  write_csv(data.frame(corte = cortes_rf, f1_cv = f1_cortes_rf),
            file.path(salida_rf, paste0("cortes_", version, ".csv")))

  # ---------------------------------------------------------------------------
  # Comparar en la misma validacion interna, con corte ya congelado
  # ---------------------------------------------------------------------------

  inicio_validacion <- Sys.time()
  preparado_rf <- imputar_matriz(x[i_ajuste, vars, drop = FALSE])
  evaluacion_rf <- imputar_matriz(x[i_validacion, vars, drop = FALSE],
                                   preparado_rf$medianas)$x
  bosque_rf <- ranger(
    x = preparado_rf$x, y = factor(y[i_ajuste], levels = 0:1),
    probability = TRUE, num.trees = arboles_rf, mtry = mtry_rf,
    min.node.size = nodo_rf, max.depth = 0, splitrule = "gini",
    replace = TRUE, sample.fraction = 1, importance = "none",
    oob.error = FALSE, num.threads = hilos_rf, seed = 2026
  )
  prob_rf <- predict(bosque_rf, data = evaluacion_rf,
                     num.threads = hilos_rf)$predictions[, "1"]
  pred_rf <- as.integer(prob_rf >= corte_rf)
  verdad_rf <- y[i_validacion]
  tp <- sum(pred_rf == 1 & verdad_rf == 1)
  fp <- sum(pred_rf == 1 & verdad_rf == 0)
  fn <- sum(pred_rf == 0 & verdad_rf == 1)
  tn <- sum(pred_rf == 0 & verdad_rf == 0)
  comparacion_rf[[version]] <- data.frame(
    version = version, columnas = length(vars), mtry = mtry_rf,
    arboles = arboles_rf, min_node_size = nodo_rf, corte = corte_rf,
    f1_cv = f1_cortes_rf[mejor_corte_rf],
    f1_validacion = 2 * tp / max(2 * tp + fp + fn, 1),
    precision = if (tp + fp == 0) NA_real_ else tp / (tp + fp),
    sensibilidad = tp / (tp + fn), especificidad = tn / (tn + fp),
    accuracy = mean(pred_rf == verdad_rf),
    verdaderos_positivos = tp, falsos_positivos = fp,
    falsos_negativos = fn, verdaderos_negativos = tn
  )
  saveRDS(list(modelo = bosque_rf, medianas = preparado_rf$medianas,
               columnas = vars, corte = corte_rf),
          file.path(salida_rf, paste0("modelo_validacion_", version, ".rds")))
  tiempos_rf[[paste(version, "validacion")]] <- data.frame(
    version = version, etapa = "validacion",
    minutos = as.numeric(difftime(Sys.time(), inicio_validacion, units = "mins")))
  write_csv(bind_rows(comparacion_rf), file.path(salida_rf, "comparacion_rf.csv"))
  write_csv(bind_rows(tiempos_rf), file.path(salida_rf, "tiempos.csv"))
  rm(bosque_rf)
  gc(verbose = FALSE)
}

comparacion_rf <- bind_rows(comparacion_rf) |>
  arrange(desc(f1_validacion), desc(f1_cv))
write_csv(comparacion_rf, file.path(salida_rf, "comparacion_rf.csv"))
print(comparacion_rf)
# La validacion ya se uso para otros modelos: es desarrollo, no evaluacion final
# independiente. Test oficial no tiene etiquetas para calcular F1.

# ------------------------------------------------------------------------------
# Bosque final ganador, importancia por permutacion y prediccion de test
# ------------------------------------------------------------------------------

inicio_final <- Sys.time()
ganador_rf <- comparacion_rf$version[1]
vars <- versiones_rf[[ganador_rf]]
preparado_rf <- imputar_matriz(x[, vars, drop = FALSE])
evaluacion_rf <- imputar_matriz(x_test[, vars, drop = FALSE],
                                 preparado_rf$medianas)$x
bosque_final_rf <- ranger(
  x = preparado_rf$x, y = factor(y, levels = 0:1),
  probability = TRUE, num.trees = arboles_rf,
  mtry = max(1L, floor(sqrt(length(vars)))),
  min.node.size = nodo_rf, max.depth = 0, splitrule = "gini",
  replace = TRUE, sample.fraction = 1, importance = "permutation",
  oob.error = TRUE, num.threads = hilos_rf, seed = 2026
)
# En ranger probabilistico, la importancia por permutacion mide deterioro OOB
# del Brier, NO caida de F1. Se calcula solo en el final para ahorrar tiempo.
importancia_rf <- data.frame(
  variable = names(bosque_final_rf$variable.importance),
  importancia_permutacion_brier = as.numeric(bosque_final_rf$variable.importance)
) |>
  arrange(desc(importancia_permutacion_brier))
prob_test_rf <- predict(bosque_final_rf, data = evaluacion_rf,
                        num.threads = hilos_rf)$predictions[, "1"]
predicciones_rf <- data.frame(id = test$id, probabilidad = prob_test_rf,
                              pobre = as.integer(prob_test_rf >= comparacion_rf$corte[1]))
write_csv(importancia_rf, file.path(salida_rf, "importancia_permutacion_brier.csv"))
write_csv(predicciones_rf, file.path(salida_rf, "predicciones_test.csv"))
write_csv(predicciones_rf[c("id", "pobre")],
          file.path(salida_rf, "clasificacion_test.csv"))
saveRDS(list(modelo = bosque_final_rf, version = ganador_rf, columnas = vars,
             corte = comparacion_rf$corte[1], medianas = preparado_rf$medianas,
             niveles = niveles, predictores = predictores, semilla = 2026,
             comparacion = comparacion_rf), file.path(salida_rf, "ganador_rf.rds"))
tiempos_rf[["final"]] <- data.frame(
  version = ganador_rf, etapa = "final_con_importancia",
  minutos = as.numeric(difftime(Sys.time(), inicio_final, units = "mins")))
write_csv(bind_rows(tiempos_rf), file.path(salida_rf, "tiempos.csv"))
message("Ganador: ", ganador_rf, "; minutos totales: ",
         round(as.numeric(difftime(Sys.time(), inicio_rf, units = "mins")), 1))
# min.node.size=20 fija el minimo para INTENTAR dividir un nodo; pueden resultar
# hojas menores de 20. min.bucket seria el parametro para un minimo por hoja.


exportar_adicional(29)


# ------------------------------------------------------------------------------
# (5) Modelo 30: XGBoost sin ingreso predicho
# ------------------------------------------------------------------------------


seleccion_xgb <- read_csv(file.path(dir_resultados, "lasso_elastic_net",
                                    "seleccion_lasso_lineal.csv"),
                          show_col_types = FALSE)$variable
stopifnot(length(seleccion_xgb) > 0, all(seleccion_xgb %in% colnames(x)),
          nrow(x) == length(y), nrow(x_test) == nrow(test),
          identical(colnames(x), colnames(x_test)),
          length(folds_ajuste) == length(i_ajuste),
          setequal(unique(folds_ajuste), 1:5), !anyNA(y), all(y %in% 0:1))

# ------------------------------------------------------------------------------
# Configuracion: busqueda pequena y paralelizacion dentro de cada ajuste
# ------------------------------------------------------------------------------

set.seed(2026)
inicio_xgb <- Sys.time()
profundidades_xgb <- c(4L, 6L)
max_rondas_xgb <- 1000L
paciencia_xgb <- 50L
cortes_xgb <- seq(0.02, 0.98, by = 0.02)
cores_xgb <- parallel::detectCores(logical = FALSE)
if (is.na(cores_xgb)) cores_xgb <- 2L
hilos_xgb <- max(1L, cores_xgb - 1L)
parametros_xgb <- list(
  objective = "binary:logistic", eval_metric = "logloss", tree_method = "hist",
  eta = 0.05, min_child_weight = 10, subsample = 0.8, colsample_bytree = 0.8,
  lambda = 1, alpha = 0, gamma = 0, scale_pos_weight = 1,
  nthread = hilos_xgb, seed = 2026
)
# min_child_weight suma Hessianos: NO equivale a un numero minimo de personas.
# Sin pesos de clase. XGBoost aprende la direccion de los NA: no imputar medianas.
versiones_xgb <- list(todas = colnames(x), seleccion_lasso = seleccion_xgb)
salida_xgb <- file.path(dir_resultados, "xgboost")
dir.create(salida_xgb, recursive = TRUE, showWarnings = FALSE)
particiones_xgb <- data.frame(id = train$id, grupo = grupo, fold_cv = NA_integer_)
particiones_xgb$fold_cv[i_ajuste] <- folds_ajuste
write_csv(particiones_xgb, file.path(salida_xgb, "particiones.csv"))

# Una funcion: parada interna y reajuste sin tocar el fold de evaluacion
entrenar_xgb <- function(matriz, respuesta, parametros, max_rondas, paciencia) {
  set.seed(2026)
  parada <- integer()
  for (clase in 0:1) {
    filas <- which(respuesta == clase)
    if (length(filas) < 2) stop("No hay suficientes casos por clase para parada interna.")
    n <- min(length(filas) - 1L, max(1L, floor(0.1 * length(filas))))
    parada <- c(parada, filas[sample.int(length(filas), n)])
  }
  aprendizaje <- setdiff(seq_along(respuesta), parada)
  d_aprender <- xgb.DMatrix(matriz[aprendizaje, , drop = FALSE],
                            label = respuesta[aprendizaje], missing = NA)
  d_parada <- xgb.DMatrix(matriz[parada, , drop = FALSE],
                          label = respuesta[parada], missing = NA)
  argumentos <- list(params = parametros, data = d_aprender, nrounds = max_rondas,
                      early_stopping_rounds = paciencia, maximize = FALSE, verbose = 0)
  # xgboost reciente usa evals; versiones anteriores denominan watchlist.
  nombre_evals <- if ("evals" %in% names(formals(xgb.train))) "evals" else "watchlist"
  argumentos[[nombre_evals]] <- list(parada = d_parada)
  modelo_parada <- do.call(xgb.train, argumentos)
  historial <- modelo_parada$evaluation_log
  if (is.null(historial)) historial <- attr(modelo_parada, "evaluation_log")
  historial <- as.data.frame(historial)
  columna_loss <- grep("parada.*logloss", names(historial), value = TRUE)
  if (length(columna_loss) != 1) stop("No se encontro logloss de parada interna.")
  # Numero de rondas = fila del minimo. Evita diferencias de indices 0/1 entre APIs.
  rondas <- which.min(historial[[columna_loss]])
  # Liberar los datos y el bosque de parada antes de reservar la matriz completa.
  alcanzo_limite <- nrow(historial) == max_rondas
  rm(modelo_parada, argumentos, d_aprender, d_parada)
  gc(verbose = FALSE)
  d_completo <- xgb.DMatrix(matriz, label = respuesta, missing = NA)
  modelo <- xgb.train(params = parametros, data = d_completo,
                      nrounds = rondas, verbose = 0)
  list(modelo = modelo, rondas = rondas,
        alcanzo_limite = alcanzo_limite)
}

# ------------------------------------------------------------------------------
# CV: dos versiones por dos profundidades; mismo conjunto de cinco folds
# ------------------------------------------------------------------------------

tabla_xgb <- list()
tiempos_xgb <- list()
for (version in names(versiones_xgb)) {
  vars <- versiones_xgb[[version]]
  for (profundidad in profundidades_xgb) {
    nombre <- paste0(version, "_prof", profundidad)
    parametros <- parametros_xgb
    parametros$max_depth <- profundidad
    prob_oof <- rep(NA_real_, length(i_ajuste))
    rondas_folds <- integer(5)
    limites_folds <- logical(5)
    for (k in 1:5) {
      inicio_fold <- Sys.time()
      dentro <- folds_ajuste != k
      filas_train <- i_ajuste[dentro]
      filas_eval <- i_ajuste[!dentro]
      ajuste_xgb <- entrenar_xgb(x[filas_train, vars, drop = FALSE], y[filas_train],
                                  parametros, max_rondas_xgb, paciencia_xgb)
      d_eval <- xgb.DMatrix(x[filas_eval, vars, drop = FALSE], missing = NA)
      prob_oof[!dentro] <- predict(ajuste_xgb$modelo, d_eval)
      rondas_folds[k] <- ajuste_xgb$rondas
      limites_folds[k] <- ajuste_xgb$alcanzo_limite
      minutos <- as.numeric(difftime(Sys.time(), inicio_fold, units = "mins"))
      tiempos_xgb[[paste(nombre, k)]] <- data.frame(
        configuracion = nombre, etapa = paste0("fold_", k),
        rondas = ajuste_xgb$rondas, minutos = minutos)
      message(nombre, ": fold ", k, "/5; rondas=", ajuste_xgb$rondas,
               "; minutos=", round(minutos, 1))
      rm(ajuste_xgb, d_eval)
      gc(verbose = FALSE)
    }

    # El numero de rondas se decide por logloss interna; configuracion y corte,
    # por F1 OOF. Ninguna de estas decisiones utiliza la validacion reservada.
    f1_cortes <- numeric(length(cortes_xgb))
    verdad <- y[i_ajuste]
    for (j in seq_along(cortes_xgb)) {
      pred <- as.integer(prob_oof >= cortes_xgb[j])
      tp <- sum(pred == 1 & verdad == 1)
      fp <- sum(pred == 1 & verdad == 0)
      fn <- sum(pred == 0 & verdad == 1)
      f1_cortes[j] <- 2 * tp / max(2 * tp + fp + fn, 1)
    }
    mejor <- which.max(f1_cortes)
    tabla_xgb[[nombre]] <- data.frame(
      configuracion = nombre, version = version, columnas = length(vars),
      max_depth = profundidad, eta = parametros$eta,
      rondas = as.integer(round(median(rondas_folds))),
      corte = cortes_xgb[mejor], f1_cv = f1_cortes[mejor],
      folds_al_limite = sum(limites_folds)
    )
    write_csv(data.frame(id = train$id[i_ajuste], fold = folds_ajuste,
                         pobre = verdad, probabilidad = prob_oof),
              file.path(salida_xgb, paste0("oof_", nombre, ".csv")))
    write_csv(data.frame(corte = cortes_xgb, f1_cv = f1_cortes),
              file.path(salida_xgb, paste0("cortes_", nombre, ".csv")))
    write_csv(bind_rows(tabla_xgb), file.path(salida_xgb, "busqueda_cv.csv"))
    write_csv(bind_rows(tiempos_xgb), file.path(salida_xgb, "tiempos.csv"))
  }
}
tabla_xgb <- bind_rows(tabla_xgb) |>
  arrange(desc(f1_cv), max_depth)

# ------------------------------------------------------------------------------
# Evaluar la mejor configuracion CV de cada version en validacion interna
# ------------------------------------------------------------------------------

comparacion_xgb <- list()
for (version in names(versiones_xgb)) {
  elegido <- filter(tabla_xgb, .data$version == .env$version)[1, ]
  vars <- versiones_xgb[[version]]
  parametros <- parametros_xgb
  parametros$max_depth <- elegido$max_depth
  d_ajuste <- xgb.DMatrix(x[i_ajuste, vars, drop = FALSE], label = y[i_ajuste], missing = NA)
  modelo <- xgb.train(params = parametros, data = d_ajuste,
                      nrounds = elegido$rondas, verbose = 0)
  d_validacion <- xgb.DMatrix(x[i_validacion, vars, drop = FALSE], missing = NA)
  prob <- predict(modelo, d_validacion)
  pred <- as.integer(prob >= elegido$corte)
  verdad <- y[i_validacion]
  tp <- sum(pred == 1 & verdad == 1)
  fp <- sum(pred == 1 & verdad == 0)
  fn <- sum(pred == 0 & verdad == 1)
  tn <- sum(pred == 0 & verdad == 0)
  comparacion_xgb[[version]] <- mutate(
    elegido, f1_validacion = 2 * tp / max(2 * tp + fp + fn, 1),
    precision = if (tp + fp == 0) NA_real_ else tp / (tp + fp),
    sensibilidad = tp / (tp + fn), especificidad = tn / (tn + fp),
    accuracy = mean(pred == verdad), verdaderos_positivos = tp,
    falsos_positivos = fp, falsos_negativos = fn, verdaderos_negativos = tn)
  xgb.save(modelo, file.path(salida_xgb, paste0("modelo_validacion_", version, ".ubj")))
  write_csv(data.frame(id = train$id[i_validacion], pobre = verdad,
                       probabilidad = prob, clasificacion = pred),
            file.path(salida_xgb, paste0("validacion_", version, ".csv")))
}
comparacion_xgb <- bind_rows(comparacion_xgb) |>
  arrange(desc(f1_validacion), desc(f1_cv))
write_csv(comparacion_xgb, file.path(salida_xgb, "comparacion_xgb.csv"))
print(comparacion_xgb)
# Validacion compartida con experimentos anteriores: conjunto de desarrollo,
# no una estimacion final independiente despues de comparar multiples modelos.

# ------------------------------------------------------------------------------
# Reestimar el ganador con todo train y aplicar al test oficial
# ------------------------------------------------------------------------------

elegido_xgb <- comparacion_xgb[1, ]
vars <- versiones_xgb[[elegido_xgb$version]]
parametros <- parametros_xgb
parametros$max_depth <- elegido_xgb$max_depth
d_final <- xgb.DMatrix(x[, vars, drop = FALSE], label = y, missing = NA)
modelo_final_xgb <- xgb.train(params = parametros, data = d_final,
                              nrounds = elegido_xgb$rondas, verbose = 0)
d_test <- xgb.DMatrix(x_test[, vars, drop = FALSE], missing = NA)
prob_test_xgb <- predict(modelo_final_xgb, d_test)
predicciones_xgb <- data.frame(id = test$id, probabilidad = prob_test_xgb,
                               pobre = as.integer(prob_test_xgb >= elegido_xgb$corte))
write_csv(predicciones_xgb, file.path(salida_xgb, "predicciones_test.csv"))
write_csv(predicciones_xgb[c("id", "pobre")],
          file.path(salida_xgb, "clasificacion_test.csv"))
# Gain mide mejora de la funcion objetivo en divisiones, NO importancia por
# permutacion ni caida de F1; no equiparar sus magnitudes a las de random forest.
importancia_xgb <- xgb.importance(model = modelo_final_xgb)
write_csv(as.data.frame(importancia_xgb), file.path(salida_xgb, "importancia_gain.csv"))
xgb.save(modelo_final_xgb, file.path(salida_xgb, "ganador_xgb.ubj"))
saveRDS(list(configuracion = elegido_xgb, parametros = parametros, columnas = vars,
             niveles = niveles, predictores = predictores, semilla = 2026,
             comparacion = comparacion_xgb, version_paquete = as.character(packageVersion("xgboost"))),
        file.path(salida_xgb, "configuracion_ganador.rds"))
message("Ganador: ", elegido_xgb$configuracion, "; minutos totales: ",
         round(as.numeric(difftime(Sys.time(), inicio_xgb, units = "mins")), 1))


exportar_adicional(30)


# ------------------------------------------------------------------------------
# (6) Modelo 31: LightGBM sin ingreso predicho
# ------------------------------------------------------------------------------


set.seed(2026)
inicio_lgb <- Sys.time()
hojas_lgb <- c(31L, 63L)
max_rondas_lgb <- 1000L
paciencia_lgb <- 50L
cortes_lgb <- seq(0.02, 0.98, by = 0.02)
nucleos_lgb <- parallel::detectCores(logical = FALSE)
if (is.na(nucleos_lgb)) nucleos_lgb <- 2L
hilos_lgb <- max(1L, nucleos_lgb - 1L)
parametros_lgb <- list(
  objective = "binary", metric = "binary_logloss", boosting = "gbdt",
  learning_rate = 0.05, max_depth = -1L, min_data_in_leaf = 50L,
  feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1L,
  lambda_l1 = 0, lambda_l2 = 1, max_bin = 255L,
  scale_pos_weight = 1, num_threads = hilos_lgb,
  seed = 2026L, feature_fraction_seed = 2026L, bagging_seed = 2026L,
  data_random_seed = 2026L, deterministic = TRUE, force_col_wise = TRUE,
  verbosity = -1L
)
# Paralelizar dentro de LightGBM, no los folds. No ponderar clases inicialmente.
# Los NA se manejan de forma nativa; las dummies y columnas son las mismas del 04.
salida_lgb <- file.path(dir_resultados, "lightgbm")
dir.create(salida_lgb, recursive = TRUE, showWarnings = FALSE)
particiones_lgb <- data.frame(id = train$id, grupo = grupo, fold_cv = NA_integer_)
particiones_lgb$fold_cv[i_ajuste] <- folds_ajuste
write_csv(particiones_lgb, file.path(salida_lgb, "particiones.csv"))

# Parada temprana interna: el fold OOF no decide el numero de rondas
entrenar_lgb <- function(matriz, respuesta, parametros, max_rondas, paciencia) {
  set.seed(2026)
  parada <- integer()
  for (clase in 0:1) {
    filas <- which(respuesta == clase)
    if (length(filas) < 2) stop("Casos insuficientes para parada interna.")
    n <- min(length(filas) - 1L, max(1L, floor(0.1 * length(filas))))
    parada <- c(parada, filas[sample.int(length(filas), n)])
  }
  aprendizaje <- setdiff(seq_along(respuesta), parada)
  d_aprender <- lgb.Dataset(matriz[aprendizaje, , drop = FALSE],
                            label = respuesta[aprendizaje])
  d_parada <- lgb.Dataset(matriz[parada, , drop = FALSE], label = respuesta[parada],
                          reference = d_aprender)
  modelo_parada <- lgb.train(
    params = parametros, data = d_aprender, nrounds = max_rondas,
    valids = list(parada = d_parada), early_stopping_rounds = paciencia,
    verbose = -1, record = TRUE
  )
  historial <- modelo_parada$record_evals$parada$binary_logloss$eval
  if (!length(historial)) stop("No se encontro el historial de logloss interna.")
  rondas <- which.min(unlist(historial))
  d_completo <- lgb.Dataset(matriz, label = respuesta)
  modelo <- lgb.train(params = parametros, data = d_completo,
                      nrounds = rondas, verbose = -1)
  list(modelo = modelo, rondas = rondas, alcanzo_limite = length(historial) == max_rondas)
}

# ------------------------------------------------------------------------------
# Cinco folds por configuracion: elegir hojas y corte por F1
# ------------------------------------------------------------------------------

tabla_lgb <- list()
tiempos_lgb <- list()
for (hojas in hojas_lgb) {
  nombre <- paste0("hojas_", hojas)
  parametros <- parametros_lgb
  parametros$num_leaves <- hojas
  prob_oof <- rep(NA_real_, length(i_ajuste))
  rondas_folds <- integer(5)
  limites_folds <- logical(5)
  for (k in 1:5) {
    inicio_fold <- Sys.time()
    dentro <- folds_ajuste != k
    filas_train <- i_ajuste[dentro]
    filas_eval <- i_ajuste[!dentro]
    ajuste_lgb <- entrenar_lgb(x[filas_train, , drop = FALSE], y[filas_train],
                                parametros, max_rondas_lgb, paciencia_lgb)
    prob_oof[!dentro] <- predict(ajuste_lgb$modelo, x[filas_eval, , drop = FALSE])
    rondas_folds[k] <- ajuste_lgb$rondas
    limites_folds[k] <- ajuste_lgb$alcanzo_limite
    minutos <- as.numeric(difftime(Sys.time(), inicio_fold, units = "mins"))
    tiempos_lgb[[paste(nombre, k)]] <- data.frame(
      configuracion = nombre, etapa = paste0("fold_", k), rondas = ajuste_lgb$rondas,
      minutos = minutos)
    message(nombre, ": fold ", k, "/5; rondas=", ajuste_lgb$rondas,
             "; minutos=", round(minutos, 1))
    rm(ajuste_lgb)
    gc(verbose = FALSE)
  }
  verdad <- y[i_ajuste]
  f1_cortes <- numeric(length(cortes_lgb))
  for (j in seq_along(cortes_lgb)) {
    pred <- as.integer(prob_oof >= cortes_lgb[j])
    tp <- sum(pred == 1 & verdad == 1)
    fp <- sum(pred == 1 & verdad == 0)
    fn <- sum(pred == 0 & verdad == 1)
    f1_cortes[j] <- 2 * tp / max(2 * tp + fp + fn, 1)
  }
  mejor <- which.max(f1_cortes)
  tabla_lgb[[nombre]] <- data.frame(
    configuracion = nombre, columnas = ncol(x), num_leaves = hojas,
    learning_rate = parametros$learning_rate,
    rondas = as.integer(round(median(rondas_folds))), corte = cortes_lgb[mejor],
    f1_cv = f1_cortes[mejor], folds_al_limite = sum(limites_folds))
  write_csv(data.frame(id = train$id[i_ajuste], fold = folds_ajuste,
                       pobre = verdad, probabilidad = prob_oof),
            file.path(salida_lgb, paste0("oof_", nombre, ".csv")))
  write_csv(data.frame(corte = cortes_lgb, f1_cv = f1_cortes),
            file.path(salida_lgb, paste0("cortes_", nombre, ".csv")))
  write_csv(bind_rows(tabla_lgb), file.path(salida_lgb, "busqueda_cv.csv"))
  write_csv(bind_rows(tiempos_lgb), file.path(salida_lgb, "tiempos.csv"))
}
tabla_lgb <- bind_rows(tabla_lgb) |>
  arrange(desc(f1_cv), num_leaves)

# ------------------------------------------------------------------------------
# Validacion del ganador CV: no volver a elegir el corte aqui
# ------------------------------------------------------------------------------

elegido_lgb <- tabla_lgb[1, ]
parametros <- parametros_lgb
parametros$num_leaves <- elegido_lgb$num_leaves
d_ajuste <- lgb.Dataset(x[i_ajuste, , drop = FALSE], label = y[i_ajuste])
modelo_validacion_lgb <- lgb.train(params = parametros, data = d_ajuste,
                                   nrounds = elegido_lgb$rondas, verbose = -1)
prob <- predict(modelo_validacion_lgb, x[i_validacion, , drop = FALSE])
pred <- as.integer(prob >= elegido_lgb$corte)
verdad <- y[i_validacion]
tp <- sum(pred == 1 & verdad == 1)
fp <- sum(pred == 1 & verdad == 0)
fn <- sum(pred == 0 & verdad == 1)
tn <- sum(pred == 0 & verdad == 0)
comparacion_lgb <- mutate(
  elegido_lgb, f1_validacion = 2 * tp / max(2 * tp + fp + fn, 1),
  precision = if (tp + fp == 0) NA_real_ else tp / (tp + fp),
  sensibilidad = tp / (tp + fn), especificidad = tn / (tn + fp),
  accuracy = mean(pred == verdad), verdaderos_positivos = tp,
  falsos_positivos = fp, falsos_negativos = fn, verdaderos_negativos = tn)
print(comparacion_lgb)
write_csv(comparacion_lgb, file.path(salida_lgb, "comparacion_lgb.csv"))
write_csv(data.frame(id = train$id[i_validacion], pobre = verdad,
                     probabilidad = prob, clasificacion = pred),
          file.path(salida_lgb, "predicciones_validacion.csv"))
lgb.save(modelo_validacion_lgb, file.path(salida_lgb, "modelo_validacion.txt"))
# Esta validacion se comparte con experimentos previos: es desarrollo,
# no una estimacion final independiente despues de elegir entre muchos modelos.

# ------------------------------------------------------------------------------
# Reestimar con todo train; conservar configuracion y corte de CV
# ------------------------------------------------------------------------------

d_final <- lgb.Dataset(x, label = y)
modelo_final_lgb <- lgb.train(params = parametros, data = d_final,
                              nrounds = elegido_lgb$rondas, verbose = -1)
prob_test_lgb <- predict(modelo_final_lgb, x_test)
predicciones_lgb <- data.frame(id = test$id, probabilidad = prob_test_lgb,
                               pobre = as.integer(prob_test_lgb >= elegido_lgb$corte))
write_csv(predicciones_lgb, file.path(salida_lgb, "predicciones_test.csv"))
write_csv(predicciones_lgb[c("id", "pobre")],
          file.path(salida_lgb, "clasificacion_test.csv"))
importancia_lgb <- lgb.importance(modelo_final_lgb, percentage = TRUE)
write_csv(as.data.frame(importancia_lgb), file.path(salida_lgb, "importancia_gain.csv"))
# Ganancia de divisiones, NO importancia por permutacion ni caida de F1.
lgb.save(modelo_final_lgb, file.path(salida_lgb, "ganador_lgb.txt"))
saveRDS(list(configuracion = elegido_lgb, parametros = parametros,
             columnas = colnames(x), niveles = niveles, predictores = predictores,
             semilla = 2026, comparacion = comparacion_lgb,
             version_paquete = as.character(packageVersion("lightgbm"))),
        file.path(salida_lgb, "configuracion_ganador.rds"))
message("Minutos totales: ", round(as.numeric(difftime(Sys.time(), inicio_lgb,
                                                     units = "mins")), 1))


exportar_adicional(31)


# ------------------------------------------------------------------------------
# (7) Modelo 32: elastic net logit con ingreso RF dentro de cada fold
# ------------------------------------------------------------------------------


consolidado_preparar <- function() {
  tr <- train_completo; te <- test_completo
  y <- tr$pobre
  predictores <- setdiff(names(tr), c("id", "pobre"))
  datos <- as.data.frame(tr[predictores]); datos_test <- as.data.frame(te[predictores])
  categoricas <- union(intersect(c("clase", "dominio", "depto", "tenencia_vivienda",
    "nivel_educativo_max", "nivel_educativo_max_viejo", "oficio_jefe", "oficio_pareja"), predictores),
    names(datos)[vapply(datos, function(v) is.character(v) || is.factor(v), logical(1))])
  niveles <- list()
  for (v in categoricas) {
    niveles[[v]] <- c(sort(unique(na.omit(as.character(datos[[v]])))), "Desconocido")
    a <- as.character(datos[[v]]); b <- as.character(datos_test[[v]])
    a[is.na(a)] <- "Desconocido"; b[is.na(b) | !b %in% niveles[[v]]] <- "Desconocido"
    datos[[v]] <- factor(a, levels = niveles[[v]])
    datos_test[[v]] <- factor(b, levels = niveles[[v]])
  }
  for (v in setdiff(predictores, categoricas)) {
    datos[[v]][!is.finite(datos[[v]])] <- NA_real_
    datos_test[[v]][!is.finite(datos_test[[v]])] <- NA_real_
    if (!v %in% c("num_ing_ug_hat", "num_ing_pc_hat", "num_ratio_lp_hat")) {
      datos[[paste0(v, "_faltante")]] <- as.integer(is.na(datos[[v]]))
      datos_test[[paste0(v, "_faltante")]] <- as.integer(is.na(datos_test[[v]]))
    }
  }
  x <- model.matrix(~ . - 1, model.frame(~ . - 1, datos, na.action = na.pass))
  xt <- model.matrix(~ . - 1, model.frame(~ . - 1, datos_test, na.action = na.pass))
  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores)) cores <- 2L
  list(x = x, xt = xt[, colnames(x), drop = FALSE], y = y, id = tr$id,
       id_test = te$id, ia = i_ajuste, iv = i_validacion, folds = folds_ajuste,
       cache = file.path(dir_resultados, "matrices_rf"), hilos = max(1L, cores - 1L),
       niveles = niveles, predictores = predictores)
}

consolidado_ingresos_hogar <- function(predicciones, personas, hogares) {
  predicciones |>
    left_join(personas |> rename_with(tolower) |> select(id, id_persona = orden, p6050),
              by = c("id", "id_persona")) |>
    filter(p6050 %in% c(1:5, 9)) |>
    group_by(id) |>
    summarise(num_ing_ug_hat = sum(ingreso_predicho), .groups = "drop") |>
    right_join(hogares |> rename_with(tolower) |> select(id, npersug, p5090, p5130, lp), by = "id") |>
    mutate(num_ing_ug_hat = replace_na(num_ing_ug_hat, 0),
      arriendo = if_else(p5090 %in% c(1, 4) & !(p5130 %in% c(98, 99)), replace_na(p5130, 0), 0),
      num_ing_pc_hat = (num_ing_ug_hat + arriendo) / pmax(npersug, 1),
      num_ratio_lp_hat = num_ing_pc_hat / lp) |>
    select(id, num_ing_ug_hat, num_ing_pc_hat, num_ratio_lp_hat)
}

consolidado_datos <- function(exp, entrenamiento, evaluacion, nombre, test = FALSE) {
  # Solo reutilizar matrices construidas durante ESTA ejecucion (32 y 33).
  # No basta que un archivo de otra corrida tenga el mismo nombre.
  if (exists(nombre, envir = matrices_generadas, inherits = FALSE))
    return(readRDS(get(nombre, envir = matrices_generadas)))
  dir.create(exp$cache, recursive = TRUE, showWarnings = FALSE)
  ids <- exp$id[entrenamiento]
  personas <- filter(raw_train_personas, id %in% ids)
  set.seed(2026)
  internos <- createFolds(factor(exp$y[entrenamiento], levels = 0:1), k = 5, returnTrain = FALSE)
  fh <- data.frame(id = ids, fold = NA_integer_)
  for (k in 1:5) fh$fold[internos[[k]]] <- k
  ruta_modelo <- file.path(exp$cache, paste0(nombre, "_ingreso.rds"))
  # La PET oficial define quienes reciben prediccion; fuera de PET, cero.
  # Personas del fold externo nunca entrenan el ingreso ni la imputacion.
  ingresos_a <- generar_ingreso_personas(personas, TRUE, ruta_modelo,
    referencia_test = raw_test_personas, folds_hogar = fh)
  pe <- if (test) raw_test_personas else filter(raw_train_personas, id %in% exp$id[evaluacion])
  ingresos_e <- generar_ingreso_personas(pe, FALSE, ruta_modelo)
  ha <- filter(raw_train_hogares, id %in% ids)
  he <- if (test) raw_test_hogares else filter(raw_train_hogares, id %in% exp$id[evaluacion])
  ag <- consolidado_ingresos_hogar(ingresos_a, personas, ha)
  eg <- consolidado_ingresos_hogar(ingresos_e, pe, he)
  xa <- exp$x[entrenamiento, , drop = FALSE]
  xe <- if (test) exp$xt else exp$x[evaluacion, , drop = FALSE]
  ids_e <- if (test) exp$id_test else exp$id[evaluacion]
  for (v in c("num_ing_ug_hat", "num_ing_pc_hat", "num_ratio_lp_hat")) {
    xa[, v] <- ag[[v]][match(ids, ag$id)]
    xe[, v] <- eg[[v]][match(ids_e, eg$id)]
  }
  preparado <- imputar_matriz(xa)
  d <- list(x = preparado$x, y = exp$y[entrenamiento],
            xe = imputar_matriz(xe, preparado$medianas)$x,
            ye = if (test) NULL else exp$y[evaluacion], medianas = preparado$medianas,
            filas_train = entrenamiento, filas_eval = evaluacion, es_test = test,
            modelo_ingreso = ruta_modelo)
  ruta <- file.path(exp$cache, paste0(nombre, ".rds"))
  saveRDS(d, ruta); assign(nombre, ruta, envir = matrices_generadas)
  d
}

matrices_generadas <- new.env(parent = emptyenv())
experimento_compartido <- consolidado_preparar()

consolidado_corte <- function(prob, y) {
  cortes <- seq(0.02, 0.98, by = 0.02)
  intervalo <- findInterval(prob, cortes)
  tp <- rev(cumsum(rev(tabulate(intervalo[y == 1] + 1, nbins = length(cortes) + 1))))[-1]
  fp <- rev(cumsum(rev(tabulate(intervalo[y == 0] + 1, nbins = length(cortes) + 1))))[-1]
  fn <- sum(y == 1) - tp
  f1 <- 2 * tp / pmax(2 * tp + fp + fn, 1)
  j <- which.max(f1)
  data.frame(corte = cortes[j], f1_cv = f1[j])
}

consolidado_evaluar <- function(prob, y, corte) {
  pred <- as.integer(prob >= corte)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1); tn <- sum(pred == 0 & y == 0)
  data.frame(f1_validacion = 2 * tp / max(2 * tp + fp + fn, 1),
             precision = if (tp + fp == 0) NA_real_ else tp / (tp + fp),
             sensibilidad = tp / (tp + fn), especificidad = tn / (tn + fp),
             verdaderos_positivos = tp, falsos_positivos = fp,
             falsos_negativos = fn, verdaderos_negativos = tn)
}


set.seed(2026)
experimento <- experimento_compartido
salida <- file.path(dir_resultados, "logit_en_consolidado")
dir.create(salida, recursive = TRUE, showWarnings = FALSE)
alphas <- c(0.125,0.25,0.375, 0.50,0.625, 0.75,0.875)
lambdas <- 10^seq(0, -4, length.out = 25)
# glmnet estandariza dentro de cada ajuste. Sin ponderacion de clases.

# ------------------------------------------------------------------------------
# CV: comparar alpha/lambda por F1 OOF y seleccionar tambien el corte
# ------------------------------------------------------------------------------

oof <- array(NA_real_, dim = c(length(experimento$ia), length(lambdas), length(alphas)))
for (k in 1:5) {
  dentro <- experimento$folds != k
  d <- consolidado_datos(experimento, experimento$ia[dentro], experimento$ia[!dentro], paste0("fold_", k))
  for (a in seq_along(alphas)) {
    modelo <- glmnet(d$x, d$y, family = "binomial", alpha = alphas[a],
                      lambda = lambdas, standardize = TRUE)
    oof[!dentro, , a] <- predict(modelo, d$xe, s = lambdas, type = "response")
    message("Logit EN: fold ", k, "/5; alpha=", alphas[a])
  }
}
cv <- list()
for (a in seq_along(alphas)) {
  for (j in seq_along(lambdas)) {
    cv[[paste(a, j)]] <- cbind(data.frame(alpha = alphas[a], lambda = lambdas[j],
                                          indice_alpha = a, indice_lambda = j),
                               consolidado_corte(oof[, j, a], experimento$y[experimento$ia]))
  }
}
cv <- bind_rows(cv) |>
  arrange(desc(f1_cv), desc(lambda), alpha)
elegido <- cv[1, ]
write_csv(cv, file.path(salida, "busqueda_cv.csv"))
write_csv(data.frame(id = experimento$id[experimento$ia], pobre = experimento$y[experimento$ia],
                     probabilidad = oof[, elegido$indice_lambda, elegido$indice_alpha]),
          file.path(salida, "oof_ganador.csv"))

# ------------------------------------------------------------------------------
# Validacion con corte congelado y ajuste final con todo train
# ------------------------------------------------------------------------------

d <- consolidado_datos(experimento, experimento$ia, experimento$iv, "validacion")
modelo <- glmnet(d$x, d$y, family = "binomial", alpha = elegido$alpha,
                  lambda = lambdas, standardize = TRUE)
prob <- as.numeric(predict(modelo, d$xe, s = elegido$lambda, type = "response"))
resultado <- cbind(elegido, consolidado_evaluar(prob, d$ye, elegido$corte))
print(resultado)
write_csv(resultado, file.path(salida, "validacion.csv"))
d <- consolidado_datos(experimento, seq_along(experimento$y), integer(), "final", test = TRUE)
modelo <- glmnet(d$x, d$y, family = "binomial", alpha = elegido$alpha,
                  lambda = lambdas, standardize = TRUE)
prob <- as.numeric(predict(modelo, d$xe, s = elegido$lambda, type = "response"))
write_csv(data.frame(id = experimento$id_test, probabilidad = prob,
                     pobre = as.integer(prob >= elegido$corte)), file.path(salida, "predicciones_test.csv"))
nombre <- paste0("EN_logit_alpha_", elegido$alpha, "_lambda_", signif(elegido$lambda, 4),
                  "_prob_", format(elegido$corte, nsmall = 2), ".csv")
write_csv(data.frame(id = experimento$id_test, pobre = as.integer(prob >= elegido$corte)), file.path(salida, nombre))
saveRDS(list(modelo = modelo, configuracion = elegido, columnas = colnames(d$x),
             medianas = d$medianas, niveles = experimento$niveles,
             predictores = experimento$predictores, modelo_ingreso = d$modelo_ingreso,
             validacion = resultado), file.path(salida, "ganador.rds"))


exportar_adicional(32)


# ------------------------------------------------------------------------------
# (8) Modelo 33: XGBoost con las mismas matrices de ingreso RF
# ------------------------------------------------------------------------------


set.seed(2026)
experimento <- experimento_compartido
salida <- file.path(dir_resultados, "xgb_consolidado")
dir.create(salida, recursive = TRUE, showWarnings = FALSE)
grilla <- data.frame(max_depth = c(3L, 4L, 6L))
parametros <- list(objective = "binary:logistic", eval_metric = "logloss",
                   tree_method = "hist", eta = 0.05, min_child_weight = 10,
                   subsample = 0.8, colsample_bytree = 0.8,
                   lambda = 1, alpha = 0, scale_pos_weight = 1,
                   nthread = experimento$hilos, seed = 2026)
oof <- matrix(NA_real_, length(experimento$ia), nrow(grilla))
rondas <- matrix(0L, 5, nrow(grilla))
for (k in 1:5) {
  dentro <- experimento$folds != k
  if (exists("d")) rm(d)
gc(verbose = FALSE)
d <- consolidado_datos(experimento, experimento$ia[dentro], experimento$ia[!dentro], paste0("fold_", k))
  for (j in seq_len(nrow(grilla))) {
    parametros$max_depth <- grilla$max_depth[j]
    archivo_ajuste <- file.path(salida, paste0("fold_", k, "_depth_", parametros$max_depth, ".rds"))
    ajuste <- entrenar_xgb(d$x, d$y, parametros, 1000L, 50L)
    avance <- list(prob = predict(ajuste$modelo, xgb.DMatrix(d$xe)), rondas = ajuste$rondas)
    rm(ajuste); gc(verbose = FALSE)
    oof[!dentro, j] <- avance$prob
    rondas[k, j] <- avance$rondas
    message("XGBoost: fold ", k, "/5, profundidad ", parametros$max_depth)
  }
}
cv <- bind_rows(lapply(seq_len(nrow(grilla)), function(j) {
  cbind(grilla[j, , drop = FALSE], rondas = as.integer(round(median(rondas[, j]))),
         consolidado_corte(oof[, j], experimento$y[experimento$ia]))
}))
cv$indice <- seq_len(nrow(cv))
cv <- arrange(cv, desc(f1_cv), max_depth)
elegido <- cv[1, ]
write_csv(cv, file.path(salida, "busqueda_cv.csv"))
write_csv(data.frame(id = experimento$id[experimento$ia], pobre = experimento$y[experimento$ia],
                     probabilidad = oof[, elegido$indice]), file.path(salida, "oof_ganador.csv"))

# ------------------------------------------------------------------------------
# Validacion congelada y modelo final con todas las observaciones de train
# ------------------------------------------------------------------------------

if (exists("d")) rm(d)
gc(verbose = FALSE)
d <- consolidado_datos(experimento, experimento$ia, experimento$iv, "validacion")
parametros$max_depth <- elegido$max_depth
modelo <- xgb.train(params = parametros, data = xgb.DMatrix(d$x, label = d$y),
                    nrounds = elegido$rondas, verbose = 0)
prob <- predict(modelo, xgb.DMatrix(d$xe))
resultado <- cbind(elegido, consolidado_evaluar(prob, d$ye, elegido$corte))
print(resultado)
write_csv(resultado, file.path(salida, "validacion.csv"))
# Validacion usada para comparar modelos: no volver a ajustar el corte con ella.
if (exists("d")) rm(d)
gc(verbose = FALSE)
rm(modelo)
gc(verbose = FALSE)
d <- consolidado_datos(experimento, seq_along(experimento$y), integer(), "final", test = TRUE)
modelo <- xgb.train(params = parametros, data = xgb.DMatrix(d$x, label = d$y),
                    nrounds = elegido$rondas, verbose = 0)
prob <- predict(modelo, xgb.DMatrix(d$xe))
write_csv(data.frame(id = experimento$id_test, probabilidad = prob,
                     pobre = as.integer(prob >= elegido$corte)), file.path(salida, "predicciones_test.csv"))
nombre <- paste0("XGBoost_depth_", elegido$max_depth, "_eta_005_rondas_", elegido$rondas,
                  "_prob_", format(elegido$corte, nsmall = 2), ".csv")
write_csv(data.frame(id = experimento$id_test, pobre = as.integer(prob >= elegido$corte)), file.path(salida, nombre))
xgb.save(modelo, file.path(salida, "ganador.ubj"))
saveRDS(list(parametros = parametros, configuracion = elegido, columnas = colnames(d$x),
             medianas = d$medianas, niveles = experimento$niveles,
             predictores = experimento$predictores, modelo_ingreso = d$modelo_ingreso,
             validacion = resultado), file.path(salida, "configuracion.rds"))

exportar_adicional(33)


})
