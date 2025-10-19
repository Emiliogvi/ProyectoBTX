import pandas as pd

# ====================================================
# VALIDACIÓN AUTOMÁTICA DE ARCHIVOS EXCEL
# ====================================================
# Cada tipo de tabla tiene columnas esperadas.
# Si un archivo subido no las cumple → se marca como inválido.

COLUMNAS_ESPERADAS = {
    "Emisiones": ["tipo_fuente", "gas", "tasa_emision", "ubicacion", "fecha_registro"],
    "Dispersión": ["gas", "velocidad_viento", "direccion_viento", "temperatura", "concentracion", "fecha_registro"],
    "Exposición": ["zona", "tamaño_poblacion", "tiempo_exposicion", "nivel_exposicion", "fecha_registro"],
    "Salud": ["zona", "casos_asma", "casos_bronquitis", "hospitalizaciones", "mortalidad", "fecha_registro"],
    "Social": ["tipo_medida", "institucion", "fecha_implementacion", "impacto_estimado", "observaciones"]
}


def validar_archivo_excel(ruta_archivo, tipo_tabla):
    """
    Valida que el archivo tenga las columnas esperadas.
    Devuelve (es_valido, errores, num_filas)
    """
    try:
        # Detectar tipo de archivo
        if ruta_archivo.endswith('.csv'):
            df = pd.read_csv(ruta_archivo)
        else:
            df = pd.read_excel(ruta_archivo)

        columnas_esperadas = COLUMNAS_ESPERADAS.get(tipo_tabla)
        if not columnas_esperadas:
            return False, f"Tipo de tabla desconocido: {tipo_tabla}", 0

        # Normalizar nombres (minúsculas sin espacios)
        cols_norm = [c.strip().lower().replace(" ", "_") for c in df.columns]
        esperadas_norm = [c.strip().lower().replace(" ", "_") for c in columnas_esperadas]

        faltantes = [c for c in esperadas_norm if c not in cols_norm]
        extras = [c for c in cols_norm if c not in esperadas_norm]

        if faltantes:
            errores = f"Columnas faltantes: {', '.join(faltantes)}"
            if extras:
                errores += f" | Columnas extra: {', '.join(extras)}"
            return False, errores, len(df)

        return True, None, len(df)

    except Exception as e:
        return False, f"Error leyendo archivo: {str(e)}", 0
