import pandas as pd
import numpy as np
from datetime import datetime

def validar_archivo_excel(ruta_archivo, tipo_tabla):
    """
    Valida un archivo Excel según su tipo de tabla
    Retorna: (es_valido, errores, num_filas)
    """
    try:
        # Leer el archivo Excel
        excel_file = pd.ExcelFile(ruta_archivo)
        hojas_disponibles = excel_file.sheet_names
        
        print(f"DEBUG: Hojas disponibles: {hojas_disponibles}")
        
        # Usar la primera hoja por defecto
        hoja_a_usar = hojas_disponibles[0] if hojas_disponibles else 0
        print(f"DEBUG: Probando hoja '{hoja_a_usar}'")
        
        # Leer la hoja
        df = pd.read_excel(ruta_archivo, sheet_name=hoja_a_usar)
        
        # Limpiar nombres de columnas (eliminar espacios extras y convertir a minúsculas)
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"DEBUG: Columnas encontradas (en minúsculas): {list(df.columns)}")
        
        # Definir columnas requeridas por tipo de tabla (en minúsculas)
        columnas_requeridas = {
            'Emisiones': ['fuente_id', 'tipo_fuente', 'gas', 'tasa_emision', 'ubicacion', 'eficiencia_control', 'fecha_registro'],
            'Dispersión': ['gas', 'velocidad_viento', 'direccion_viento', 'temperatura', 'concentracion', 'fecha_registro'],
            'Exposición': ['zona', 'tamaño_poblacion', 'tiempo_exposicion', 'nivel_exposicion', 'fecha_registro'],
            'Salud': ['zona', 'casos_asma', 'casos_bronquitis', 'hospitalizaciones', 'mortalidad', 'fecha_registro'],
            'Social': ['tipo_medida', 'institucion', 'fecha_implementacion', 'impacto_estimado', 'observaciones']
        }
        
        # Columnas alternativas (con y sin acentos)
        columnas_alternativas = {
            'tasa_emision': ['tasa_emisión'],
            'tamaño_poblacion': ['tamaño_población'],
            'tiempo_exposicion': ['tiempo_exposición'],
            'nivel_exposicion': ['nivel_exposición'],
            'direccion_viento': ['dirección_viento'],
            'concentracion': ['concentración'],
            'institucion': ['institución'],
            'impacto_estimado': ['impacto_estimado'],
            'ubicacion': ['ubicación'],
            'eficiencia_control': ['eficiencia_control']
        }
        
        # Verificar columnas requeridas
        requeridas = columnas_requeridas.get(tipo_tabla, [])
        columnas_faltantes = []
        
        for columna in requeridas:
            if columna not in df.columns:
                # Buscar alternativas
                alternativas = columnas_alternativas.get(columna, [])
                encontrada = False
                for alt in alternativas:
                    if alt in df.columns:
                        encontrada = True
                        break
                if not encontrada:
                    columnas_faltantes.append(columna)
        
        if columnas_faltantes:
            errores = f"Faltan columnas {columnas_faltantes}. Encontradas: {list(df.columns)}"
            print(f"DEBUG: Validación fallida - {errores}")
            return False, errores, len(df)
        
        # Verificar que haya datos
        if len(df) == 0:
            errores = "El archivo no contiene datos"
            print(f"DEBUG: Validación fallida - {errores}")
            return False, errores, 0
        
        # Verificar tipos de datos básicos
        for columna in df.columns:
            if df[columna].isna().all():
                print(f"DEBUG: Advertencia - Columna '{columna}' está vacía")
        
        print(f"DEBUG: Validación completada - Hoja: {hoja_a_usar}, Filas: {len(df)}")
        return True, f"Archivo válido con {len(df)} filas", len(df)
        
    except Exception as e:
        error_msg = f"Error al validar archivo: {str(e)}"
        print(f"DEBUG: Validación fallida - {error_msg}")
        return False, error_msg, 0