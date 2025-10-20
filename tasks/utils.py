import os
import pandas as pd

def validar_archivo_excel(file_path, tipo_tabla):
    """
    Valida un archivo Excel según su tipo de tabla
    Retorna: (es_valido, errores, num_filas)
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            return False, "El archivo no existe", 0
        
        # Columnas esperadas por tipo de tabla
        columnas_por_tipo = {
            'Emisiones': ['Fuente_ID', 'Tipo_Fuente', 'Gas', 'Tasa_Emisión', 'Ubicación', 'Fecha_Registro'],
            'Dispersión': ['Registro_ID', 'Gas', 'Velocidad_Viento', 'Dirección_Viento', 'Temperatura', 'Concentración', 'Fecha_Registro'],
            'Exposición': ['Población_ID', 'Zona', 'Tamaño_Población', 'Tiempo_Exposición', 'Nivel_Exposición', 'Fecha_Registro'],
            'Salud': ['Registro_Salud_ID', 'Zona', 'Casos_Asma', 'Casos_Bronquitis', 'Hospitalizaciones', 'Mortalidad', 'Fecha_Registro'],
            'Social': ['Política_ID', 'Tipo_Medida', 'Institución', 'Fecha_Implementación', 'Impacto_Estimado', 'Observaciones']
        }
        
        columnas_esperadas = columnas_por_tipo.get(tipo_tabla)
        if not columnas_esperadas:
            return False, f"Tipo de tabla no válido: {tipo_tabla}", 0
        
        # Intentar leer el archivo Excel
        try:
            # Leer todas las hojas para encontrar la correcta
            excel_file = pd.ExcelFile(file_path)
            hojas_disponibles = excel_file.sheet_names
            
            print(f"DEBUG: Hojas disponibles: {hojas_disponibles}")
            
            # Buscar una hoja que tenga las columnas esperadas
            hoja_encontrada = None
            df_final = None
            
            for hoja in hojas_disponibles:
                try:
                    df_temp = pd.read_excel(file_path, sheet_name=hoja)
                    print(f"DEBUG: Probando hoja '{hoja}' - Columnas: {list(df_temp.columns)}")
                    
                    # Verificar si esta hoja tiene las columnas que necesitamos
                    columnas_encontradas = [col for col in columnas_esperadas if col in df_temp.columns]
                    if len(columnas_encontradas) >= 3:  # Al menos 3 columnas esperadas
                        hoja_encontrada = hoja
                        df_final = df_temp
                        print(f"DEBUG: Hoja adecuada encontrada: {hoja}")
                        break
                        
                except Exception as e:
                    print(f"DEBUG: Error leyendo hoja {hoja}: {e}")
                    continue
            
            if df_final is None:
                # Si no encontramos una hoja con las columnas, usar la primera hoja
                print("DEBUG: Usando primera hoja por defecto")
                df_final = pd.read_excel(file_path, sheet_name=0)
                hoja_encontrada = hojas_disponibles[0]
            
        except Exception as e:
            return False, f"No se pudo leer el archivo Excel: {str(e)}", 0
        
        num_filas = len(df_final)
        
        # Verificar columnas
        columnas_faltantes = [col for col in columnas_esperadas if col not in df_final.columns]
        columnas_encontradas = [col for col in columnas_esperadas if col in df_final.columns]
        
        mensaje_hoja = f"Hoja: {hoja_encontrada}"
        if columnas_faltantes:
            mensaje_validacion = f"Advertencia: Faltan columnas {columnas_faltantes}. Encontradas: {columnas_encontradas}"
        else:
            mensaje_validacion = f"Todas las columnas requeridas encontradas"
        
        print(f"DEBUG: Validación completada - {mensaje_hoja}, {mensaje_validacion}, Filas: {num_filas}")
        
        return True, f"{mensaje_hoja}. {mensaje_validacion}", num_filas
        
    except Exception as e:
        print(f"DEBUG: Error general en validar_archivo_excel: {str(e)}")
        return False, f"Error validando archivo: {str(e)}", 0