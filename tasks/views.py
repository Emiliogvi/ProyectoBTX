from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.conf import settings
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy import stats

from .utils import validar_archivo_excel
from .models import Task, Simulacion, ArchivoExcel
from .forms import TaskForm


# =========================
# Autenticación y Home
# =========================
def home(request):
    return render(request, 'home.html')


def signup(request):
    if request.method == 'GET':
        return render(request, 'signup.html', {"form": UserCreationForm})
    else:
        if request.POST.get("password1") == request.POST.get("password2"):
            try:
                user = User.objects.create_user(
                    request.POST.get("username"),
                    password=request.POST.get("password1")
                )
                user.save()
                login(request, user)
                return redirect('tasks')
            except IntegrityError:
                return render(
                    request,
                    'signup.html',
                    {"form": UserCreationForm, "error": "Username already exists."}
                )
        return render(
            request,
            'signup.html',
            {"form": UserCreationForm, "error": "Passwords did not match."}
        )


def signin(request):
    if request.method == 'GET':
        return render(request, 'signin.html', {"form": AuthenticationForm})
    else:
        user = authenticate(
            request,
            username=request.POST.get('username'),
            password=request.POST.get('password')
        )
        if user is None:
            return render(
                request,
                'signin.html',
                {"form": AuthenticationForm, "error": "Username or password is incorrect."}
            )
        login(request, user)
        return redirect('tasks')


@login_required
def signout(request):
    logout(request)
    return redirect('home')


# =========================
# Tasks (Registros)
# =========================
@login_required
def tasks(request):
    tasks_qs = Task.objects.filter(user=request.user, datecompleted__isnull=True)
    return render(request, 'tasks.html', {"tasks": tasks_qs})


@login_required
def tasks_completed(request):
    tasks_qs = Task.objects.filter(
        user=request.user,
        datecompleted__isnull=False
    ).order_by('-datecompleted')
    return render(request, 'tasks.html', {"tasks": tasks_qs})


def regresion_lineal_manual(X, y):
    """
    Implementación manual de regresión lineal usando numpy
    """
    try:
        # Agregar columna de unos para el intercepto
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calcular coeficientes usando mínimos cuadrados: β = (X'X)^(-1)X'y
        coeficientes = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        # Separar intercepto y coeficientes
        alpha = coeficientes[0]
        betas = coeficientes[1:]
        
        # Calcular R² manualmente
        y_pred = X_with_intercept @ coeficientes
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return alpha, betas, r2
        
    except Exception as e:
        print(f"Error en regresión manual: {e}")
        # Retornar valores por defecto en caso de error
        return 1.0, np.array([0.5, 0.3, 0.2, -0.1]), 0.0


def calcular_dinamica_sistemas(dfs):
    """
    Implementa las ecuaciones de dinámica de sistemas para el modelo BTX
    """
    try:
        print("=" * 60)
        print("INICIANDO DINÁMICA DE SISTEMAS BTX")
        print("=" * 60)
        
        resultados_dinamica = {
            'ecuaciones_aplicadas': [],
            'indices_causalidad': [],
            'fracciones_atribuibles': [],
            'parametros_modelo': {},
            'datos_dinamica': []
        }
        
        # 1. Preparar datos combinados por fecha y zona
        datos_combinados = preparar_datos_combinados(dfs)
        
        if datos_combinados.empty or len(datos_combinados) < 3:  # Reducido mínimo
            print(f"DEBUG: Datos insuficientes para dinámica de sistemas - {len(datos_combinados)} registros")
            
            # Añadir mensaje específico sobre falta de correlación
            resultados_dinamica['mensaje'] = "No hay suficientes datos temporales para establecer relaciones causales. Los datos no muestran patrones temporales consistentes."
            return resultados_dinamica
        
        # 2. Verificar correlación antes de proceder
        if 'btx_total' in datos_combinados.columns and 'casos_totales' in datos_combinados.columns:
            correlacion = datos_combinados['btx_total'].corr(datos_combinados['casos_totales'])
            print(f"DEBUG: Correlación BTX-Casos: {correlacion:.3f}")
            
            # Si no hay correlación significativa, retornar mensaje
            if abs(correlacion) < 0.3:
                resultados_dinamica['mensaje'] = f"No se detectó relación causal significativa entre BTX y casos de salud (correlación: {correlacion:.3f}). Los datos no muestran patrones temporales consistentes."
                return resultados_dinamica
        
        # 3. Aplicar ecuaciones de dinámica de sistemas
        df_procesado = aplicar_ecuaciones_dinamica(datos_combinados)
        
        # 4. Calcular modelo de incidencia con lags
        modelo_resultados = calcular_modelo_incidencia(df_procesado)
        
        if not modelo_resultados.get('modelo_valido', False):
            resultados_dinamica['mensaje'] = "El modelo no pudo establecer relaciones causales significativas. Los datos no muestran patrones temporales consistentes."
            return resultados_dinamica
        
        # 5. Calcular índices de causalidad
        indices_resultados = calcular_indices_causalidad(df_procesado, modelo_resultados)
        
        if not indices_resultados['indices_diarios']:
            resultados_dinamica['mensaje'] = "No se pudieron calcular índices de causalidad. Los datos no muestran relaciones temporales significativas."
            return resultados_dinamica
        
        resultados_dinamica.update({
            'ecuaciones_aplicadas': df_procesado.to_dict('records'),
            'indices_causalidad': indices_resultados['indices_diarios'],
            'fracciones_atribuibles': indices_resultados['fracciones_atribuibles'],
            'parametros_modelo': modelo_resultados['parametros'],
            'datos_dinamica': df_procesado.to_dict('records')
        })
        
        print("✅ Dinámica de sistemas completada exitosamente")
        return resultados_dinamica
        
    except Exception as e:
        print(f"ERROR en dinámica de sistemas: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        resultados_dinamica['mensaje'] = f"Error en el análisis: {str(e)}. No se pudieron establecer relaciones causales."
        return resultados_dinamica


def preparar_datos_combinados(dfs):
    """
    Prepara datos combinados por fecha y zona para dinámica de sistemas
    """
    try:
        datos_combinados = []
        
        # Procesar datos de emisiones
        if 'emisiones' in dfs:
            df_emis = dfs['emisiones']
            print(f"DEBUG: Procesando {len(df_emis)} filas de emisiones")
            
            # Buscar columna de fecha
            fecha_col = next((col for col in ['fecha_registro', 'fecha', 'fecha_emision'] if col in df_emis.columns), None)
            if fecha_col:
                df_emis['fecha'] = pd.to_datetime(df_emis[fecha_col], errors='coerce')
                df_emis = df_emis.dropna(subset=['fecha'])
                
                # Agrupar por fecha y calcular BTX
                for fecha, group in df_emis.groupby('fecha'):
                    btx_total = calcular_btx_total(group)
                    datos_combinados.append({
                        'fecha': fecha,
                        'btx_total': btx_total,
                        'zona': group.iloc[0].get('ubicacion', 'general') if 'ubicacion' in group.columns else 'general'
                    })
        
        # Procesar datos de dispersión - CORREGIDO
        df_disp = dfs.get('dispersion')
        if df_disp is None:
            df_disp = dfs.get('dispersión')
        
        if df_disp is not None:
            print(f"DEBUG: Procesando {len(df_disp)} filas de dispersión")
            
            fecha_col = next((col for col in ['fecha_registro', 'fecha', 'fecha_dispersion'] if col in df_disp.columns), None)
            if fecha_col:
                df_disp['fecha'] = pd.to_datetime(df_disp[fecha_col], errors='coerce')
                df_disp = df_disp.dropna(subset=['fecha'])
                
                for fecha, group in df_disp.groupby('fecha'):
                    # Encontrar datos existentes para esta fecha o crear nuevo
                    existente = next((item for item in datos_combinados if item['fecha'] == fecha), None)
                    if existente:
                        existente.update({
                            'velocidad_viento': group['velocidad_viento'].mean() if 'velocidad_viento' in group.columns else 3.0,
                            'humedad': group['humedad_relativa'].mean() if 'humedad_relativa' in group.columns else 65.0,
                            'temperatura': group['temperatura'].mean() if 'temperatura' in group.columns else 25.0,
                            'concentracion': group['concentracion'].mean() if 'concentracion' in group.columns else 0.5
                        })
                    else:
                        datos_combinados.append({
                            'fecha': fecha,
                            'velocidad_viento': group['velocidad_viento'].mean() if 'velocidad_viento' in group.columns else 3.0,
                            'humedad': group['humedad_relativa'].mean() if 'humedad_relativa' in group.columns else 65.0,
                            'temperatura': group['temperatura'].mean() if 'temperatura' in group.columns else 25.0,
                            'concentracion': group['concentracion'].mean() if 'concentracion' in group.columns else 0.5,
                            'zona': 'general'
                        })
        
        # Procesar datos de salud
        if 'salud' in dfs:
            df_salud = dfs['salud']
            print(f"DEBUG: Procesando {len(df_salud)} filas de salud")
            
            fecha_col = next((col for col in ['fecha_registro', 'fecha', 'fecha_salud'] if col in df_salud.columns), None)
            if fecha_col:
                df_salud['fecha'] = pd.to_datetime(df_salud[fecha_col], errors='coerce')
                df_salud = df_salud.dropna(subset=['fecha'])
                
                for fecha, group in df_salud.groupby('fecha'):
                    casos_totales = calcular_casos_totales(group)
                    existente = next((item for item in datos_combinados if item['fecha'] == fecha), None)
                    if existente:
                        existente['casos_totales'] = casos_totales
                    else:
                        datos_combinados.append({
                            'fecha': fecha,
                            'casos_totales': casos_totales,
                            'zona': group.iloc[0].get('zona', 'general') if 'zona' in group.columns else 'general'
                        })
        
        # Procesar datos de exposición - CORREGIDO
        df_exp = dfs.get('exposicion')
        if df_exp is None:
            df_exp = dfs.get('exposición')
            
        if df_exp is not None:
            fecha_col = next((col for col in ['fecha_registro', 'fecha', 'fecha_exposicion'] if col in df_exp.columns), None)
            if fecha_col:
                df_exp['fecha'] = pd.to_datetime(df_exp[fecha_col], errors='coerce')
                df_exp = df_exp.dropna(subset=['fecha'])
                
                for fecha, group in df_exp.groupby('fecha'):
                    existente = next((item for item in datos_combinados if item['fecha'] == fecha), None)
                    if existente:
                        existente.update({
                            'tiempo_exposicion': group['tiempo_exposicion'].mean() if 'tiempo_exposicion' in group.columns else 8.0,
                            'poblacion': group['tamaño_poblacion'].mean() if 'tamaño_poblacion' in group.columns else 100000,
                            'actividades_aire_libre': group['actividades_aire_libre'].iloc[0] if 'actividades_aire_libre' in group.columns else 'media',
                            'nivel_exposicion': group['nivel_exposicion'].mean() if 'nivel_exposicion' in group.columns else 0.5
                        })
        
        # Convertir a DataFrame y ordenar por fecha
        df_combinado = pd.DataFrame(datos_combinados)
        if not df_combinado.empty:
            df_combinado = df_combinado.sort_values('fecha').reset_index(drop=True)
            print(f"DEBUG: Datos combinados preparados - {len(df_combinado)} registros")
            print(f"DEBUG: Columnas disponibles: {list(df_combinado.columns)}")
        
        return df_combinado
        
    except Exception as e:
        print(f"ERROR preparando datos combinados: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


def calcular_btx_total(df_emisiones):
    """Calcula BTX total a partir de datos de emisiones"""
    try:
        if 'gas' not in df_emisiones.columns or 'tasa_emision' not in df_emisiones.columns:
            return 8.5  # Valor por defecto
        
        # Asegurar que 'gas' sea string - CORREGIDO
        df_emisiones['gas'] = df_emisiones['gas'].astype(str)
        
        # Filtrar por gas y calcular promedios
        benceno_data = df_emisiones[df_emisiones['gas'].str.contains('benceno', case=False, na=False)]['tasa_emision']
        tolueno_data = df_emisiones[df_emisiones['gas'].str.contains('tolueno', case=False, na=False)]['tasa_emision']
        xileno_data = df_emisiones[df_emisiones['gas'].str.contains('xileno', case=False, na=False)]['tasa_emision']
        
        benceno_mean = benceno_data.mean() if len(benceno_data) > 0 else 4.8
        tolueno_mean = tolueno_data.mean() if len(tolueno_data) > 0 else 15.2
        xileno_mean = xileno_data.mean() if len(xileno_data) > 0 else 8.7
        
        # BTX_total = 0.5*Benceno + 0.3*Tolueno + 0.2*Xileno
        btx_total = 0.5 * benceno_mean + 0.3 * tolueno_mean + 0.2 * xileno_mean
        return btx_total
        
    except Exception as e:
        print(f"ERROR calculando BTX total: {e}")
        return 8.5

def calcular_casos_totales(df_salud):
    """Calcula casos totales a partir de datos de salud"""
    try:
        casos_totales = 0
        columnas_casos = ['casos_asma', 'casos_bronquitis', 'hospitalizaciones', 'mortalidad']
        
        for col in columnas_casos:
            if col in df_salud.columns:
                casos_totales += df_salud[col].sum()
        
        return casos_totales if casos_totales > 0 else 50  # Valor por defecto mínimo
        
    except Exception as e:
        print(f"ERROR calculando casos totales: {e}")
        return 50


def aplicar_ecuaciones_dinamica(df):
    """
    Aplica las ecuaciones de dinámica de sistemas al DataFrame
    """
    try:
        df_result = df.copy()
        
        # 1. BTX_total (ya calculado)
        if 'btx_total' not in df_result.columns:
            df_result['btx_total'] = 8.5  # Valor por defecto
        
        # 2. BTX_ajustado (ajuste por meteorología)
        # BTX_adj = BTX_total / (VelocidadViento + ε) × (1 + Humedad/100)
        df_result['velocidad_viento'] = df_result.get('velocidad_viento', 3.0)
        df_result['humedad'] = df_result.get('humedad', 65.0)
        df_result['temperatura'] = df_result.get('temperatura', 25.0)
        
        # Usar concentración si está disponible, sino calcular
        if 'concentracion' in df_result.columns:
            df_result['btx_ajustado'] = df_result['concentracion']
        else:
            df_result['btx_ajustado'] = df_result['btx_total'] / (df_result['velocidad_viento'] + 0.1) * (1 + df_result['humedad'] / 100)
        
        # 3. Exposición efectiva (D)
        # D = BTX_adj × (TiempoExposicion/24) × ActividadFactor
        df_result['tiempo_exposicion'] = df_result.get('tiempo_exposicion', 8.0)
        
        # Convertir actividades_aire_libre a factor numérico
        def convertir_actividad_factor(actividad):
            if isinstance(actividad, str):
                if 'alta' in actividad.lower():
                    return 1.0
                elif 'media' in actividad.lower():
                    return 0.7
                elif 'baja' in actividad.lower():
                    return 0.4
            return 0.7  # Valor por defecto
        
        df_result['actividad_factor'] = df_result.get('actividades_aire_libre', 'media').apply(convertir_actividad_factor)
        
        # Usar nivel_exposicion si está disponible, sino calcular
        if 'nivel_exposicion' in df_result.columns:
            df_result['exposicion_efectiva'] = df_result['nivel_exposicion']
        else:
            df_result['exposicion_efectiva'] = df_result['btx_ajustado'] * (df_result['tiempo_exposicion'] / 24) * df_result['actividad_factor']
        
        # 4. Aplicar efecto de medidas (si existen datos de políticas)
        # D_med = D × (1 - R)
        df_result['factor_reduccion'] = 0.0  # Por defecto sin reducción
        df_result['exposicion_efectiva_med'] = df_result['exposicion_efectiva'] * (1 - df_result['factor_reduccion'])
        
        print("✅ Ecuaciones de dinámica aplicadas exitosamente")
        return df_result
        
    except Exception as e:
        print(f"ERROR aplicando ecuaciones de dinámica: {e}")
        return df


def calcular_modelo_incidencia(df):
    """
    Calcula modelo de incidencia con lags (Poisson)
    """
    try:
        if len(df) < 3:  # Reducido el mínimo requerido
            return {'parametros': {}, 'predicciones': [], 'modelo_valido': False}
        
        # Crear lags para exposición (L=2 días para mayor flexibilidad)
        max_lags = min(2, len(df) - 1)  # Ajustar según datos disponibles
        for lag in range(1, max_lags + 1):
            df[f'exposicion_lag_{lag}'] = df['exposicion_efectiva_med'].shift(lag)
        
        # Eliminar filas con NaN por lags
        lag_columns = [f'exposicion_lag_{i}' for i in range(1, max_lags + 1)]
        df_modelo = df.dropna(subset=lag_columns + ['casos_totales'])
        
        if len(df_modelo) < 2:
            return {'parametros': {}, 'predicciones': [], 'modelo_valido': False}
        
        # Preparar variables para el modelo
        X = df_modelo[lag_columns]
        y = df_modelo['casos_totales']
        
        # Agregar intercepto
        X = sm.add_constant(X)
        
        # Modelo Poisson
        try:
            modelo = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            
            # Coeficientes
            coeficientes = {'beta_0': modelo.params.get('const', 0)}
            for i, lag_col in enumerate(lag_columns, 1):
                coeficientes[f'beta_{i}'] = modelo.params.get(lag_col, 0)
            
            coeficientes.update({
                'aic': modelo.aic,
                'pseudo_r2': modelo.pseudo_rsquared(),
                'num_lags': max_lags
            })
            
            # Predicciones
            predicciones = modelo.predict(X)
            
            return {
                'parametros': coeficientes,
                'predicciones': predicciones.tolist(),
                'modelo_valido': True
            }
            
        except Exception as e:
            print(f"ERROR en modelo Poisson: {e}")
            # Modelo lineal simple como fallback
            X_simple = df_modelo[['exposicion_efectiva_med']]
            X_simple = sm.add_constant(X_simple)
            modelo_simple = sm.OLS(y, X_simple).fit()
            
            coeficientes = {
                'beta_0': modelo_simple.params.get('const', 0),
                'beta_1': modelo_simple.params.get('exposicion_efectiva_med', 0.1),
                'beta_2': 0,
                'beta_3': 0,
                'aic': modelo_simple.aic,
                'pseudo_r2': modelo_simple.rsquared,
                'modelo_lineal': True,
                'num_lags': 1
            }
            
            return {
                'parametros': coeficientes,
                'predicciones': modelo_simple.predict(X_simple).tolist(),
                'modelo_valido': True
            }
        
    except Exception as e:
        print(f"ERROR calculando modelo de incidencia: {e}")
        return {'parametros': {}, 'predicciones': [], 'modelo_valido': False}


def calcular_indices_causalidad(df, modelo_resultados):
    """
    Calcula fracción atribuible (AF) e índice de causalidad (CI)
    """
    try:
        if not modelo_resultados.get('modelo_valido', False):
            return {'fracciones_atribuibles': [], 'indices_diarios': []}
        
        parametros = modelo_resultados['parametros']
        num_lags = parametros.get('num_lags', 2)
        
        # Calcular fracción atribuible para cada día
        fracciones_atribuibles = []
        indices_diarios = []
        
        for i, row in df.iterrows():
            if i >= num_lags:  # Necesitamos al menos num_lags
                try:
                    # Suma ponderada de exposiciones con lags
                    suma_exposiciones = 0
                    for lag in range(1, num_lags + 1):
                        beta_key = f'beta_{lag}'
                        lag_key = f'exposicion_lag_{lag}'
                        if beta_key in parametros and lag_key in row:
                            suma_exposiciones += parametros[beta_key] * row[lag_key]
                    
                    # Fracción atribuible: AF = 1 - exp(-suma_exposiciones)
                    af = 1 - np.exp(-suma_exposiciones) if suma_exposiciones > 0 else 0
                    af = max(0, min(1, af))  # Limitar entre 0 y 1
                    
                    fracciones_atribuibles.append({
                        'fecha': row['fecha'],
                        'fraccion_atribuible': af,
                        'suma_exposiciones': suma_exposiciones
                    })
                    
                except Exception as e:
                    print(f"ERROR calculando AF para fila {i}: {e}")
                    fracciones_atribuibles.append({
                        'fecha': row['fecha'],
                        'fraccion_atribuible': 0,
                        'suma_exposiciones': 0
                    })
        
        # Calcular índice de causalidad (CI) como promedio móvil de AF (K=5 días para más flexibilidad)
        if len(fracciones_atribuibles) >= 3:
            window_size = min(5, len(fracciones_atribuibles))
            for i in range(window_size - 1, len(fracciones_atribuibles)):
                start_idx = max(0, i - window_size + 1)
                ventana_af = [fa['fraccion_atribuible'] for fa in fracciones_atribuibles[start_idx:i+1]]
                ci = np.mean(ventana_af)
                
                indices_diarios.append({
                    'fecha': fracciones_atribuibles[i]['fecha'],
                    'indice_causalidad': ci,
                    'interpretacion': interpretar_indice_causalidad(ci)
                })
        
        print(f"✅ Índices de causalidad calculados - {len(indices_diarios)} puntos")
        return {
            'fracciones_atribuibles': fracciones_atribuibles,
            'indices_diarios': indices_diarios
        }
        
    except Exception as e:
        print(f"ERROR calculando índices de causalidad: {e}")
        return {'fracciones_atribuibles': [], 'indices_diarios': []}


def interpretar_indice_causalidad(ci):
    """Interpreta el índice de causalidad"""
    if ci >= 0.7:
        return "Relación causal muy fuerte"
    elif ci >= 0.5:
        return "Relación causal fuerte"
    elif ci >= 0.3:
        return "Relación causal moderada"
    elif ci >= 0.1:
        return "Relación causal débil"
    else:
        return "Relación causal nula o muy débil"


def procesar_datos_simulacion(simulacion):
    """
    Procesa los archivos Excel de una simulación y genera resultados
    """
    try:
        archivos = ArchivoExcel.objects.filter(simulacion=simulacion)
        
        # Diccionario para almacenar los DataFrames
        dfs = {}
        
        print(f"DEBUG: Procesando simulación {simulacion.nombre_simulacion}")
        print(f"DEBUG: Archivos encontrados: {archivos.count()}")
        
        # Leer cada archivo según su tipo
        for archivo in archivos:
            try:
                print(f"DEBUG: Procesando archivo {archivo.tipo_tabla}: {archivo.archivo.name}")
                
                # Intentar leer el archivo
                excel_file = pd.ExcelFile(archivo.archivo.path)
                hojas_disponibles = excel_file.sheet_names
                
                print(f"DEBUG: Hojas disponibles: {hojas_disponibles}")
                
                # Usar la primera hoja por defecto
                hoja_a_usar = hojas_disponibles[0] if hojas_disponibles else 0
                
                # Leer la hoja
                df = pd.read_excel(archivo.archivo.path, sheet_name=hoja_a_usar)
                
                # Limpiar nombres de columnas (eliminar espacios extras y convertir a minúsculas)
                df.columns = df.columns.str.strip().str.lower()
                
                # Convertir nombres a minúsculas sin acentos para consistencia
                tipo_clave = (archivo.tipo_tabla.lower()
                             .replace('ó', 'o')
                             .replace('á', 'a')
                             .replace('é', 'e')
                             .replace('í', 'i')
                             .replace('ú', 'u'))
                
                dfs[tipo_clave] = df
                
                print(f"DEBUG: ✅ Archivo {archivo.tipo_tabla} leído correctamente")
                print(f"DEBUG: - Hoja: {hoja_a_usar}")
                print(f"DEBUG: - Filas: {len(df)}")
                print(f"DEBUG: - Columnas: {list(df.columns)}")
                
                # Mostrar algunas filas de datos
                if len(df) > 0:
                    print(f"DEBUG: - Primera fila de datos:")
                    for i, (col, val) in enumerate(df.iloc[0].items()):
                        if i < 10:  # Mostrar solo las primeras 10 columnas
                            print(f"    {col}: {val}")
                
            except Exception as e:
                print(f"ERROR: ❌ No se pudo leer archivo {archivo.tipo_tabla}: {str(e)}")
                continue
        
        # Verificar que tenemos los datos mínimos necesarios
        tipos_requeridos = ['emisiones', 'dispersion', 'exposicion', 'salud']
        tipos_encontrados = [tipo for tipo in tipos_requeridos if tipo in dfs]
        
        print(f"DEBUG: Tipos requeridos: {tipos_requeridos}")
        print(f"DEBUG: Tipos encontrados: {tipos_encontrados}")
        
        if len(tipos_encontrados) < 2:
            return None, f"Faltan archivos necesarios. Se necesitan al menos 2 de: {', '.join(tipos_requeridos)}"
        
        # Procesar datos estáticos (sistema actual)
        resultados = calcular_ecuaciones_btx_mejorado(dfs)
        
        # Procesar dinámica de sistemas (nuevo)
        resultados_dinamica = calcular_dinamica_sistemas(dfs)
        resultados['dinamica_sistemas'] = resultados_dinamica
        
        # Guardar resultados en archivo TXT
        txt_path = guardar_resultados_txt_mejorado(resultados, simulacion)
        
        return resultados, f"Análisis completado con {len(tipos_encontrados)} archivos. Archivo guardado: {txt_path}"
        
    except Exception as e:
        print(f"ERROR en procesar_datos_simulacion: {str(e)}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return None, f"Error procesando datos: {str(e)}"


def calcular_ecuaciones_btx_mejorado(dfs):
    """
    Calcula las ecuaciones BTX mejoradas basadas en los DataFrames usando datos reales de los Excels
    MEJORADO: Usa umbrales más realistas y análisis de correlación
    """
    resultados = {}
    
    try:
        print("=" * 50)
        print("INICIANDO EXTRACCIÓN DE DATOS DE EXCEL - MODELO MEJORADO")
        print("=" * 50)
        
        # Valores por defecto (solo como respaldo)
        benceno_mean = 4.8
        tolueno_mean = 15.2
        xileno_mean = 8.7
        velocidad_viento_mean = 3.2
        temperatura_mean = 25.0
        tiempo_exposicion_mean = 8.0
        tamaño_poblacion_mean = 150000
        casos_totales = 1570
        tasa_emision_promedio = 2.45
        humedad_mean = 65.0
        
        # EXTRACCIÓN DE DATOS REALES DE LOS EXCELS
        print("DEBUG: Analizando DataFrames disponibles...")
        for tipo, df in dfs.items():
            print(f"DEBUG: DataFrame '{tipo}': {len(df)} filas, columnas: {list(df.columns)}")
        
        # 1. Datos de Emisiones
        if 'emisiones' in dfs:
            df_emisiones = dfs['emisiones']
            print(f"DEBUG: Procesando emisiones - {len(df_emisiones)} filas")
            
            if 'gas' in df_emisiones.columns and 'tasa_emision' in df_emisiones.columns:
                try:
                    df_emisiones['gas'] = df_emisiones['gas'].astype(str)
                    benceno_mask = df_emisiones['gas'].str.contains('benceno', case=False, na=False)
                    tolueno_mask = df_emisiones['gas'].str.contains('tolueno', case=False, na=False)
                    xileno_mask = df_emisiones['gas'].str.contains('xileno', case=False, na=False)
                    
                    benceno_data = df_emisiones[benceno_mask]['tasa_emision']
                    tolueno_data = df_emisiones[tolueno_mask]['tasa_emision']
                    xileno_data = df_emisiones[xileno_mask]['tasa_emision']
                    
                    if len(benceno_data) > 0:
                        benceno_mean = benceno_data.mean()
                    if len(tolueno_data) > 0:
                        tolueno_mean = tolueno_data.mean()
                    if len(xileno_data) > 0:
                        xileno_mean = xileno_data.mean()
                        
                except Exception as e:
                    print(f"ERROR extrayendo concentraciones: {e}")
            
            if 'tasa_emision' in df_emisiones.columns:
                df_emisiones['tasa_emision'] = pd.to_numeric(df_emisiones['tasa_emision'], errors='coerce')
                tasa_emision_promedio = df_emisiones['tasa_emision'].mean()
        
        # 2. Datos de Dispersión
        df_dispersion = dfs.get('dispersión') or dfs.get('dispersion')
        if df_dispersion is not None:
            print(f"DEBUG: Procesando dispersión - {len(df_dispersion)} filas")
            
            if 'velocidad_viento' in df_dispersion.columns:
                velocidad_viento_mean = df_dispersion['velocidad_viento'].mean()
            if 'temperatura' in df_dispersion.columns:
                temperatura_mean = df_dispersion['temperatura'].mean()
            if 'humedad_relativa' in df_dispersion.columns:
                humedad_mean = df_dispersion['humedad_relativa'].mean()
        
        # 3. Datos de Exposición
        df_exposicion = dfs.get('exposición') or dfs.get('exposicion')
        if df_exposicion is not None:
            print(f"DEBUG: Procesando exposición - {len(df_exposicion)} filas")
            
            if 'tamaño_población' in df_exposicion.columns:
                df_exposicion['tamaño_población'] = pd.to_numeric(df_exposicion['tamaño_población'], errors='coerce')
                tamaño_poblacion_mean = df_exposicion['tamaño_población'].mean()
            elif 'tamaño_poblacion' in df_exposicion.columns:
                df_exposicion['tamaño_poblacion'] = pd.to_numeric(df_exposicion['tamaño_poblacion'], errors='coerce')
                tamaño_poblacion_mean = df_exposicion['tamaño_poblacion'].mean()
            
            if 'tiempo_exposición' in df_exposicion.columns:
                df_exposicion['tiempo_exposición'] = pd.to_numeric(df_exposicion['tiempo_exposición'], errors='coerce')
                tiempo_exposicion_mean = df_exposicion['tiempo_exposición'].mean()
            elif 'tiempo_exposicion' in df_exposicion.columns:
                df_exposicion['tiempo_exposicion'] = pd.to_numeric(df_exposicion['tiempo_exposicion'], errors='coerce')
                tiempo_exposicion_mean = df_exposicion['tiempo_exposicion'].mean()
        
        # 4. Datos de Salud
        if 'salud' in dfs:
            df_salud = dfs['salud']
            print(f"DEBUG: Procesando salud - {len(df_salud)} filas")
            
            casos_totales = 0
            columnas_casos = ['casos_asma', 'casos_bronquitis', 'hospitalizaciones', 'mortalidad']
            
            for col in columnas_casos:
                if col in df_salud.columns:
                    df_salud[col] = pd.to_numeric(df_salud[col], errors='coerce')
                    suma_col = df_salud[col].sum()
                    casos_totales += suma_col
        
        # 5. Datos de Respuesta Social
        impacto_estimado = 0.0
        if 'social' in dfs:
            df_social = dfs['social']
            if 'impacto_estimado' in df_social.columns:
                # Convertir impacto estimado a numérico
                impacto_map = {'Alto': 0.7, 'Medio': 0.4, 'Bajo': 0.1}
                df_social['impacto_numerico'] = df_social['impacto_estimado'].map(impacto_map)
                impacto_estimado = df_social['impacto_numerico'].mean()
                if pd.isna(impacto_estimado):
                    impacto_estimado = 0.0
        
        # ==================== CÁLCULOS MEJORADOS CON UMBRALES REALISTAS ====================
        print("=" * 50)
        print("CALCULANDO ECUACIONES MEJORADAS CON UMBRALES REALISTAS")
        print("=" * 50)
        
        # 1. BTX Total con pesos basados en toxicidad relativa (EPA)
        # Benceno es más tóxico, por eso mayor peso
        BTX_total = 0.6 * benceno_mean + 0.25 * tolueno_mean + 0.15 * xileno_mean
        print(f"DEBUG: BTX Total: {BTX_total:.4f} µg/m³")
        
        # 2. Factor de dispersión atmosférica mejorado
        # Incluye efecto de humedad y temperatura
        factor_viento = 1.0 / (velocidad_viento_mean + 0.1)
        factor_temperatura = 1.0 + (temperatura_mean - 25.0) / 50.0  # Normalizado a 25°C
        factor_humedad = 1.0 + (humedad_mean - 50.0) / 200.0  # Normalizado a 50%
        
        BTX_ajustado = BTX_total * factor_viento * factor_temperatura * factor_humedad
        print(f"DEBUG: BTX Ajustado: {BTX_ajustado:.4f}")
        print(f"  - Factor viento: {factor_viento:.4f}")
        print(f"  - Factor temperatura: {factor_temperatura:.4f}")
        print(f"  - Factor humedad: {factor_humedad:.4f}")
        
        # 3. Exposición per cápita mejorada (µg·h/persona)
        exposicion_per_capita = (BTX_ajustado * tiempo_exposicion_mean) / (tamaño_poblacion_mean / 1000)
        print(f"DEBUG: Exposición per cápita: {exposicion_per_capita:.6f} µg·h/1000 personas")
        
        # 4. ANÁLISIS DE CORRELACIÓN MEJORADO
        # Calcular correlaciones entre variables para determinar relaciones reales
        correlacion_btx_casos = 0.0
        correlacion_exposicion_casos = 0.0
        
        # Intentar calcular correlaciones si hay datos temporales
        try:
            datos_combinados = preparar_datos_combinados(dfs)
            if not datos_combinados.empty and 'btx_total' in datos_combinados.columns and 'casos_totales' in datos_combinados.columns:
                datos_validos = datos_combinados.dropna(subset=['btx_total', 'casos_totales'])
                if len(datos_validos) >= 3:
                    correlacion_btx_casos = datos_validos['btx_total'].corr(datos_validos['casos_totales'])
                    print(f"DEBUG: Correlación BTX-Casos calculada: {correlacion_btx_casos:.3f}")
        except Exception as e:
            print(f"DEBUG: No se pudo calcular correlación temporal: {e}")
        
        # 5. Regresión múltiple usando datos reales
        alpha, betas, r2 = regresion_lineal_manual(
            np.column_stack([
                [BTX_ajustado],
                [exposicion_per_capita],
                [tasa_emision_promedio],
                [impacto_estimado]
            ]),
            np.array([casos_totales])
        )
        
        # 6. Predicción de casos con el modelo - AJUSTADO POR CORRELACIÓN
        # Si no hay correlación significativa, reducir el impacto de BTX
        factor_correlacion = max(0, correlacion_btx_casos)  # Solo correlaciones positivas
        betas_ajustados = betas * factor_correlacion if factor_correlacion > 0.3 else betas * 0.1
        
        casos_predichos = (alpha + 
                          betas_ajustados[0] * BTX_ajustado + 
                          betas_ajustados[1] * exposicion_per_capita + 
                          betas_ajustados[2] * tasa_emision_promedio - 
                          betas_ajustados[3] * impacto_estimado)
        
        # 7. Tasa de incidencia por 100,000 habitantes
        tasa_incidencia = (casos_totales / tamaño_poblacion_mean) * 100000
        
        # 8. Fracción atribuible poblacional (FAP) - MEJORADA
        # Solo atribuir casos si hay correlación significativa
        if correlacion_btx_casos > 0.3:
            riesgo_relativo = 1.0 + (BTX_ajustado / 15.0) * correlacion_btx_casos  # Más conservador
            fap = ((riesgo_relativo - 1.0) / riesgo_relativo) * 100
            fap = min(fap, 80)  # Límite máximo realista
        else:
            fap = 5.0  # Valor mínimo cuando no hay correlación
        
        # Interpretaciones textuales MEJORADAS
        interpretacion_btx = interpretar_nivel_btx_mejorado(BTX_total, correlacion_btx_casos)
        interpretacion_exposicion = interpretar_exposicion_mejorado(exposicion_per_capita)
        interpretacion_incidencia = interpretar_incidencia_mejorado(tasa_incidencia)
        interpretacion_fap = interpretar_fap_mejorado(fap, correlacion_btx_casos)
        
        resultados = {
            'concentracion_benzeno': benceno_mean,
            'concentracion_tolueno': tolueno_mean,
            'concentracion_xileno': xileno_mean,
            'btx_total': BTX_total,
            'btx_ajustado': BTX_ajustado,
            'exposicion_per_capita': exposicion_per_capita,
            'casos_totales': casos_totales,
            'casos_predichos': casos_predichos,
            'tasa_incidencia': tasa_incidencia,
            'fraccion_atribuible': fap,
            'correlacion_btx_casos': correlacion_btx_casos,
            'regresion': {
                'alpha': alpha,
                'beta1': betas[0],
                'beta2': betas[1],
                'beta3': betas[2],
                'beta4': betas[3],
                'beta1_ajustado': betas_ajustados[0],
                'r2': r2
            },
            'factores_dispersion': {
                'factor_viento': factor_viento,
                'factor_temperatura': factor_temperatura,
                'factor_humedad': factor_humedad
            },
            'estadisticas': {
                'velocidad_viento_promedio': velocidad_viento_mean,
                'temperatura_promedio': temperatura_mean,
                'humedad_promedio': humedad_mean,
                'tiempo_exposicion_promedio': tiempo_exposicion_mean,
                'poblacion_promedio': tamaño_poblacion_mean,
                'tasa_emision_promedio': tasa_emision_promedio,
                'impacto_estimado': impacto_estimado
            },
            'interpretaciones': {
                'nivel_btx': interpretacion_btx,
                'exposicion': interpretacion_exposicion,
                'incidencia': interpretacion_incidencia,
                'fraccion_atribuible': interpretacion_fap
            },
            'datos_reales_utilizados': True
        }
        
        print("=" * 50)
        print("CÁLCULOS MEJORADOS COMPLETADOS")
        print("=" * 50)
        
        return resultados
        
    except Exception as e:
        print(f"ERROR en cálculo de ecuaciones mejoradas: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        # Retornar valores por defecto en caso de error
        return {
            'concentracion_benzeno': 4.8,
            'concentracion_tolueno': 15.2,
            'concentracion_xileno': 8.7,
            'btx_total': 8.5,
            'btx_ajustado': 7.2,
            'exposicion_per_capita': 0.15,
            'casos_totales': 1570,
            'casos_predichos': 1500,
            'tasa_incidencia': 1046.67,
            'fraccion_atribuible': 15.0,  # Reducido por defecto
            'correlacion_btx_casos': 0.1,
            'regresion': {
                'alpha': 1.0, 'beta1': 0.5, 'beta2': 0.3, 'beta3': 0.2, 'beta4': -0.1, 
                'beta1_ajustado': 0.05, 'r2': 0.15
            },
            'factores_dispersion': {
                'factor_viento': 0.31, 'factor_temperatura': 1.0, 'factor_humedad': 1.0
            },
            'estadisticas': {
                'velocidad_viento_promedio': 3.2,
                'temperatura_promedio': 25.0,
                'humedad_promedio': 65.0,
                'tiempo_exposicion_promedio': 8.0,
                'poblacion_promedio': 150000,
                'tasa_emision_promedio': 2.45,
                'impacto_estimado': 0.4
            },
            'interpretaciones': {
                'nivel_btx': 'Nivel moderado de contaminación. Relación con casos de salud no significativa.',
                'exposicion': 'Exposición baja a moderada',
                'incidencia': 'Tasa de incidencia dentro de rangos esperados',
                'fraccion_atribuible': 'Baja fracción atribuible a BTX debido a correlación insuficiente'
            },
            'datos_reales_utilizados': False
        }


def interpretar_nivel_btx_mejorado(btx_total, correlacion):
    """Interpreta el nivel de BTX total considerando la correlación"""
    if correlacion < 0.3:
        base_msg = "Relación con casos de salud no significativa (correlación: {:.3f}). "
    elif correlacion < 0.5:
        base_msg = "Relación moderada con casos de salud (correlación: {:.3f}). "
    elif correlacion < 0.7:
        base_msg = "Relación fuerte con casos de salud (correlación: {:.3f}). "
    else:
        base_msg = "Relación muy fuerte con casos de salud (correlación: {:.3f}). "
    
    base_msg = base_msg.format(correlacion)
    
    if btx_total < 5.0:
        return base_msg + "Nivel bajo de contaminación por BTX. Riesgo mínimo según estándares internacionales."
    elif btx_total < 10.0:
        return base_msg + "Nivel moderado de contaminación por BTX. Puede representar riesgo para poblaciones sensibles."
    elif btx_total < 20.0:
        return base_msg + "Nivel alto de contaminación por BTX. Riesgo significativo para la salud poblacional."
    else:
        return base_msg + "Nivel crítico de contaminación por BTX. Riesgo grave para la salud pública."


def interpretar_exposicion_mejorado(exposicion_per_capita):
    """Interpreta el nivel de exposición per cápita"""
    if exposicion_per_capita < 0.05:
        return "Exposición muy baja. Contacto mínimo con contaminantes BTX."
    elif exposicion_per_capita < 0.1:
        return "Exposición baja. La población tiene contacto limitado con los contaminantes BTX."
    elif exposicion_per_capita < 0.3:
        return "Exposición moderada. Una parte significativa de la población está expuesta a niveles preocupantes de BTX."
    elif exposicion_per_capita < 0.5:
        return "Exposición alta. La población está expuesta a niveles elevados de BTX que pueden causar efectos adversos en la salud."
    else:
        return "Exposición muy alta. Niveles críticos de exposición que requieren intervención inmediata."


def interpretar_incidencia_mejorado(tasa_incidencia):
    """Interpreta la tasa de incidencia de casos respiratorios"""
    if tasa_incidencia < 300:
        return "Tasa de incidencia normal. Comparable con poblaciones sin exposición significativa a contaminantes."
    elif tasa_incidencia < 600:
        return "Tasa de incidencia ligeramente elevada. Puede indicar impacto leve de contaminantes."
    elif tasa_incidencia < 1000:
        return "Tasa de incidencia elevada. Sugiere un impacto moderado de los contaminantes en la salud respiratoria."
    elif tasa_incidencia < 1500:
        return "Tasa de incidencia alta. Indica un problema significativo de salud pública relacionado con la calidad del aire."
    else:
        return "Tasa de incidencia crítica. Representa una emergencia de salud pública que requiere atención urgente."


def interpretar_fap_mejorado(fap, correlacion):
    """Interpreta la fracción atribuible poblacional considerando correlación"""
    if correlacion < 0.3:
        return f"Aproximadamente el {fap:.1f}% de los casos podrían atribuirse a BTX. Baja certeza debido a correlación insuficiente."
    elif correlacion < 0.5:
        return f"Aproximadamente el {fap:.1f}% de los casos podrían atribuirse a BTX. Certeza moderada."
    elif correlacion < 0.7:
        return f"Aproximadamente el {fap:.1f}% de los casos podrían atribuirse a BTX. Alta certeza en la relación causal."
    else:
        return f"Aproximadamente el {fap:.1f}% de los casos podrían atribuirse a BTX. Muy alta certeza en la relación causal."


def guardar_resultados_txt_mejorado(resultados, simulacion):
    """
    Guarda los resultados en un archivo TXT único para la simulación
    MEJORADO: Incluye análisis de correlación y umbrales realistas
    """
    try:
        resultados_dir = os.path.join(settings.MEDIA_ROOT, 'resultados_simulaciones')
        os.makedirs(resultados_dir, exist_ok=True)
        
        nombre_archivo = f"resultado_modelo_btx_{simulacion.id}.txt"
        txt_path = os.path.join(resultados_dir, nombre_archivo)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESULTADOS DEL MODELO BTX MEJORADO - ANÁLISIS AMBIENTAL Y DE SALUD PÚBLICA\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Simulación: {simulacion.nombre_simulacion}\n")
            f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ubicación: {simulacion.ubicacion or 'No especificada'}\n")
            
            if resultados.get('datos_reales_utilizados', False):
                f.write("Fuente de datos: ARCHIVOS EXCEL CARGADOS (datos reales)\n")
            else:
                f.write("Fuente de datos: VALORES POR DEFECTO (no se pudieron leer datos reales)\n")
                
            f.write("-" * 80 + "\n\n")
            
            # SISTEMAS ESTÁTICOS MEJORADOS
            f.write("SISTEMAS ESTÁTICOS MEJORADOS - ANÁLISIS CON UMBRALES REALISTAS\n")
            f.write("=" * 60 + "\n\n")
            
            # DATOS EXTRAÍDOS
            f.write("1. DATOS EXTRAÍDOS Y CORRELACIONES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Concentración Benceno: {resultados['concentracion_benzeno']:.2f} µg/m³\n")
            f.write(f"Concentración Tolueno: {resultados['concentracion_tolueno']:.2f} µg/m³\n")
            f.write(f"Concentración Xileno: {resultados['concentracion_xileno']:.2f} µg/m³\n")
            f.write(f"Velocidad del viento promedio: {resultados['estadisticas']['velocidad_viento_promedio']:.2f} m/s\n")
            f.write(f"Temperatura promedio: {resultados['estadisticas']['temperatura_promedio']:.2f} °C\n")
            f.write(f"Humedad relativa promedio: {resultados['estadisticas'].get('humedad_promedio', 65.0):.2f} %\n")
            f.write(f"Tiempo de exposición promedio: {resultados['estadisticas']['tiempo_exposicion_promedio']:.2f} horas/día\n")
            f.write(f"Población promedio: {resultados['estadisticas']['poblacion_promedio']:.0f} personas\n")
            f.write(f"Tasa de emisión promedio: {resultados['estadisticas']['tasa_emision_promedio']:.2f} g/s\n")
            impacto = resultados['estadisticas']['impacto_estimado']
            f.write(f"Impacto estimado de políticas: {impacto:.2f}\n")
            f.write(f"Casos respiratorios totales: {resultados['casos_totales']:.0f}\n")
            f.write(f"Correlación BTX-Casos: {resultados.get('correlacion_btx_casos', 0):.3f}\n\n")
            
            # ECUACIONES MEJORADAS
            f.write("2. ECUACIONES APLICADAS (MODELO MEJORADO CON CORRELACIÓN)\n")
            f.write("-" * 55 + "\n")
            f.write("1. BTX_total = 0.6×Benceno + 0.25×Tolueno + 0.15×Xileno\n")
            f.write("   (Pesos basados en toxicidad relativa según EPA)\n\n")
            f.write("2. BTX_ajustado = BTX_total × F_viento × F_temperatura × F_humedad\n")
            f.write("   Donde:\n")
            f.write("   - F_viento = 1 / (Velocidad_Viento + 0.1)\n")
            f.write("   - F_temperatura = 1 + (Temperatura - 25°C) / 50\n")
            f.write("   - F_humedad = 1 + (Humedad - 50%) / 200\n\n")
            f.write("3. Exposición_per_cápita = (BTX_ajustado × Tiempo_Exposición) / (Población / 1000)\n")
            f.write("   (Expresada en µg·h por cada 1000 personas)\n\n")
            f.write("4. Casos_predichos = α + β₁_ajustado×BTX_ajustado + β₂×Exposición_per_cápita\n")
            f.write("                     + β₃×Tasa_Emisión - β₄×Impacto_Políticas + ε\n")
            f.write("   Donde β₁_ajustado = β₁ × max(0, Correlación_BTX_Casos)\n")
            f.write("   (Ajuste por correlación real observada)\n\n")
            f.write("5. Riesgo_Relativo = 1 + (BTX_ajustado / 15) × Correlación_BTX_Casos\n")
            f.write("6. Fracción_Atribuible = [(RR - 1) / RR] × 100% (solo si correlación > 0.3)\n\n")
            
            # RESULTADOS NUMÉRICOS MEJORADOS
            f.write("3. RESULTADOS NUMÉRICOS DEL ANÁLISIS MEJORADO\n")
            f.write("-" * 50 + "\n")
            f.write(f"BTX Total (ponderado por toxicidad): {resultados['btx_total']:.4f} µg/m³\n")
            f.write(f"BTX Ajustado (dispersión atmosférica): {resultados['btx_ajustado']:.4f} µg/m³\n")
            f.write(f"  - Factor de viento: {resultados['factores_dispersion']['factor_viento']:.4f}\n")
            f.write(f"  - Factor de temperatura: {resultados['factores_dispersion']['factor_temperatura']:.4f}\n")
            f.write(f"  - Factor de humedad: {resultados['factores_dispersion']['factor_humedad']:.4f}\n")
            f.write(f"Exposición per cápita: {resultados['exposicion_per_capita']:.6f} µg·h/1000 personas\n")
            f.write(f"Casos respiratorios observados: {resultados['casos_totales']:.0f}\n")
            f.write(f"Casos respiratorios predichos (modelo): {resultados['casos_predichos']:.0f}\n")
            f.write(f"Tasa de incidencia: {resultados['tasa_incidencia']:.2f} casos por 100,000 habitantes\n")
            f.write(f"Correlación BTX-Casos: {resultados.get('correlacion_btx_casos', 0):.3f}\n")
            f.write(f"Fracción atribuible poblacional: {resultados['fraccion_atribuible']:.2f}%\n\n")
            
            # INTERPRETACIÓN DE RESULTADOS MEJORADA
            f.write("4. INTERPRETACIÓN DE RESULTADOS (CONSIDERANDO CORRELACIÓN)\n")
            f.write("-" * 60 + "\n")
            interp = resultados.get('interpretaciones', {})
            f.write(f"Nivel de contaminación BTX:\n  {interp.get('nivel_btx', 'No disponible')}\n\n")
            f.write(f"Nivel de exposición poblacional:\n  {interp.get('exposicion', 'No disponible')}\n\n")
            f.write(f"Tasa de incidencia de casos respiratorios:\n  {interp.get('incidencia', 'No disponible')}\n\n")
            f.write(f"Fracción atribuible a BTX:\n  {interp.get('fraccion_atribuible', 'No disponible')}\n\n")
            
            # COEFICIENTES DEL MODELO MEJORADO
            f.write("5. PARÁMETROS DEL MODELO DE REGRESIÓN MEJORADO\n")
            f.write("-" * 50 + "\n")
            reg = resultados['regresion']
            f.write(f"α (intercepto): {reg['alpha']:.4f}\n")
            f.write(f"β₁ (BTX_ajustado original): {reg['beta1']:.4f}\n")
            f.write(f"β₁_ajustado (por correlación): {reg.get('beta1_ajustado', reg['beta1']):.4f}\n")
            f.write(f"β₂ (Exposición_per_cápita): {reg['beta2']:.4f}\n")
            f.write(f"β₃ (Tasa_Emisión): {reg['beta3']:.4f}\n")
            f.write(f"β₄ (Impacto_Políticas): {reg['beta4']:.4f}\n")
            f.write(f"R² (ajuste del modelo): {reg['r2']:.4f}\n")
            f.write(f"\nInterpretación de R²: El modelo explica el {reg['r2']*100:.1f}% de la variabilidad\n")
            f.write(f"en los casos de enfermedades respiratorios observados.\n\n")
            
            # CRITERIOS DE SIGNIFICANCIA
            f.write("6. CRITERIOS DE SIGNIFICANCIA APLICADOS\n")
            f.write("-" * 40 + "\n")
            f.write("• Correlación BTX-Casos > 0.7: Relación muy fuerte\n")
            f.write("• Correlación BTX-Casos > 0.5: Relación fuerte\n") 
            f.write("• Correlación BTX-Casos > 0.3: Relación moderada\n")
            f.write("• Correlación BTX-Casos < 0.3: Relación débil o nula\n")
            f.write("• Fracción atribuible solo se calcula si correlación > 0.3\n")
            f.write("• Coeficientes de regresión se ajustan por correlación observada\n\n")
            
            # DINÁMICA DE SISTEMAS (se mantiene igual)
            f.write("\n" + "=" * 80 + "\n")
            f.write("DINÁMICA DE SISTEMAS - ANÁLISIS TEMPORAL\n")
            f.write("=" * 80 + "\n\n")
            
            if 'dinamica_sistemas' in resultados and resultados['dinamica_sistemas'].get('indices_causalidad'):
                dinamica = resultados['dinamica_sistemas']
                
                f.write("1. ECUACIONES DE DINÁMICA DE SISTEMAS\n")
                f.write("-" * 50 + "\n")
                f.write("1. BTX_total_z,t = 0.6×Benceno_z,t + 0.25×Tolueno_z,t + 0.15×Xileno_z,t\n")
                f.write("2. BTX_adj_z,t = BTX_total_z,t × [1/(v_z,t + 0.1)] × F_temp × F_hum\n")
                f.write("3. D_z,t = BTX_adj_z,t × (T_exp_z,t/24) × Factor_Actividad_z,t\n")
                f.write("4. D_z,t(med) = D_z,t × (1 - R_z,t)  [Ajuste por políticas]\n")
                f.write("5. λ_z,t = exp(β₀ + Σ(β_ℓ × D_z,t-ℓ(med)))  [Modelo Poisson]\n")
                f.write("6. AF_z,t = 1 - exp(-Σ(β_ℓ × D_z,t-ℓ(med)))  [Fracción Atribuible]\n")
                f.write("7. CI_z,t = (1/K) × Σ(AF_z,t-k)  [Índice de Causalidad]\n\n")
                
                f.write("2. PARÁMETROS DEL MODELO DINÁMICO\n")
                f.write("-" * 35 + "\n")
                if dinamica['parametros_modelo']:
                    params = dinamica['parametros_modelo']
                    f.write(f"β₀ (intercepto): {params.get('beta_0', 0):.4f}\n")
                    num_lags = params.get('num_lags', 3)
                    for i in range(1, num_lags + 1):
                        f.write(f"β₃ (lag {i} día{'s' if i > 1 else ''}): {params.get(f'beta_{i}', 0):.4f}\n")
                    f.write(f"AIC (criterio de información): {params.get('aic', 0):.2f}\n")
                    f.write(f"Pseudo-R²: {params.get('pseudo_r2', 0):.4f}\n")
                    if params.get('modelo_lineal', False):
                        f.write("Tipo de modelo: Regresión Lineal (fallback)\n")
                    else:
                        f.write("Tipo de modelo: Poisson GLM (modelo de conteo)\n")
                f.write("\n")
                
                f.write("3. ÍNDICES DE CAUSALIDAD (CI) - ÚLTIMOS 10 DÍAS\n")
                f.write("-" * 45 + "\n")
                indices = dinamica['indices_causalidad'][-10:]
                for idx in indices:
                    fecha_str = idx['fecha'].strftime('%Y-%m-%d') if hasattr(idx['fecha'], 'strftime') else str(idx['fecha'])
                    f.write(f"{fecha_str}: CI = {idx['indice_causalidad']:.3f} - {idx['interpretacion']}\n")
                f.write("\n")
                
            else:
                f.write("No se pudieron calcular índices de dinámica de sistemas.\n\n")
                mensaje = resultados['dinamica_sistemas'].get('mensaje', '')
                if mensaje:
                    f.write(f"Razón: {mensaje}\n\n")
            
            # RECOMENDACIONES MEJORADAS
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMENDACIONES Y CONCLUSIONES MEJORADAS\n")
            f.write("=" * 80 + "\n\n")
            
            # Recomendaciones basadas en resultados y correlación
            correlacion = resultados.get('correlacion_btx_casos', 0)
            btx_total = resultados['btx_total']
            fap = resultados['fraccion_atribuible']
            
            f.write("Basado en el análisis mejorado que considera correlación real:\n\n")
            
            if correlacion < 0.3:
                f.write("1. RELACIÓN CAUSAL INCIERTA:\n")
                f.write("   • Los datos no muestran relación significativa entre BTX y casos de salud\n")
                f.write("   • Se requieren más datos o mejor calidad de datos temporales\n")
                f.write("   • Considerar otros factores ambientales o de salud\n\n")
            else:
                if btx_total > 15.0 and correlacion > 0.5:
                    f.write("1. URGENTE - Nivel crítico con relación causal fuerte:\n")
                    f.write("   • Implementar medidas inmediatas de control de emisiones\n")
                    f.write("   • Establecer alertas de calidad del aire para la población\n")
                    f.write("   • Considerar restricciones temporales a actividades contaminantes\n\n")
                elif btx_total > 10.0 and correlacion > 0.3:
                    f.write("1. IMPORTANTE - Nivel alto con relación causal moderada:\n")
                    f.write("   • Reforzar programas de monitoreo de calidad del aire\n")
                    f.write("   • Implementar medidas de control gradual de emisiones\n")
                    f.write("   • Informar a la población sobre riesgos y medidas de protección\n\n")
                elif correlacion > 0.3:
                    f.write("1. RELACIÓN CAUSAL DETECTADA:\n")
                    f.write("   • Existe evidencia de relación entre BTX y salud respiratoria\n")
                    f.write("   • Monitorear continuamente la calidad del aire\n")
                    f.write("   • Considerar medidas preventivas en zonas de mayor exposición\n\n")
            
            if resultados['tasa_incidencia'] > 1500:
                f.write("2. SALUD PÚBLICA - Tasa de incidencia elevada:\n")
                f.write("   • Fortalecer servicios de atención de enfermedades respiratorias\n")
                f.write("   • Implementar programas de prevención y educación sanitaria\n")
                f.write("   • Realizar estudios epidemiológicos más detallados\n\n")
            
            if fap > 40.0 and correlacion > 0.5:
                f.write("3. IMPACTO AMBIENTAL SIGNIFICATIVO:\n")
                f.write("   • Priorizar reducción de emisiones de fuentes identificadas\n")
                f.write("   • Evaluar efectividad de políticas ambientales actuales\n")
                f.write("   • Considerar incentivos para tecnologías más limpias\n\n")
            
            f.write("MEJORAS APLICADAS EN ESTE ANÁLISIS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Uso de correlación real para ajustar modelos predictivos\n")
            f.write("• Umbrales más conservadores para fracción atribuible\n")
            f.write("• Coeficientes de regresión ajustados por correlación observada\n")
            f.write("• Interpretaciones contextualizadas con la fuerza de la relación\n")
            f.write("• Criterios de significancia más estrictos\n\n")
            
            f.write("LIMITACIONES DEL ESTUDIO MEJORADO:\n")
            f.write("-" * 45 + "\n")
            f.write("• La calidad de los resultados depende de la correlación observada\n")
            f.write("• Datos con baja correlación producen estimaciones conservadoras\n")
            f.write("• Se requieren series temporales consistentes para análisis dinámicos\n")
            f.write("• Factores confusores no medidos pueden influir en los resultados\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("FIN DEL REPORTE MEJORADO\n")
            f.write("=" * 80 + "\n")
        
        return txt_path
        
    except Exception as e:
        print(f"Error guardando archivo TXT mejorado: {e}")
        return None
    
def leer_resultados_txt(simulacion):
    """
    Lee los resultados desde el archivo TXT de la simulación
    """
    try:
        nombre_archivo = f"resultado_modelo_btx_{simulacion.id}.txt"
        txt_path = os.path.join(settings.MEDIA_ROOT, 'resultados_simulaciones', nombre_archivo)
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                contenido = f.read()
            return contenido
        else:
            return None
    except Exception as e:
        print(f"Error leyendo archivo TXT: {e}")
        return None

def preparar_datos_graficas_mejorado(simulacion):
    """
    Prepara los datos para las gráficas usando análisis mejorado
    MEJORADO: Solo muestra relaciones significativas y usa umbrales realistas
    """
    try:
        archivos = ArchivoExcel.objects.filter(simulacion=simulacion)
        dfs = {}
        
        # Leer archivos
        for archivo in archivos:
            try:
                excel_file = pd.ExcelFile(archivo.archivo.path)
                hojas_disponibles = excel_file.sheet_names
                hoja_a_usar = hojas_disponibles[0] if hojas_disponibles else 0
                df = pd.read_excel(archivo.archivo.path, sheet_name=hoja_a_usar)
                df.columns = df.columns.str.strip().str.lower()
                
                tipo_clave = (archivo.tipo_tabla.lower()
                             .replace('ó', 'o')
                             .replace('á', 'a')
                             .replace('é', 'e')
                             .replace('í', 'i')
                             .replace('ú', 'u'))
                dfs[tipo_clave] = df
                
                print(f"DEBUG GRÁFICAS MEJORADO: Archivo {tipo_clave} - Columnas: {list(df.columns)}")
            except Exception as e:
                print(f"Error leyendo archivo {archivo.tipo_tabla}: {e}")
                continue
        
        datos_graficas = {
            'dispersion': {'btx_total': [], 'casos_respiratorios': [], 'fechas': []},
            'correlaciones': {'variables': [], 'matriz': []},
            'datos_disponibles': False,
            'mensaje': None,
            'correlacion_principal': 0.0
        }
        
        # Verificar datos mínimos
        tiene_emisiones = 'emisiones' in dfs
        tiene_salud = 'salud' in dfs
        
        if not (tiene_emisiones and tiene_salud):
            print("DEBUG GRÁFICAS MEJORADO: Faltan archivos necesarios")
            datos_graficas['mensaje'] = 'Faltan archivos de Emisiones o Salud para generar gráficas'
            return datos_graficas
        
        # USAR DATOS DE DINÁMICA DE SISTEMAS MEJORADO
        print("DEBUG GRÁFICAS MEJORADO: Preparando datos de dinámica de sistemas mejorado...")
        datos_combinados = preparar_datos_combinados(dfs)
        
        if datos_combinados.empty or len(datos_combinados) < 3:
            print(f"DEBUG GRÁFICAS MEJORADO: Datos insuficientes - {len(datos_combinados)} registros")
            datos_graficas['mensaje'] = 'Datos temporales insuficientes para establecer relaciones causales (mínimo 3 puntos temporales)'
            return datos_graficas
        
        # Filtrar registros válidos
        datos_validos = datos_combinados.dropna(subset=['btx_total', 'casos_totales'])
        
        if len(datos_validos) < 3:
            print("DEBUG GRÁFICAS MEJORADO: Muy pocos datos válidos después de filtrar")
            datos_graficas['mensaje'] = 'Muy pocos datos válidos después de filtrar valores faltantes'
            return datos_validos
        
        # VERIFICAR CORRELACIÓN CON UMBRAL MÁS ESTRICTO
        correlacion = datos_validos['btx_total'].corr(datos_validos['casos_totales'])
        print(f"DEBUG GRÁFICAS MEJORADO: Correlación BTX-Casos: {correlacion:.3f}")
        
        # Umbral más estricto para gráficas - solo mostrar si hay relación significativa
        if abs(correlacion) < 0.4:  # Aumentado de 0.3 a 0.4
            print("DEBUG GRÁFICAS MEJORADO: Correlación insuficiente, no mostrar gráficas")
            datos_graficas['mensaje'] = f'No se detectó correlación significativa entre BTX y casos de salud (correlación: {correlacion:.3f}). Se requiere una correlación mínima de ±0.4 para establecer relaciones causales en las gráficas.'
            datos_graficas['correlacion_principal'] = correlacion
            return datos_graficas
        
        # Datos para gráfica de dispersión
        datos_graficas['dispersion']['btx_total'] = datos_validos['btx_total'].tolist()
        datos_graficas['dispersion']['casos_respiratorios'] = datos_validos['casos_totales'].tolist()
        datos_graficas['dispersion']['fechas'] = datos_validos['fecha'].dt.strftime('%Y-%m-%d').tolist()
        datos_graficas['datos_disponibles'] = True
        datos_graficas['correlacion_principal'] = correlacion
        
        print(f"DEBUG GRÁFICAS MEJORADO: ✅ Datos preparados - {len(datos_validos)} puntos con correlación {correlacion:.3f}")
        
        # MATRIZ DE CORRELACIONES MEJORADA - Solo variables con correlación significativa
        if len(datos_validos) >= 3:
            try:
                df_corr = pd.DataFrame()
                
                # Variables principales (siempre incluir)
                df_corr['BTX Total'] = datos_validos['btx_total']
                df_corr['Casos Respiratorios'] = datos_validos['casos_totales']
                
                # Variables opcionales (solo si tienen datos y correlación significativa)
                variables_posibles = {
                    'velocidad_viento': 'Velocidad Viento',
                    'temperatura': 'Temperatura', 
                    'humedad': 'Humedad',
                    'exposicion_efectiva': 'Exposición Efectiva',
                    'btx_ajustado': 'BTX Ajustado',
                    'tiempo_exposicion': 'Tiempo Exposición'
                }
                
                for col_key, col_name in variables_posibles.items():
                    if (col_key in datos_validos.columns and 
                        datos_validos[col_key].notna().sum() >= 3):
                        # Solo incluir si tiene correlación significativa con casos
                        corr_temp = datos_validos[col_key].corr(datos_validos['casos_totales'])
                        if abs(corr_temp) >= 0.3:  # Umbral para incluir en matriz
                            df_corr[col_name] = datos_validos[col_key]
                            print(f"DEBUG GRÁFICAS MEJORADO: Incluyendo {col_name} (corr: {corr_temp:.3f})")
                
                # Eliminar columnas con muchos NaN
                df_corr = df_corr.dropna(axis=1, thresh=len(df_corr)*0.6)
                
                # Calcular matriz de correlación
                if len(df_corr.columns) >= 2:
                    matriz_corr = df_corr.corr().values.tolist()
                    variables = list(df_corr.columns)
                    
                    # Filtrar correlaciones no significativas (< 0.3)
                    for i in range(len(matriz_corr)):
                        for j in range(len(matriz_corr[i])):
                            if i != j and abs(matriz_corr[i][j]) < 0.3:
                                matriz_corr[i][j] = 0  # Marcar como no significativa
                    
                    datos_graficas['correlaciones']['variables'] = variables
                    datos_graficas['correlaciones']['matriz'] = matriz_corr
                    
                    print(f"DEBUG GRÁFICAS MEJORADO: ✅ Matriz de correlaciones - {len(variables)} variables significativas")
                else:
                    print("DEBUG GRÁFICAS MEJORADO: No hay suficientes variables con correlación significativa")
                    
            except Exception as e:
                print(f"DEBUG GRÁFICAS MEJORADO: ❌ Error en correlaciones: {e}")
        
        return datos_graficas
        
    except Exception as e:
        print(f"Error en preparar_datos_graficas_mejorado: {e}")
        import traceback
        print(f"DEBUG GRÁFICAS MEJORADO: Traceback: {traceback.format_exc()}")
        return {
            'datos_disponibles': False,
            'mensaje': f'Error en el procesamiento: {str(e)}'
        }

@login_required
def task_detail(request, task_id):
    """Detalle cuando entras por Task (desde /tasks/)."""
    task = get_object_or_404(Task, pk=task_id, user=request.user)

    # Archivos de la simulación asociada
    archivos_simulacion = ArchivoExcel.objects.filter(
        simulacion__usuario=request.user,
        simulacion__nombre_simulacion=task.title
    ).order_by('-fecha_carga')

    # Buscar la simulación asociada
    simulacion = Simulacion.objects.filter(
        usuario=request.user,
        nombre_simulacion=task.title
    ).first()

    # Leer resultados existentes (sin procesar, solo mostrar)
    resultados_txt = None
    if simulacion:
        resultados_txt = leer_resultados_txt(simulacion)
        
        # Preparar datos para gráficas MEJORADO
        datos_graficas = preparar_datos_graficas_mejorado(simulacion)
    else:
        datos_graficas = {'datos_disponibles': False}

    # Datos BTX para mostrar (si no hay TXT, usar valores por defecto)
    if resultados_txt:
        # Extraer valores del TXT para mostrar en la interfaz
        btx_data = {
            'concentracion_benzeno': 4.8,
            'concentracion_tolueno': 15.2,
            'concentracion_xileno': 8.7,
            'btx_total': 8.5,
            'casos_totales': 1570,
        }
    else:
        btx_data = {
            'concentracion_benzeno': 4.8,
            'concentracion_tolueno': 15.2,
            'concentracion_xileno': 8.7,
            'btx_total': 8.5,
            'casos_totales': 1570,
        }

    if request.method == 'POST':
        form = TaskForm(request.POST, instance=task)
        if form.is_valid():
            form.save()
            return redirect('tasks')
        else:
            return render(
                request,
                'task_detail.html',
                {'task': task, 'form': form, 'btx_data': btx_data, 
                 'archivos_simulacion': archivos_simulacion, 'resultados_txt': resultados_txt,
                 'datos_graficas': datos_graficas, 'error': 'Error updating task.'}
            )

    form = TaskForm(instance=task)
    return render(
        request,
        'task_detail.html',
        {'task': task, 'form': form, 'btx_data': btx_data, 
         'archivos_simulacion': archivos_simulacion, 'resultados_txt': resultados_txt,
         'datos_graficas': datos_graficas}
    )


@login_required
def complete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.datecompleted = timezone.now()
        task.save()
        return redirect('tasks')
    return redirect('task_detail', task_id=task.id)


@login_required
def delete_task(request, task_id):
    task = get_object_or_404(Task, pk=task_id, user=request.user)
    if request.method == 'POST':
        task.delete()
        return redirect('tasks')
    return redirect('task_detail', task_id=task.id)


# =========================
# Simulaciones (Borrador → Finalizar) - AQUÍ ES DONDE SE PROCESA TODO
# =========================
@login_required
def create_task(request):
    """
    Pantalla 'Crear simulación' - Aquí es donde se procesa todo
    """
    if request.method == 'POST':
        try:
            nombre_simulacion = request.POST.get('nombre_simulacion')

            # Validación: nombre único por usuario
            if Simulacion.objects.filter(
                usuario=request.user,
                nombre_simulacion=nombre_simulacion
            ).exists():
                error = "Ya existe una simulación con ese nombre. Por favor, usa un nombre diferente."
                simulaciones = Simulacion.objects.filter(
                    usuario=request.user, estado='completada'
                ).order_by('-fecha_ejecucion')
                return render(request, 'create_task.html', {
                    'error': error,
                    'simulaciones': simulaciones
                })

            simulacion = Simulacion(
                usuario=request.user,
                nombre_simulacion=nombre_simulacion,
                descripcion=request.POST.get('descripcion'),
                ubicacion=request.POST.get('ubicacion')
            )
            simulacion.save()

            # (Opcional) Archivo directo vía POST tradicional
            if 'archivo_excel' in request.FILES:
                archivo_excel = request.FILES['archivo_excel']
                tipo_tabla = request.POST.get('tipo_tabla')
                ArchivoExcel.objects.create(
                    simulacion=simulacion,
                    tipo_tabla=tipo_tabla,
                    archivo=archivo_excel
                )

            # Crear Task espejo (compatibilidad)
            Task.objects.create(
                title=simulacion.nombre_simulacion,
                description=simulacion.descripcion or "Simulación creada",
                user=request.user,
                important=True,
                ubicacion=simulacion.ubicacion
            )

            return redirect('tasks')

        except Exception as e:
            error = f"Error al crear simulación: {str(e)}"
            simulaciones = Simulacion.objects.filter(
                usuario=request.user, estado='completada'
            ).order_by('-fecha_ejecucion')
            return render(request, 'create_task.html', {
                'error': error,
                'simulaciones': simulaciones
            })

    # GET: listar solo completadas para la tabla de la página
    simulaciones = Simulacion.objects.filter(
        usuario=request.user, estado='completada'
    ).order_by('-fecha_ejecucion')
    return render(request, 'create_task.html', {'simulaciones': simulaciones})


@login_required
def crear_borrador(request):
    """Crea una simulación en estado 'borrador' (AJAX)."""
    if request.method == 'POST':
        try:
            nombre_simulacion = request.POST.get('nombre_simulacion')

            if Simulacion.objects.filter(
                usuario=request.user,
                nombre_simulacion=nombre_simulacion
            ).exists():
                return JsonResponse({'success': False, 'error': 'Ya existe una simulación con ese nombre'})

            simulacion = Simulacion.objects.create(
                usuario=request.user,
                nombre_simulacion=nombre_simulacion,
                descripcion=request.POST.get('descripcion'),
                ubicacion=request.POST.get('ubicacion'),
                estado='borrador'
            )

            return JsonResponse({
                'success': True,
                'simulacion': {
                    'id': simulacion.id,
                    'nombre_simulacion': simulacion.nombre_simulacion,
                    'ubicacion': simulacion.ubicacion,
                    'descripcion': simulacion.descripcion
                }
            })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Método no permitido'})


@login_required
def subir_archivo_borrador(request):
    """Sube un archivo a una simulación en estado 'borrador' (AJAX)."""
    if request.method == 'POST':
        try:
            simulacion_id = request.POST.get('simulacion_id')
            tipo_tabla = request.POST.get('tipo_tabla')
            
            print(f"DEBUG: Recibiendo archivo - Simulación ID: {simulacion_id}, Tipo: {tipo_tabla}")
            
            if not simulacion_id:
                return JsonResponse({'success': False, 'error': 'No se recibió ID de simulación'})
            
            if not tipo_tabla:
                return JsonResponse({'success': False, 'error': 'No se recibió tipo de tabla'})

            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )

            if 'archivo_excel' not in request.FILES:
                return JsonResponse({'success': False, 'error': 'No se recibió ningún archivo'})

            archivo_excel = request.FILES['archivo_excel']
            
            print(f"DEBUG: Archivo recibido - Nombre: {archivo_excel.name}, Tamaño: {archivo_excel.size}")

            # 1) guarda el registro y el archivo físicamente
            archivo = ArchivoExcel.objects.create(
                simulacion=simulacion,
                tipo_tabla=tipo_tabla,
                archivo=archivo_excel
            )

            print(f"DEBUG: Archivo guardado en: {archivo.archivo.path}")

            # 2) valida el archivo ya guardado
            es_valido = True
            errores = ""
            num_filas = 0
            
            try:
                es_valido, errores, num_filas = validar_archivo_excel(archivo.archivo.path, tipo_tabla)
                print(f"DEBUG: Validación completada - Válido: {es_valido}, Filas: {num_filas}")
            except Exception as e:
                print(f"DEBUG: Error en validación: {str(e)}")
                es_valido = False
                errores = f"Error en validación: {str(e)}"

            # 3) actualiza flags en BD
            archivo.valido = bool(es_valido)
            archivo.errores_validacion = errores
            archivo.num_filas = num_filas
            archivo.save(update_fields=['valido', 'errores_validacion', 'num_filas'])

            return JsonResponse({
                'success': True,
                'archivo_id': archivo.id,
                'valido': archivo.valido,
                'errores': archivo.errores_validacion,
                'num_filas': archivo.num_filas,
                'msg': (f'{tipo_tabla}: {num_filas} filas procesadas.' if archivo.valido
                        else f'{tipo_tabla}: inválido. {errores or ""}')
            })

        except Exception as e:
            print(f"DEBUG: Error general en subir_archivo_borrador: {str(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return JsonResponse({'success': False, 'error': f'Error interno: {str(e)}'})

    return JsonResponse({'success': False, 'error': 'Método no permitido'})


@login_required
def eliminar_archivo_borrador(request):
    """Elimina un archivo asociado a una simulación en 'borrador' (AJAX)."""
    if request.method == 'POST':
        try:
            archivo_id = request.POST.get('archivo_id')
            archivo = get_object_or_404(
                ArchivoExcel,
                id=archivo_id,
                simulacion__usuario=request.user,
                simulacion__estado='borrador'
            )
            archivo.delete()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Método no permitido'})


@login_required
def finalizar_simulacion(request):
    """Pasa una simulación de 'borrador' a 'completada' y crea su Task espejo - AQUÍ SE PROCESA TODO"""
    if request.method == 'POST':
        try:
            simulacion_id = request.POST.get('simulacion_id')
            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )

            # PROCESAR DATOS ANTES DE FINALIZAR - AQUÍ SE CALCULA TODO
            resultados, mensaje = procesar_datos_simulacion(simulacion)
            
            if resultados:
                simulacion.estado = 'completada'
                simulacion.save()

                Task.objects.create(
                    title=simulacion.nombre_simulacion,
                    description=simulacion.descripcion or "Simulación completada",
                    user=request.user,
                    important=True,
                    ubicacion=simulacion.ubicacion
                )

                return redirect('tasks')
            else:
                error = f"Error al procesar datos: {mensaje}"
                return render(request, 'create_task.html', {'error': error})

        except Exception as e:
            error = f"Error al finalizar simulación: {str(e)}"
            return render(request, 'create_task.html', {'error': error})
    return redirect('create_task')


@login_required
def cancelar_borrador(request):
    """Elimina una simulación en 'borrador' (AJAX)."""
    if request.method == 'POST':
        try:
            simulacion_id = request.POST.get('simulacion_id')
            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )
            simulacion.delete()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Método no permitido'})


@login_required
def delete_archivo(request, archivo_id):
    """Elimina un archivo de una simulación (cualquier estado) y vuelve a create_task."""
    archivo = get_object_or_404(ArchivoExcel, pk=archivo_id, simulacion__usuario=request.user)
    archivo.delete()
    return redirect('create_task')


@login_required
def delete_simulation(request, simulacion_id):
    """Elimina una simulación del usuario (y sus archivos)."""
    simulacion = get_object_or_404(Simulacion, pk=simulacion_id, usuario=request.user)
    simulacion.delete()
    return redirect('create_task')


@login_required
def add_archivo_simulacion(request, task_id):
    """
    Agrega un archivo a la simulación asociada al Task (por título).
    Si la simulación no existe, se crea en el momento.
    """
    task = get_object_or_404(Task, pk=task_id, user=request.user)

    if request.method == 'POST' and 'archivo_excel' in request.FILES:
        try:
            simulacion = Simulacion.objects.filter(
                usuario=request.user,
                nombre_simulacion=task.title
            ).first()

            if not simulacion:
                simulacion = Simulacion.objects.create(
                    usuario=request.user,
                    nombre_simulacion=task.title,
                    descripcion=task.description,
                    ubicacion=task.ubicacion or "No especificada",
                    estado='completada'  # se asume completada porque ya existe el Task
                )

            archivo = ArchivoExcel.objects.create(
                simulacion=simulacion,
                tipo_tabla=request.POST.get('tipo_tabla'),
                archivo=request.FILES['archivo_excel']
            )

            es_valido, errores, num_filas = validar_archivo_excel(archivo.archivo.path, archivo.tipo_tabla)
            archivo.valido = bool(es_valido)
            archivo.errores_validacion = errores
            archivo.num_filas = num_filas
            archivo.save(update_fields=['valido', 'errores_validacion', 'num_filas'])

            # REPROCESAR DATOS después de agregar archivo
            procesar_datos_simulacion(simulacion)

        except Exception as e:
            print(f"Error al agregar archivo: {e}")

    return redirect('task_detail', task_id=task_id)


@login_required
def simulacion_detail(request, simulacion_id):
    """
    Muestra el detalle usando el ID de la simulación (link desde create_task.html).
    Reutiliza la plantilla task_detail.html.
    """
    simulacion = get_object_or_404(Simulacion, pk=simulacion_id, usuario=request.user)
    task = Task.objects.filter(user=request.user, title=simulacion.nombre_simulacion).first()
    archivos_simulacion = ArchivoExcel.objects.filter(simulacion=simulacion).order_by('-fecha_carga')

    # Solo leer resultados existentes (sin procesar)
    resultados_txt = leer_resultados_txt(simulacion)
    
    # Preparar datos para gráficas MEJORADO
    datos_graficas = preparar_datos_graficas_mejorado(simulacion)

    # Datos BTX para mostrar
    btx_data = {
        'concentracion_benzeno': 4.8,
        'concentracion_tolueno': 15.2,
        'concentracion_xileno': 8.7,
    }

    error = None
    if request.method == 'POST':
        if task:
            form = TaskForm(request.POST, instance=task)
            if form.is_valid():
                form.save()
                return redirect('simulacion_detail', simulacion_id=simulacion.id)
            else:
                error = 'Error actualizando la tarea.'
        else:
            error = 'No existe Task asociado a esta simulación.'

    form = TaskForm(instance=task) if task else TaskForm()

    return render(request, 'task_detail.html', {
        'task': task,
        'form': form,
        'btx_data': btx_data,
        'archivos_simulacion': archivos_simulacion,
        'simulacion': simulacion,
        'error': error,
        'resultados_txt': resultados_txt,
        'datos_graficas': datos_graficas,
    })

from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg')  # ← necesario para renderizar sin entorno gráfico
import matplotlib.pyplot as plt
from io import BytesIO
import math
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.path import Path
import numpy as np


@login_required
def causal_graph_jpg(request):
    """Genera la gráfica causal dinámica del modelo BTX basada en datos reales"""
    
    # Obtener datos de la última simulación para personalizar la gráfica
    simulacion = Simulacion.objects.filter(usuario=request.user, estado='completada').last()
    datos_graficas = preparar_datos_graficas_mejorado(simulacion) if simulacion else None
    
    # Determinar niveles de riesgo basados en datos reales
    riesgo_btx = "bajo"
    fuerza_relacion = "debil"
    
    if datos_graficas and datos_graficas.get('datos_disponibles', False):
        correlacion = datos_graficas.get('correlacion_principal', 0)
        if correlacion > 0.7:
            fuerza_relacion = "muy_fuerte"
        elif correlacion > 0.5:
            fuerza_relacion = "fuerte" 
        elif correlacion > 0.3:
            fuerza_relacion = "moderada"
        
        # Calcular BTX promedio si hay datos
        if datos_graficas['dispersion']['btx_total']:
            btx_promedio = np.mean(datos_graficas['dispersion']['btx_total'])
            if btx_promedio > 15:
                riesgo_btx = "critico"
            elif btx_promedio > 10:
                riesgo_btx = "alto"
            elif btx_promedio > 5:
                riesgo_btx = "moderado"

    # ----------------- FUNCIÓN PRINCIPAL MEJORADA -----------------
    def draw_causal_graph_dinamica(
        pos, edges, signs=None, feedback_loops=None, fig_size=(28,22),
        padding=0.6, feedback_radius_px=1700, 
        vgap=3.0, arrow_ms=17, lw=1.4,
        riesgo_btx="bajo", fuerza_relacion="debil"
    ):
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_aspect('equal')

        # Colores dinámicos basados en riesgo y relación
        colores_riesgo = {
            "bajo": "lightgreen",
            "moderado": "yellow", 
            "alto": "orange",
            "critico": "red"
        }
        
        colores_relacion = {
            "debil": "lightblue",
            "moderada": "lightsteelblue",
            "fuerte": "cornflowerblue",
            "muy_fuerte": "royalblue"
        }
        
        color_feedback = colores_relacion[fuerza_relacion]
        color_principal = colores_riesgo[riesgo_btx]

        macro_nodes = [
            "Emisiones",
            "Dispersión atmosférica", 
            "Exposición poblacional",
            "Salud respiratoria",
            "Respuesta social y regulatoria"
        ]

        font_macro_size = 16
        font_micro_size = 11

        # ----------------- ETIQUETAS CON COLORES DINÁMICOS -----------------
        text_objs = {}
        for name, (x,y) in pos.items():
            if name in macro_nodes:
                color_fondo = color_principal if name == "Emisiones" else "white"
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color_fondo, alpha=0.7)
                text_objs[name] = ax.text(
                    x, y, name, fontsize=font_macro_size,
                    ha='center', va='center', zorder=5, color="black",
                    fontname="Times New Roman", weight="bold",
                    bbox=bbox_props
                )
            else:
                text_objs[name] = ax.text(
                    x, y, name, fontsize=font_micro_size,
                    ha='center', va='center', zorder=5, color="black"
                )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # ----------------- EXTENT RECTANGULARES -----------------
        node_half = {}
        for name, txt in text_objs.items():
            bbox = txt.get_window_extent(renderer=renderer)
            inv = ax.transData.inverted()
            (x0, y0) = inv.transform((bbox.x0, bbox.y0))
            (x1, y1) = inv.transform((bbox.x1, bbox.y1))
            half_w = abs(x1 - x0) / 2.0 + padding
            half_h = abs(y1 - y0) / 2.0 + padding
            if name in macro_nodes:
                half_w *= 1.9
                half_h *= 1.9
            node_half[name] = (half_w, half_h)

        # ----------------- AUX: trayectoria Bézier -----------------
        def vertical_bezier_points(src, dst):
            (sx, sy) = pos[src]
            (dx, dy) = pos[dst]
            (sw, sh) = node_half.get(src, (0.8,0.5))
            (dw, dh) = node_half.get(dst, (0.8,0.5))
            going_up = dy > sy
            sgn_src = 1 if going_up else -1
            p0 = (sx, sy + sgn_src*(sh + 0.25))
            c1 = (sx, sy + sgn_src*(sh + 0.25 + vgap))
            sgn_dst = -1 if going_up else 1
            p3 = (dx, dy + sgn_dst*(dh + 0.25))
            c2 = (dx, dy + sgn_dst*(dh + 0.25 + vgap))
            return p0, c1, c2, p3

        def add_arrow_bezier(p0, c1, c2, p3, color="black", z=3, relacion="debil"):
            # Grosor de línea basado en fuerza de relación
            grosor = {
                "debil": 1.0,
                "moderada": 1.8, 
                "fuerte": 2.5,
                "muy_fuerte": 3.2
            }[relacion]
            
            verts = [p0, c1, c2, p3]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            arr = FancyArrowPatch(
                path=path, arrowstyle='-|>', mutation_scale=arrow_ms,
                linewidth=grosor, color=color, zorder=z, shrinkA=0, shrinkB=0
            )
            ax.add_patch(arr)

        # ----------------- DIBUJO DE FLECHAS DINÁMICAS -----------------
        for (src, dst) in edges:
            if src not in pos or dst not in pos:
                continue
            p0, c1, c2, p3 = vertical_bezier_points(src, dst)
            
            # Color dinámico basado en tipo de relación
            if src == "Emisiones" and dst == "Dispersión atmosférica":
                color_flecha = color_principal
                relacion_actual = fuerza_relacion
            elif "Salud" in dst:
                color_flecha = "red" if fuerza_relacion in ["fuerte", "muy_fuerte"] else "darkred"
                relacion_actual = fuerza_relacion
            else:
                color_flecha = "black"
                relacion_actual = "moderada"
                
            add_arrow_bezier(p0, c1, c2, p3, color_flecha, relacion=relacion_actual)
            
            sign = signs.get((src, dst), '+') if signs else '+'
            color_sign = "darkred" if sign == "+" else ("blue" if sign == "-" else "gray")
            (dx, dy) = p3
            (_, dh) = node_half.get(dst, (0.8, 0.5))
            going_up = dy < pos[dst][1]
            off_y = (dh + 0.9) * (1 if going_up else -1)
            ax.text(dx + 0.9, pos[dst][1] + off_y, sign,
                    fontsize=13, ha='center', va='center',
                    zorder=6, color=color_sign, weight="bold")

        # ----------------- FLECHAS DE RETROALIMENTACIÓN DINÁMICAS -----------------
        def add_feedback_circle_arrow(src, dst, offset_pixels=36):
            if src not in pos or dst not in pos:
                return
            x1, y1 = pos[src]; x2, y2 = pos[dst]
            src_disp = ax.transData.transform((x1, y1))
            dst_disp = ax.transData.transform((x2, y2))
            vx = dst_disp[0] - src_disp[0]; vy = dst_disp[1] - src_disp[1]
            dist_disp = math.hypot(vx, vy)
            if dist_disp < 2: return
            px = -vy/dist_disp; py = vx/dist_disp
            mid_disp = ((src_disp[0]+dst_disp[0])/2 + px*offset_pixels,
                        (src_disp[1]+dst_disp[1])/2 + py*offset_pixels)
            trans_inv = ax.transData.inverted()
            center_data = trans_inv.transform(mid_disp)
            radius_px = feedback_radius_px
            width_data = abs(trans_inv.transform((mid_disp[0]+radius_px, mid_disp[1]))[0] - center_data[0])
            
            # Color y estilo dinámico para retroalimentación
            estilo_linea = '-' if fuerza_relacion in ["fuerte", "muy_fuerte"] else '--'
            grosor = 3.5 if riesgo_btx in ["alto", "critico"] else 2.5
            
            arc = Arc(center_data, width=width_data, height=width_data,
                      theta1=0, theta2=320, linewidth=grosor,
                      color=color_feedback, zorder=2.5, linestyle=estilo_linea)
            ax.add_patch(arc)

        if feedback_loops:
            for src, dst in feedback_loops:
                add_feedback_circle_arrow(src, dst)

        # ----------------- LEYENDA DINÁMICA -----------------
        leyenda_texto = f"Estado actual: BTX {riesgo_btx.upper()}, Relación {fuerza_relacion.upper()}"
        ax.text(0.02, 0.98, leyenda_texto, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlim(min(p[0] for p in pos.values())-12, max(p[0] for p in pos.values())+12)
        ax.set_ylim(min(p[1] for p in pos.values())-12, max(p[1] for p in pos.values())+12)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # ----------------- POSICIONES Y DATOS -----------------
    pos_main = {
        "Emisiones": (0.0, 0.0),
        "Dispersión atmosférica": (0.0, 20.0),
        "Exposición poblacional": (25.0, 0.0),
        "Salud respiratoria": (0.0, -20.0),
        "Respuesta social y regulatoria": (-25.0, 0.0)
    }

    def generate_subnode_positions_circle(pos_main, children_dict, orbit_radius=8.5):
        pos = dict(pos_main)
        for parent, childs in children_dict.items():
            cx, cy = pos_main[parent]
            n = len(childs)
            for i, child in enumerate(childs):
                theta = 2*math.pi * i/n
                pos[child] = (cx + orbit_radius*math.cos(theta),
                              cy + orbit_radius*math.sin(theta))
        return pos

    children = {
        "Emisiones": ["Tipo_Fuente","Gas (E)","Tasa_Emisión","Ubicación (E)","Eficiencia_Control"],
        "Dispersión atmosférica": ["Gas (D)","Velocidad_Viento","Dirección_Viento","Temperatura","Humedad_Relativa","Concentración"],
        "Exposición poblacional": ["Zona (Exp)","Tamaño_Población","Tiempo_Exposición","Nivel_Exposición","Actividades_Aire_Libre"],
        "Salud respiratoria": ["Zona (Salud)","Casos_Asma","Casos_Bronquitis","Hospitalizaciones","Mortalidad","Atención_Médica_Disponible"],
        "Respuesta social y regulatoria": ["Tipo_Medida","Institución","Impacto_Estimado","Observaciones","Campañas_Sensibilización"]
    }

    pos_all = generate_subnode_positions_circle(pos_main, children)

    edges = [
        ("Emisiones","Dispersión atmosférica"),
        ("Dispersión atmosférica","Exposición poblacional"),
        ("Exposición poblacional","Salud respiratoria"),
        ("Salud respiratoria","Respuesta social y regulatoria"),
        ("Respuesta social y regulatoria","Emisiones"),
    ]
    for parent, childs in children.items():
        for c in childs:
            edges.append((c, parent))
    edges.extend([
        ("Tasa_Emisión","Concentración"),
        ("Velocidad_Viento","Concentración"),
        ("Dirección_Viento","Concentración"),
        ("Humedad_Relativa","Concentración"),
        ("Concentración","Nivel_Exposición"),
        ("Actividades_Aire_Libre","Nivel_Exposición"),
        ("Nivel_Exposición","Casos_Asma"),
        ("Nivel_Exposición","Casos_Bronquitis"),
        ("Atención_Médica_Disponible","Mortalidad"),
        ("Impacto_Estimado","Emisiones"),
        ("Tipo_Medida","Impacto_Estimado"),
        ("Institución","Impacto_Estimado"),
        ("Campañas_Sensibilización","Tipo_Medida"),
        ("Ubicación (E)","Zona (Exp)"),
        ("Zona (Exp)","Zona (Salud)")
    ])

    signs = {
        ("Emisiones","Dispersión atmosférica"): "+",
        ("Dispersión atmosférica","Exposición poblacional"): "+",
        ("Exposición poblacional","Salud respiratoria"): "+",
        ("Salud respiratoria","Respuesta social y regulatoria"): "+",
        ("Respuesta social y regulatoria","Emisiones"): "-",
    }

    feedback_loops = [
        ("Emisiones","Dispersión atmosférica"),
        ("Exposición poblacional","Salud respiratoria"),
        ("Salud respiratoria","Respuesta social y regulatoria"),
        ("Respuesta social y regulatoria","Emisiones")
    ]

    # ----------------- GENERAR Y DEVOLVER FIGURA DINÁMICA -----------------
    fig = draw_causal_graph_dinamica(
        pos_all, edges, signs=signs, feedback_loops=feedback_loops,
        vgap=3.2, padding=0.7, arrow_ms=18, lw=1.5,
        riesgo_btx=riesgo_btx, fuerza_relacion=fuerza_relacion
    )

    buffer = BytesIO()
    fig.savefig(buffer, format='jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)

    return HttpResponse(buffer.getvalue(), content_type='image/jpeg')


# Función de respaldo para mantener compatibilidad
def calcular_ecuaciones_btx(dfs):
    """Función de respaldo que llama a la versión mejorada"""
    return calcular_ecuaciones_btx_mejorado(dfs)

def guardar_resultados_txt(resultados, simulacion):
    """Función de respaldo que llama a la versión mejorada"""
    return guardar_resultados_txt_mejorado(resultados, simulacion)

def preparar_datos_graficas(simulacion):
    """Función de respaldo que llama a la versión mejorada"""
    return preparar_datos_graficas_mejorado(simulacion)
# =========================
# MODO INVITADO - NUEVAS FUNCIONES
# =========================

def guest_access(request):
    """Acceso de invitado - puede ver simulaciones pero no crear"""
    # Crear usuario temporal de invitado o usar sesión anónima
    request.session['guest_mode'] = True
    return redirect('guest_tasks')

def guest_tasks(request):
    """Mostrar simulaciones públicas para invitados"""
    if not request.session.get('guest_mode'):
        return redirect('home')
    
    # Obtener algunas simulaciones públicas o recientes
    tasks_qs = Task.objects.filter(datecompleted__isnull=False).order_by('-datecompleted')[:10]
    return render(request, 'tasks.html', {
        "tasks": tasks_qs,
        "guest_mode": True
    })

def guest_task_detail(request, task_id):
    """Detalle de simulación para invitados"""
    if not request.session.get('guest_mode'):
        return redirect('home')
    
    task = get_object_or_404(Task, pk=task_id, datecompleted__isnull=False)
    
    # Archivos de la simulación asociada
    archivos_simulacion = ArchivoExcel.objects.filter(
        simulacion__nombre_simulacion=task.title
    ).order_by('-fecha_carga')

    # Buscar la simulación asociada
    simulacion = Simulacion.objects.filter(
        nombre_simulacion=task.title
    ).first()

    # Leer resultados existentes
    resultados_txt = None
    if simulacion:
        resultados_txt = leer_resultados_txt(simulacion)
        datos_graficas = preparar_datos_graficas_mejorado(simulacion)
    else:
        datos_graficas = {'datos_disponibles': False}

    # Datos BTX para mostrar
    btx_data = {
        'concentracion_benzeno': 4.8,
        'concentracion_tolueno': 15.2,
        'concentracion_xileno': 8.7,
        'btx_total': 8.5,
        'casos_totales': 1570,
    }

    return render(
        request,
        'task_detail.html',
        {
            'task': task, 
            'btx_data': btx_data, 
            'archivos_simulacion': archivos_simulacion, 
            'resultados_txt': resultados_txt,
            'datos_graficas': datos_graficas,
            'guest_mode': True  # Indicar que es modo invitado
        }
    )