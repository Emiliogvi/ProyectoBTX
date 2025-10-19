from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import os

# -------------------------
# 1. Tareas (opcional)
# -------------------------
class Task(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(max_length=1000)
    created = models.DateTimeField(auto_now_add=True)
    datecompleted = models.DateTimeField(null=True, blank=True)
    important = models.BooleanField(default=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    ubicacion = models.CharField(max_length=150, blank=True, null=True)
    archivos_procesados = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"{self.title} - {self.user.username}"


# -------------------------
# 2. Perfil de Usuario (extiende User)
# -------------------------
class PerfilUsuario(models.Model):
    ROLES = [
        ('Administrador', 'Administrador'),
        ('Avanzado', 'Avanzado'),
        ('Observador', 'Observador'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    rol = models.CharField(max_length=20, choices=ROLES)
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.rol})"


# -------------------------
# 3. Simulaciones
# -------------------------
class Simulacion(models.Model):
    ESTADOS = [
        ('borrador', 'Borrador'),
        ('completada', 'Completada'),
    ]
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    nombre_simulacion = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    fecha_ejecucion = models.DateTimeField(auto_now_add=True)
    ubicacion = models.CharField(max_length=255, blank=True, null=True)
    datos_txt = models.FileField(
        upload_to='simulaciones_txt/',
        blank=True,
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=['txt'])]
    )
    estado = models.CharField(max_length=20, choices=ESTADOS, default='borrador')  # ← NUEVO CAMPO

    def __str__(self):
        return f"{self.nombre_simulacion} - {self.usuario.username} ({self.estado})"

class ArchivoExcel(models.Model):
    TIPOS = [
        ('Emisiones', 'Emisiones'),
        ('Dispersión', 'Dispersión'),
        ('Exposición', 'Exposición'),
        ('Salud', 'Salud'),
        ('Social', 'Social'),
    ]
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    tipo_tabla = models.CharField(max_length=20, choices=TIPOS)
    archivo = models.FileField(
        upload_to='archivos_excel/%Y/%m/%d/',
        validators=[FileExtensionValidator(allowed_extensions=['xls', 'xlsx', 'csv'])]
    )
    fecha_carga = models.DateTimeField(auto_now_add=True)

    # 👇 NUEVOS CAMPOS
    valido = models.BooleanField(default=False)
    errores_validacion = models.TextField(blank=True, null=True)
    num_filas = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"{self.tipo_tabla} ({self.simulacion.nombre_simulacion})"



# -------------------------
# 5. Variables de Submodelos
# -------------------------
class VariablesEmisiones(models.Model):
    simulacion = models.OneToOneField(Simulacion, on_delete=models.CASCADE)
    parque_vehicular = models.IntegerField()
    inventario_industrial = models.IntegerField()
    factor_emision = models.FloatField()
    kilometraje_promedio = models.FloatField()
    consumo_combustible = models.FloatField()
    eficiencia_control = models.FloatField()
    politica = models.CharField(max_length=100)

class VariablesDispersion(models.Model):
    simulacion = models.OneToOneField(Simulacion, on_delete=models.CASCADE)
    concentracion_BTX = models.FloatField()
    velocidad_viento = models.FloatField()
    direccion_viento = models.CharField(max_length=20)
    altura_mezcla = models.FloatField()
    radiacion_solar = models.FloatField()
    vida_media_BTX = models.FloatField()

class VariablesExposicion(models.Model):
    simulacion = models.OneToOneField(Simulacion, on_delete=models.CASCADE)
    poblacion_zona = models.IntegerField()
    tiempo_exposicion = models.FloatField()
    infiltracion = models.FloatField()
    microambiente = models.CharField(max_length=100)
    uso_proteccion = models.BooleanField(default=False)

class VariablesSalud(models.Model):
    simulacion = models.OneToOneField(Simulacion, on_delete=models.CASCADE)
    casos_respiratorios = models.IntegerField()
    hospitalizaciones = models.IntegerField()
    urgencias = models.IntegerField()
    incidencia_base = models.FloatField()
    coeficiente_beta = models.FloatField()
    lag_dias = models.IntegerField()

class VariablesSocial(models.Model):
    simulacion = models.OneToOneField(Simulacion, on_delete=models.CASCADE)
    presion_publica = models.FloatField()
    inversion_tp = models.FloatField()
    capacidad_vigilancia = models.FloatField()
    adopcion_tecnologica = models.FloatField()
    cumplimiento_normativo = models.FloatField()


# -------------------------
# 6. Datos Originales (de los Excels)
# -------------------------
class EmisionesOriginal(models.Model):
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    tipo_fuente = models.CharField(max_length=100)
    gas = models.CharField(max_length=50)
    tasa_emision = models.FloatField()
    ubicacion = models.CharField(max_length=150)
    fecha_registro = models.DateTimeField()

class DispersionOriginal(models.Model):
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    gas = models.CharField(max_length=50)
    velocidad_viento = models.FloatField()
    direccion_viento = models.CharField(max_length=20)
    temperatura = models.FloatField()
    concentracion = models.FloatField()
    fecha_registro = models.DateTimeField()

class ExposicionOriginal(models.Model):
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    zona = models.CharField(max_length=100)
    tamaño_poblacion = models.IntegerField()
    tiempo_exposicion = models.FloatField()
    nivel_exposicion = models.FloatField()
    fecha_registro = models.DateTimeField()

class SaludOriginal(models.Model):
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    zona = models.CharField(max_length=100)
    casos_asma = models.IntegerField()
    casos_bronquitis = models.IntegerField()
    hospitalizaciones = models.IntegerField()
    mortalidad = models.IntegerField()
    fecha_registro = models.DateTimeField()

class SocialOriginal(models.Model):
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    tipo_medida = models.CharField(max_length=100)
    institucion = models.CharField(max_length=100)
    fecha_implementacion = models.DateField()
    impacto_estimado = models.CharField(max_length=100)
    observaciones = models.TextField(blank=True, null=True)


# -------------------------
# 7. Resultados (imágenes, PDF, etc.)
# -------------------------
class Resultado(models.Model):
    TIPOS_RESULTADO = [
        ('Diagrama causal', 'Diagrama causal'),
        ('Tornado chart', 'Tornado chart'),
        ('Gráfica de dispersión', 'Gráfica de dispersión'),
        ('Bandas de incertidumbre', 'Bandas de incertidumbre'),
        ('Gráfica de fases', 'Gráfica de fases'),
    ]
    simulacion = models.ForeignKey(Simulacion, on_delete=models.CASCADE)
    tipo_resultado = models.CharField(max_length=50, choices=TIPOS_RESULTADO)
    imagen = models.ImageField(
        upload_to='resultados/',
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])]
    )
    formato = models.CharField(max_length=10, default='JPG')
    fecha_generacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.tipo_resultado} ({self.simulacion.nombre_simulacion})"
