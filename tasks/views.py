from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
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


@login_required
def task_detail(request, task_id):
    """Detalle cuando entras por Task (desde /tasks/)."""
    task = get_object_or_404(Task, pk=task_id, user=request.user)

    # Archivos de la simulación asociada (nombre del Task == nombre de simulación)
    archivos_simulacion = ArchivoExcel.objects.filter(
        simulacion__usuario=request.user,
        simulacion__nombre_simulacion=task.title
    ).order_by('-fecha_carga')

    # Datos BTX de ejemplo (puedes sustituir por consulta real)
    btx_data = {
        'concentracion_benzeno': 4.8,
        'concentracion_tolueno': 15.2,
        'concentracion_xileno': 8.7,
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
                {'task': task, 'form': form, 'btx_data': btx_data, 'archivos_simulacion': archivos_simulacion,
                 'error': 'Error updating task.'}
            )

    form = TaskForm(instance=task)
    return render(
        request,
        'task_detail.html',
        {'task': task, 'form': form, 'btx_data': btx_data, 'archivos_simulacion': archivos_simulacion}
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
# Simulaciones (Borrador → Finalizar)
# =========================
@login_required
def create_task(request):
    """
    Pantalla 'Crear simulación':
      - GET: muestra formulario + lista de simulaciones COMPLETADAS del usuario.
      - POST (opcional): crea simulación + archivo directo (si envías formulario tradicional).
        Nota: El flujo principal de creación de borrador/archivos se hace con AJAX abajo.
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
            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )

            if 'archivo_excel' not in request.FILES:
                return JsonResponse({'success': False, 'error': 'No se recibió ningún archivo'})

            archivo_excel = request.FILES['archivo_excel']
            tipo_tabla = request.POST.get('tipo_tabla')

            archivo = ArchivoExcel.objects.create(
                simulacion=simulacion,
                tipo_tabla=tipo_tabla,
                archivo=archivo_excel
            )

            return JsonResponse({'success': True, 'archivo_id': archivo.id})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
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
    """Pasa una simulación de 'borrador' a 'completada' y crea su Task espejo."""
    if request.method == 'POST':
        try:
            simulacion_id = request.POST.get('simulacion_id')
            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )

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

            ArchivoExcel.objects.create(
                simulacion=simulacion,
                tipo_tabla=request.POST.get('tipo_tabla'),
                archivo=request.FILES['archivo_excel']
            )

        except Exception as e:
            # Log simple por consola; puedes mejorar con logging
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
    })

@login_required
def subir_archivo_borrador(request):
    if request.method == 'POST':
        try:
            simulacion_id = request.POST.get('simulacion_id')
            simulacion = get_object_or_404(
                Simulacion, id=simulacion_id, usuario=request.user, estado='borrador'
            )

            if 'archivo_excel' not in request.FILES:
                return JsonResponse({'success': False, 'error': 'No se recibió ningún archivo'})

            archivo_excel = request.FILES['archivo_excel']
            tipo_tabla = request.POST.get('tipo_tabla')

            # 1) guarda el registro y el archivo físicamente
            archivo = ArchivoExcel.objects.create(
                simulacion=simulacion,
                tipo_tabla=tipo_tabla,
                archivo=archivo_excel
            )

            # 2) valida el archivo ya guardado
            es_valido, errores, num_filas = validar_archivo_excel(archivo.archivo.path, tipo_tabla)

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
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Método no permitido'})

# al inicio del archivo ya importaste: from .utils import validar_archivo_excel

@login_required
def add_archivo_simulacion(request, task_id):
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
                    estado='completada'
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

        except Exception as e:
            print(f"Error al agregar archivo: {e}")

    return redirect('task_detail', task_id=task_id)
