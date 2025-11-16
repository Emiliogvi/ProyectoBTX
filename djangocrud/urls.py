"""djangocrud URL Configuration"""

from django.contrib import admin
from django.urls import path
from tasks import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('signup/', views.signup, name='signup'),
    path('tasks/', views.tasks, name='tasks'),
    path('tasks_completed/', views.tasks_completed, name='tasks_completed'),
    path('logout/', views.signout, name='logout'),
    path('signin/', views.signin, name='signin'),
    path('create_task/', views.create_task, name='create_task'),
    path('tasks/<int:task_id>', views.task_detail, name='task_detail'),
    path('tasks/<int:task_id>/complete', views.complete_task, name='complete_task'),
    path('tasks/<int:task_id>/delete', views.delete_task, name='delete_task'),
    path('add_archivo/<int:task_id>/', views.add_archivo_simulacion, name='add_archivo_simulacion'),
    path('delete_archivo/<int:archivo_id>/', views.delete_archivo, name='delete_archivo'),
    path('delete_simulation/<int:simulacion_id>/', views.delete_simulation, name='delete_simulation'),
    path('crear_borrador/', views.crear_borrador, name='crear_borrador'),
    path('subir_archivo_borrador/', views.subir_archivo_borrador, name='subir_archivo_borrador'),
    path('eliminar_archivo_borrador/', views.eliminar_archivo_borrador, name='eliminar_archivo_borrador'),
    path('finalizar_simulacion/', views.finalizar_simulacion, name='finalizar_simulacion'),
    path('cancelar_borrador/', views.cancelar_borrador, name='cancelar_borrador'),
    path('simulaciones/<int:simulacion_id>/', views.simulacion_detail, name='simulacion_detail'),
    path('add_archivo_simulacion/<int:task_id>/', views.add_archivo_simulacion, name='add_archivo_simulacion'),
    path('simulacion_detail/<int:simulacion_id>/', views.simulacion_detail, name='simulacion_detail'),

    # Nueva ruta para la gráfica causal
    path('grafica/causal.jpg', views.causal_graph_jpg, name='causal_graph_jpg'),
    path('guest/', views.guest_access, name='guest_access'),
    path('guest/tasks/', views.guest_tasks, name='guest_tasks'),
    path('guest/tasks/<int:task_id>/', views.guest_task_detail, name='guest_task_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

path('grafica/causal.jpg', views.causal_graph_jpg, name='causal_graph_jpg')