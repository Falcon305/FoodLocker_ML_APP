from django.urls import path

from . import views

urlpatterns = [
    #path('dashboard', views.dashboard, name='dashboard')
    path('', views.create_project, name='create_project'),
    path('load_dataset', views.load_dataset, name='load_dataset'),
]