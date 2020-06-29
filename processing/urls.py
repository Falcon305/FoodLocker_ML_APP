from django.urls import path

from . import views

urlpatterns = [
    #path('create_order/<str:pk>/', views.createOrder, name="create_order"),
    path('create', views.create_project, name='create_project'),
    path('load_dataset', views.load_dataset, name='load_dataset'),
    path('view_data', views.view_data, name='view_data'),
    path('preprocessing', views.preprocessing, name='preprocessing'),
    path('model_selection', views.model_selection, name='model_selection'),
    path('model_exec', views.model_exec, name='model_exec'),
    path('save_md', views.save_md, name='save_md'),
    path('MLP', views.MLP, name='MLP'),
    path('LSTM_M', views.LSTM_M, name='LSTM_M'),
    path('ARIMA_M', views.ARIMA_M, name='ARIMA_M'),
    path('MLP_GRID', views.MLP_GRID, name='MLP_GRID'),
    path('mgs', views.mgs, name='mgs'),
]
