from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_next_value, name='predict_next_value'),
]
