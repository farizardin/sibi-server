from django.urls import path, include
from .views import (
    SibiRecognition,
)

urlpatterns = [
    path('api', SibiRecognition.as_view()),
]