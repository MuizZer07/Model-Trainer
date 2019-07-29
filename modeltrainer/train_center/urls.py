from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload_dataset', views.upload_dataset, name='upload_dataset'),
    path('upload_dataset_request', views.upload_dataset_request, name='upload_dataset_request'),
    path('train_model', views.train_model, name='train_model'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
