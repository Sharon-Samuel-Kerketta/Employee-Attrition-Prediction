from django.urls import path
from . import views
urlpatterns = [
    path('',views.index,name='index'),
    path('index_table/',views.index_table),
    path('index_table/probability/',views.prediction),
    path('precision/',views.feature_importances),
    path('predict_attr/',views.predict_attr_fn)
]