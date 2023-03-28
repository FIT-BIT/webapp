from django.urls import path
from . import views

urlpatterns = [
    path('', views.register, name='register'),
    path('logout', views.usrlogout, name='logout'),
    path('workout_routine_create/', views.create_workout_routine, name='create_workout_routine'),
    path('user-profile/', views.profile, name='user_profile'),
    # path('profile', views.profile, name='profile'),
]
