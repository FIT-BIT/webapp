from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse


def home(request):
    return render(request, 'homepage.html')
    # return render( request, 'index.html' )


def workouts(request):
    return render(request, 'workouts.html')


def myworkouts(request):
    return render(request, 'myworkouts.html')
