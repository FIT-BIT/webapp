from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render

def home(request):
    return render( request, 'homepage.html' )
    # return render( request, 'index.html' )

# def contact(request):
#     return render( request, 'contact.html')