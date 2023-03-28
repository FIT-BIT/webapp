from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
import numpy as np

import cv2
import threading
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from core import AudioCommSys as audio
# from crypt import methods
from requests import Response
from core import ExercisesModule as trainer
# from core.camera import VideoCamera
from core.exercise_counter import main as exerciseView
from core.gameControlers import main as gameControls

difficulty_level = ''
set = 0

def gen():
    cap = cv2.VideoCapture(0)
    print(difficulty_level,set)
    for i in exerciseView(cap,difficulty_level,set):
        yield i
    print("DONE!")
    

@gzip.gzip_page
def video_feed(request):
    response = StreamingHttpResponse(
        gen(), content_type="multipart/x-mixed-replace;boundary=frame")
    return response

# camera = VideoCamera()

# # Create a view that streams video frames as a multipart response
# @gzip.gzip_page
# def video_feed_camera(request):
#     response = StreamingHttpResponse(generator(camera), content_type="multipart/x-mixed-replace;boundary=frame")
#     return response

# # A generator function that yields video frames as multipart responses
# def generator(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def home(request):
    print("xxxxxxxxxxxxxx")
    return render(request, 'homepage.html')
    # return render( request, 'index.html' )


def workouts(request):
    return render(request, 'workouts.html')


def myworkouts(request):
    return render(request, 'myworkouts.html')


class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

# Create an instance of the Camera class
# camera = Camera()

# Create a view that streams video frames as a multipart response


@gzip.gzip_page
def video_feed_camera(request):
    response = StreamingHttpResponse(
        generator(Camera()), content_type="multipart/x-mixed-replace;boundary=frame")
    return response

# A generator function that yields video frames as multipart responses


def generator(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def exercise(request):
    # if request.method == 'POST':
    global difficulty_level ,set 
    set = request.GET.get('workout')
    difficulty_level = request.GET.get('difficulty_level')
    print("difficulty_level,set")
    print(difficulty_level,set)
    return render (request,'exercisepage.html')
    # return render(request, 'myworkouts.html')

def myRoutine(request):
    context = {}
    context['workout1'] = ["Walk", "Jumping Jacks","Bicep Curls", "Shoulder Press", "Right Knee Touches", "squats"]
    context['workout2'] = ["Walk", "Jumping Jacks","Crunches", "Push Ups", "Hammer Curl", "Arm Raise", "Sit Ups"]
    return render(request, 'myRoutine.html',context)



def generateGame():
	# grab global references to the output frame and lock variables
    for i in gameControls():
        print(i)
        
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg",i)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            encodedImage.tobytes() + b'\r\n')

 

def game(request):
    # exercise_counter.main()
    return render(request, 'gamepage.html')

@gzip.gzip_page
def video_feed_game(request):
    response = StreamingHttpResponse(
        generateGame(), content_type="multipart/x-mixed-replace;boundary=frame")
    return response
