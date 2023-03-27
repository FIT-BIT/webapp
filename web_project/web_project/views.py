from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse


import cv2
import threading
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from  core import AudioCommSys as audio
#from crypt import methods
from requests import Response
from core import ExercisesModule as trainer
# from core.camera import VideoCamera
from core.exercise_counter import bicep_curl_rep, main


def gen():
    level = 1
    cap = cv2.VideoCapture(0)
    for i in main(cap):
        yield i
    # for i in trainer.start_workout_session(level).complete_path("Easy"):
    #     yield i
        # print("hereee ------------------------------------")
    print("DONE!")
    

@gzip.gzip_page
def video_feed(request):
    response = StreamingHttpResponse(gen(), content_type="multipart/x-mixed-replace;boundary=frame")
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
    response = StreamingHttpResponse(generator(Camera()), content_type="multipart/x-mixed-replace;boundary=frame")
    return response

# A generator function that yields video frames as multipart responses
def generator(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def exercise(request):
    # exercise_counter.main()
    return render (request,'exercisepage.html')

