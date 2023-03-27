#!/usr/bin/env python
# coding: utf-8

# # Install and Import Dependencies

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np 
import time
import speech_recognition as sr
from gtts import gTTS
import os
from io import BytesIO
from playsound import playsound
from datetime import datetime
# from camera import VideoCamera

# import Audio_Communication_System as audio
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

language = 'en'

# # Global Variables

# In[ ]:


video_capture_number = 0
BAR_WIDTH = 400


# # Calculate Angles

# In[ ]:


#angle between any three points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # a[0] = x a[1]=y
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


# # Performance Bar

# In[ ]:


def draw_performance_bar(image, per, bar):
    cv2.rectangle(image, (580, 50), (600, 380), (0, 255, 0), 3)
    cv2.rectangle(image, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
    cv2.putText(image, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    
def progress_bar(image, progress):
    # Draw the progress bar
    BAR_WIDTH = 400
    BAR_HEIGHT = 15
    BAR_POS = (20, 110)
    cv2.rectangle(image, BAR_POS, (BAR_POS[0] + progress, BAR_POS[1] + BAR_HEIGHT), (0, 255, 0), -1)
    cv2.rectangle(image, BAR_POS, (BAR_POS[0] + BAR_WIDTH, BAR_POS[1] + BAR_HEIGHT), (255, 255, 255), 2)
    
def display_reps(image, angle, reps, stage):
    # Display the bicep angle and number of reps on the image
    cv2.putText(image, "Angle: {:.2f} deg".format(angle), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "Reps: {}".format(reps), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "Stage: {}".format(stage), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def display_picture(exercise):
    image = cv2.imread(r"/home/nuke/Desktop/project/AI-Personal-Trainer-Rep-Counter/{}.jpg".format(exercise))
    cv2.imshow('Exercise Counter',image)
    cv2.waitKey(1000)


# # Adjust Camera position 

# In[ ]:


def adjust_camera_position(cap):

    while True:
        
        position = False
        # Read a frame from the video capture.
        ret, frame = cap.read()

        # Convert the image from BGR (OpenCV's default color space) to RGB (MediaPipe's required color space).
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the pose landmarks of the user using MediaPipe.
        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            results = pose.process(frame)

            # Check if any pose landmarks were detected.
            if results.pose_landmarks is not None:
                # Draw the pose landmarks on the frame.
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get the x, y, and z coordinates of the left shoulder and left hip.
                left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                left_shoulder_z = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
                left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
                left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                left_hip_z = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z
                left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
                left_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
                left_ankle_z = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z
                
                # Calculate the distance between the left shoulder and left hip.
                distance_upper = ((left_shoulder_x - left_hip_x) ** 2 + (left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_z - left_hip_z) ** 2) ** 0.5
                
                # Calculate the distance between the left shoulder and left hip.
                distance_lower = ((left_ankle_x - left_hip_x) ** 2 + (left_ankle_y - left_hip_y) ** 2 + (left_ankle_z - left_hip_z) ** 2) ** 0.5

                # If the distance is smaller than a threshold, then the user is standing correctly.
                if distance_upper < 0.5 and distance_lower < 0.5:
                    cv2.putText(frame, "You're standing correctly!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    position = True
                # Otherwise, instruct the user to adjust the camera position.
                elif distance_upper < 0.5:
                    cv2.putText(frame, "Move backward, lower body not in frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                elif distance_upper < 0.5:
                    cv2.putText(frame, "Move backward, upper body not in frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Adjust camera position until entire body is in frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert the image back from RGB to BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show the frame in a window.
        cv2.imshow("Exercise Counter", frame)

        # Wait for the user to press a key.
        key = cv2.waitKey(1)
        
        if position == True:
            time.sleep(1)
            return
        # If the user presses the 'q' key, then exit the loop.
        if key == ord("q"):
            break

    # Release the video capture and close all windows.
    return


# # Arm Raise

# In[ ]:


def arm_raise_rep(cap, total_reps):
   
    # Initialize variables
    prev_reps = 0
    reps = 0
    start_curl = False
    end_curl = False
    bicep_angle = 0
    previous_bicep_angle = 0
    progress = 0
    stage = "down"
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while reps < total_reps:
            ret, image = cap.read()

            # Convert the image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the shoulder, elbow and wrist landmarks
            if results.pose_landmarks is not None:
                shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                elbow_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

                shoulder = np.array([shoulder_landmark.x, shoulder_landmark.y])
                elbow = np.array([elbow_landmark.x, elbow_landmark.y])
                hip = np.array([hip_landmark.x, hip_landmark.y])

                bicep_angle = calculate_angle(shoulder, elbow, hip)

                # Check if the bicep is being curled
                if bicep_angle < 20 and not start_curl:
                    start_curl = True
                    previous_bicep_angle = bicep_angle
                    stage = "down"
                elif bicep_angle > 90 and start_curl:
                    end_curl = True
                    start_curl = False
                    stage = "up"
                
                #Percentage of success of pushup
                per = np.interp(bicep_angle, (20, 90), (100, 0))

                #Bar to show Pushup progress
                bar = np.interp(bicep_angle, (20, 90), (50, 380))
                # Count the number of bicep curls completed
                
                #increase rep count
                if end_curl:
                    reps += 1
                    end_curl = False

            # Update the progress bar
            if reps != prev_reps:
                prev_reps = reps
                progress = int(reps / total_reps * BAR_WIDTH)
            

            progress_bar(image, progress)
            display_reps(image, bicep_angle, reps, stage)
            draw_performance_bar(image, per, bar)
            
            # Add pose correction visualization
            if bicep_angle < 20:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            elif bicep_angle > 90:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
           
            
            # Exit loop if the 'q' key is pressed
            cv2.imshow('Exercise Counter', image)                             

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    return


# # Bicep Curls

# In[ ]:
from core.camera import VideoCamera
camera = VideoCamera()

def bicep_curl_rep(total_reps):
    # Initialize the video capture object
    #cap = cv2.VideoCapture(video_capture_number)
    
    # Initialize variables
    # print("in bicep_dfkasdjf")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
    prev_reps = 0
    reps = 0
    start_curl = False
    end_curl = False
    bicep_angle = 0
    previous_bicep_angle = 0
    progress = 0
    stage = "down"
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while reps < total_reps:
            # image = camera.get_frame()
            ret, image = cap.read()
            print("dfddddddddddddd",image.size())
            # Convert the image to RGB format

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the shoulder, elbow and wrist landmarks
            if results.pose_landmarks is not None:
                shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                elbow_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

                shoulder = np.array([shoulder_landmark.x, shoulder_landmark.y])
                elbow = np.array([elbow_landmark.x, elbow_landmark.y])
                wrist = np.array([wrist_landmark.x, wrist_landmark.y])

                bicep_angle = calculate_angle(shoulder, elbow, wrist)

                # Check if the bicep is being curled
                if bicep_angle > 160 and not start_curl:
                    start_curl = True
                    previous_bicep_angle = bicep_angle
                    stage = "down"
                elif bicep_angle < 40 and start_curl:
                    end_curl = True
                    start_curl = False
                    stage = "up"
                
                #Percentage of success of pushup
                per = np.interp(bicep_angle, (40, 160), (100, 0))

                #Bar to show Pushup progress
                bar = np.interp(bicep_angle, (40, 160), (50, 380))
                # Count the number of bicep curls completed
                
                #increase rep count
                if end_curl:
                    reps += 1
                    end_curl = False

            # Update the progress bar
            if reps != prev_reps:
                prev_reps = reps
                progress = int(reps / total_reps * BAR_WIDTH)
            

            progress_bar(image, progress)
            display_reps(image, bicep_angle, reps, stage)
            draw_performance_bar(image, per, bar)
            
            # Add pose correction visualization
            if bicep_angle < 40:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            elif bicep_angle > 160:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
           
            
            # Exit loop if the 'q' key is pressed
            ret, jpeg = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )
            # cv2.imshow('Exercise Counter', image)                             

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    #cap.release()
    #cv2.destroyAllWindows()
    return


# # Squat

# In[ ]:


def squat_rep(cap, total_reps):
    # Initialize the video capture object
    #cap = cv2.VideoCapture(video_capture_number)  
    
    # Initialize variables
    prev_reps = 0
    reps = 0
    start_squat = False
    end_squat = False
    knee_angle = 0
    previous_knee_angle = 0
    stage = "up"
    progress = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while reps < total_reps:
            ret, image = cap.read()
            # Convert the image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run the pose estimator
            results = pose.process(image)
            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Calculate the angle between the hip, knee and ankle landmarks
            if results.pose_landmarks is not None:
                hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                ankle_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

                hip = np.array([hip_landmark.x, hip_landmark.y])
                knee = np.array([knee_landmark.x, knee_landmark.y])
                ankle = np.array([ankle_landmark.x, ankle_landmark.y])

                knee_angle = calculate_angle(hip, knee, ankle)

                # Check if the squat is being performed
                if knee_angle < 100 and not start_squat:
                    start_squat = True
                    stage = "up"
                    previous_knee_angle = knee_angle
                elif knee_angle > 140 and start_squat:
                    stage = "down"
                    end_squat = True
                    start_squat = False
                    
                #Percentage of success of pushup
                per = np.interp(knee_angle, (100, 140), (100, 0))

                #Bar to show Pushup progress
                bar = np.interp(knee_angle, (100, 140), (50, 380))
                
                # Count the number of squats completed
                if end_squat:
                    reps += 1
                    end_squat = False
            
            if reps != prev_reps:
                prev_reps = reps
                progress = int(reps / total_reps * BAR_WIDTH)
            

            # Draw the progress bar
            progress_bar(image, progress)
            display_reps(image, knee_angle, reps, stage)
            draw_performance_bar(image, per, bar)
            
            # Add pose correction visualization
            if knee_angle > 140:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            elif knee_angle < 100:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
           
            cv2.imshow('Exercise Counter', image)                             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    #cap.release()
    #cv2.destroyAllWindows()  
    return


# # Shoulder Press

# In[ ]:


def shoulder_press_rep(cap, total_reps):
    prev_reps = 0
    reps = 0
    prev_pose = "down"
    progress = 0
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while reps < total_reps:
            # Read frame
            ret, image = cap.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get coordinates of relevant landmarks
            if results.pose_landmarks is not None:
                shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                elbow_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

                if shoulder_landmark and elbow_landmark and wrist_landmark:
                    # Calculate angle between landmarks
                    shoulder = np.array([shoulder_landmark.x, shoulder_landmark.y])
                    elbow = np.array([elbow_landmark.x, elbow_landmark.y])
                    wrist = np.array([wrist_landmark.x, wrist_landmark.y])

                    bicep_angle = calculate_angle(shoulder, elbow, wrist)
                    # Check if the person is in the correct position for shoulder press
                    if bicep_angle > 170 and shoulder[1] > elbow[1] and elbow[1] > wrist[1]:
                        current_pose = "up"
                        if prev_pose == "down" and current_pose == "up":
                            reps += 1
                        prev_pose = current_pose
                    else:
                        prev_pose = "down"
            #Percentage of success of pushup
            per = np.interp(bicep_angle, (40, 160), (0, 100))

            #Bar to show Pushup progress
            bar = np.interp(bicep_angle, (40, 160), (380, 50))
            
            # Count the number of bicep curls completed
            if reps != prev_reps:
                prev_reps = reps
                progress = int(reps / total_reps * BAR_WIDTH)
                
            progress_bar(image, progress)
            display_reps(image, bicep_angle, reps, prev_pose)
            draw_performance_bar(image, per, bar)
            
            # Add pose correction visualization
            if bicep_angle > 170:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            elif bicep_angle < 40:
                # Red color for incorrect pose
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
            
            # Display frame
            cv2.imshow('Exercise Counter', image)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video and close windows
    return


# # Jumping Jacks

# In[ ]:


def jumping_jacks_rep(cap, total_reps):
    # set up initial values
    prev_reps = 0
    reps = 0
    stage = "down"
    progress = 0
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:
            # read video feed
            ret, frame = cap.read()

            # flip image horizontally for natural viewing
            frame = cv2.flip(frame, 1)

            # convert image to RGB for mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # detect landmarks using mediapipe
            results = pose.process(image)

            # draw landmarks on image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # calculate positions of landmarks
            if results.pose_landmarks:
                left_shoulder=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_wrist=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_ankle=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_shoulder=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_wrist=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_ankle=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # Calculate angle
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
                
                #Percentage of success of pushup
                per = np.interp(left_shoulder_angle, (30, 160), (0, 100))

                #Bar to show Pushup progress
                bar = np.interp(left_shoulder_angle, (30, 160), (380, 50))
                
                # calculate difference in shoulder and knee positions
                if left_shoulder_angle < 30 and right_shoulder_angle < 30 and left_hip_angle > 170 and right_hip_angle > 170 :
                    stage = "down"
                if left_shoulder_angle > 100 and right_shoulder_angle > 100 and left_hip_angle < 165 and right_hip_angle < 165 and stage == "down":
                    stage = "up"
                    reps+=1
                
                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, right_hip_angle, reps, stage)
                draw_performance_bar(image, per, bar)
                
                # Add pose correction visualization
                if left_shoulder_angle < 30 and right_shoulder_angle < 30 and left_hip_angle > 170 and right_hip_angle > 170:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif left_shoulder_angle > 100 and right_shoulder_angle > 100 and left_hip_angle < 165 and right_hip_angle < 165:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))


                # display image
                cv2.imshow('Exercise Counter', image)

                # stop the program on the press of the 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


# # Right Knee Touches

# In[ ]:


def right_knee_touch_rep(cap, total_reps):
    # Initialize variables for rep counting and pose correction
    reps = 0
    prev_reps = 0
    stage = "down"
    progress = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:
            # Read the frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error reading the camera frame.")
                break

            # Flip the frame horizontally for a more natural viewing experience
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB format and process it with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the right shoulder, right hip, and right knee landmarks
            if results.pose_landmarks is not None:
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

                left_angle = calculate_angle(np.array([left_shoulder.x, left_shoulder.y]),np.array([left_hip.x, left_hip.y]),np.array([left_knee.x, left_knee.y]))

                if left_angle > 160 and left_angle < 200:
                    stage = "down"
                elif left_angle < 100:
                    if stage == "down":
                        reps += 1
                    stage = "up"

                left_per = np.interp(left_angle, (100, 165), (100, 0))
                left_bar = np.interp(left_angle, (100, 165), (50, 380))
                
                
                # Add pose correction visualization
                if left_angle > 160 and left_angle < 200:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif left_angle < 100:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, left_angle, reps, stage)
                draw_performance_bar(image, left_per, left_bar)

                # Display the frame
                cv2.imshow("Exercise Counter", image)

                # Exit the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# # Left Knee Touches

# In[ ]:


def left_knee_touch_rep(cap, total_reps):
    # Initialize variables for rep counting and pose correction
    reps = 0
    prev_reps = 0
    stage = "down"
    progress = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:
            # Read the frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error reading the camera frame.")
                break

            # Flip the frame horizontally for a more natural viewing experience
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB format and process it with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the right shoulder, right hip, and right knee landmarks
            if results.pose_landmarks is not None:
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

                right_angle = calculate_angle(np.array([right_shoulder.x, right_shoulder.y]),np.array([right_hip.x, right_hip.y]),np.array([right_knee.x, right_knee.y]))

                if right_angle > 160 and right_angle < 200:
                    stage = "down"
                elif right_angle < 100:
                    if stage == "down":
                        reps += 1
                    stage = "up"

                right_per = np.interp(left_angle, (100, 165), (100, 0))
                right_bar = np.interp(left_angle, (100, 165), (50, 380))
                
                
                # Add pose correction visualization
                if right_angle > 160 and right_angle < 200:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif right_angle < 100:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, right_angle, reps, stage)
                draw_performance_bar(image, right_per, right_bar)

                # Display the frame
                cv2.imshow("Exercise Counter", image)

                # Exit the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# # Crunches

# In[ ]:


def crunches_rep(cap, total_reps):

    # Initialize the video capture object
    # cap = cv2.VideoCapture(video_capture_number)

    # Initialize variables
    prev_reps = 0
    reps = 0
    start_crunch = False
    end_crunch = False
    crunch_angle = 0
    previous_crunch_angle = 0
    countdown = 0
    progress = 0
    stage = "down"
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:    
            ret, image = cap.read()

            # Convert the image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the shoulder, hip and knee landmarks
            if results.pose_landmarks is not None:
                shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

                shoulder = np.array([shoulder_landmark.x, shoulder_landmark.y])
                hip = np.array([hip_landmark.x, hip_landmark.y])
                knee = np.array([knee_landmark.x, knee_landmark.y])
                
                crunch_angle = calculate_angle(shoulder, hip, knee)

                # Check if the crunch is being performed
                if crunch_angle > 130 and not start_crunch:
                    start_crunch = True
                    previous_crunch_angle = crunch_angle
                    stage = "down"
                elif crunch_angle < 100 and start_crunch:
                    end_crunch = True
                    start_crunch = False
                    stage = "up"
                # Count the number of crunches completed
                if end_crunch:
                    reps += 1
                    end_crunch = False
                
                #Percentage of success of pushup
                per = np.interp(crunch_angle, (130, 100), (0, 100))

                #Bar to show Pushup progress
                bar = np.interp(crunch_angle, (130, 100), (50, 380))
                
                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, crunch_angle, reps, stage)
                draw_performance_bar(image, per, bar)

                if crunch_angle > 130:
                # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif crunch_angle < 100:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                # Show the image
                cv2.imshow('Exercise Counter', image)

                # Exit loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture object and close all windows
        # cap.release()
        #cv2.destroyAllWindows()
        return


# # Push Ups

# In[ ]:


def pushups_rep(cap, total_reps):

    # Initialize the pose estimator
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize the video capture object
    #cap = cv2.VideoCapture(video_capture_number)

    # Initialize variables
    prev_reps = 0
    reps = 0
    direction = 0
    form = 0
    stage = "Fix Form"
    progress = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:
            ret, image = cap.read()

            # Convert the image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip the image horizontally
            image = cv2.flip(image, 1)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the angle between the shoulder, elbow and wrist landmarks
            if results.pose_landmarks is not None:
                shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                elbow_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                
                shoulder = np.array([shoulder_landmark.x, shoulder_landmark.y])
                hip = np.array([hip_landmark.x, hip_landmark.y])
                elbow = np.array([elbow_landmark.x, elbow_landmark.y])
                wrist = np.array([wrist_landmark.x, wrist_landmark.y])
                knee = np.array([knee_landmark.x, knee_landmark.y])
                
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(elbow, shoulder, hip)
                hip_angle = calculate_angle(shoulder, hip, knee)

                #Percentage of success of pushup
                per = np.interp(elbow_angle, (90, 160), (0, 100))

                #Bar to show Pushup progress
                bar = np.interp(elbow_angle, (90, 160), (380, 50))
                # Check if the pushup is being performed correctly
                
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 140:
                    form = 1
                    
                if form == 1:
                    if elbow_angle <= 100 and hip_angle > 140:
                        stage = "down"
                        if direction == 0:
                            reps += 1
                            direction = 1
                    else:
                        stage = "Fix Form"

                    if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 140:
                        stage = "up"
                        if direction == 1:
                            #reps += 0.5
                            direction = 0
                    else:
                        stage = "Fix Form"
                    
                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)

                 # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, elbow_angle, reps, stage)
                draw_performance_bar(image, per, bar)

                if elbow_angle > 160:
                # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif elbow_angle < 90:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                # Show the image
                cv2.imshow('Exercise Counter', image)

                # Exit loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture object and close all windows
        #cap.release()
        #cv2.destroyAllWindows()
        return

# text_to_speech
def text_to_speech(text):
    output = gTTS(text=text, lang=language, slow=False)
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "output"+date_string+"audio.mp3"
    output.save(filename)
    playsound(filename)
    os.remove(filename)

# # Difficulty Level

# In[ ]:


def difficulty(level):
    
    if level == "Hard":
        return 3
    elif level == "Medium":
        return 2
    else:
        return 1


# # Total Repetitions

# In[ ]:


def total_repetitions(level):
    base_reps = 5
    return level * base_reps


# # Main function

# In[ ]:


def function_map():
    function_mapping = {"bicep_curls": bicep_curl_rep, "squats": squat_rep, "pushups": pushups_rep, "crunches":crunches_rep,
                       "jumpingjacks":jumping_jacks_rep, "RightKneeTouches":right_knee_touch_rep, 
                        "LeftKneeTouches":left_knee_touch_rep, "shoulderPress":shoulder_press_rep, "armRaise":arm_raise_rep}
    return function_mapping
def routine():   
    workout_routine = ["bicep_curls", "squats", "jumpingjacks", "pushups", "crunches","jumpingjacks","armRaise","shoulderPress","LeftKneeTouches"]
    #workout_routine = ["armRaise"]
    return workout_routine


# In[ ]:


def main():
    
    #Please enter the difficulty level
    difficulty_level = input("Easy, Medium or Hard: ")
    level = difficulty(difficulty_level)
    total_reps = total_repetitions(level)
    function_mapping = function_map()
    workout_routine = routine()
    cap = cv2.VideoCapture(video_capture_number)
    #adjust_camera_position(cap)
    #display_picture("bicep_curls")
    #audio.text_to_speech("You have chosen " + difficulty_level + " level of difficulty. Let's start, you got this !")
    
    for exercise in workout_routine:
        display_picture("bicep_curls")
        text_to_speech("Next Exercise is " + exercise + ", The exercise starts in ")
        for i in range(2,0,-1):
            text_to_speech(str(i))
            time.sleep(1)
        function_mapping[exercise](cap,total_reps)

    cap.release()
    cv2.destroyAllWindows()


