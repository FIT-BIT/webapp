#!/usr/bin/env python
# coding: utf-8

# # Install and Import Dependencies

# In[1]:


import cv2
import mediapipe as mp
import numpy as np 
import time
from  core import PoseModule as pm
from  core import AudioCommSys as audio
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# # Utilities 

# In[2]:


def display_picture(exercise, seconds = 2):
        image = cv2.imread(r"C:\Users\navya\Jupyter Notebooks\Major Project\webapp\web_project\core\images\{}.jpg".format(exercise))
        image = cv2.resize(image, (800, 500))
        while seconds > 0:
            ret, jpeg = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )
            time.sleep(1)
            seconds-=1

def draw_performance_bar(image, per, bar):
    cv2.rectangle(image, (580, 50), (600, 380), (0, 255, 0), 3)
    cv2.rectangle(image, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
    cv2.putText(image, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0), 2)
    return

def progress_bar(image, progress):
    # Draw the progress bar
    BAR_WIDTH = 400
    BAR_HEIGHT = 15
    BAR_POS = (20, 110)
    cv2.rectangle(image, BAR_POS, (BAR_POS[0] + progress, BAR_POS[1] + BAR_HEIGHT), (0, 255, 0), -1)
    cv2.rectangle(image, BAR_POS, (BAR_POS[0] + BAR_WIDTH, BAR_POS[1] + BAR_HEIGHT), (255, 255, 255), 2)
    return

def display_reps(image, angle, reps, stage, type_name = "Reps"):
    # Display the bicep angle and number of reps on the image
    if angle is not None:
        cv2.putText(image, "Angle: {:.2f} deg".format(angle), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "{}: {}".format(type_name, reps), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if stage is not None:
        cv2.putText(image, "Stage: {}".format(stage), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return


# # Global Variables

# In[3]:


video_capture_number = 0
BAR_WIDTH = 400


# # Calculate Angles

# In[4]:


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


# # Adjust Camera position 

# In[5]:


def adjust_camera_position(cap):

    while True:
        
        position = False
        # Read a frame from the video capture.
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 500))
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
                    cv2.putText(frame, "Entire body should be in frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert the image back from RGB to BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
        )
        
        if position == True:
            time.sleep(1)
            return

    # Release the video capture and close all windows.
    return


# # Arm Raise

# In[6]:


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
            image = cv2.resize(image, (800, 500))
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
            ret, jpeg = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )
                
    return


# # Bicep Curls

# In[7]:


def bicep_curl_rep(cap, total_reps):
    # Initialize the video capture object
    #cap = cv2.VideoCapture(video_capture_number)
    
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
            image = cv2.resize(image, (800, 500))
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
    
    #cap.release()
    #cv2.destroyAllWindows()
    return


# # Hammer Curls

# In[8]:


def hammer_curl_rep(cap, total_reps):
    # Initialize the video capture object
    #cap = cv2.VideoCapture(video_capture_number)
    
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
            image = cv2.resize(image, (800, 500))
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
    
    #cap.release()
    #cv2.destroyAllWindows()
    return


# # Squat

# In[9]:


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
            image = cv2.resize(image, (800, 500))
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
            
           
            ret, jpeg = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )
                
    #cap.release()
    #cv2.destroyAllWindows()  
    return


# # Shoulder Press

# In[10]:


def shoulder_press_rep(cap, total_reps):
    prev_reps = 0
    reps = 0
    prev_pose = "down"
    progress = 0
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while reps < total_reps:
            # Read frame
            ret, image = cap.read()
            image = cv2.resize(image, (800, 500))
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
            ret, jpeg = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )

    # Release video and close windows
    return


# # Jumping Jacks

# In[11]:


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
            frame = cv2.resize(frame, (800, 500))
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
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )

    # release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


# # Right Knee Touches

# In[12]:


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
            frame = cv2.resize(frame, (800, 500))
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
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )

    return    


# # Walk

# In[13]:


def walk(cap, total_reps):
    # Initialize variables for rep counting and pose correction
    reps = 0
    prev_reps = 0
    progress = 0
    status = False
    if total_reps < 10:
        total_reps*=10
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while  reps < total_reps:
            # Read the frame from the camera
            ret, frame = cap.read()
            frame = cv2.resize(frame, (800, 500))
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
                
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                
                left_knee = np.array([left_knee.x, left_knee.y])
                right_knee = np.array([right_knee.x, right_knee.y])
                
                if status:
                    if left_knee[0] > right_knee[0]:
                        reps += 1
                        status = False

                else:
                    if left_knee[0] < right_knee[0]:
                        reps += 1
                        status = True

                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, None, reps, None, "Steps")
                # Display the frame
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )


# # Left Knee Touches

# In[14]:


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
            frame = cv2.resize(frame, (800, 500))
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
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )


# # Crunches

# In[15]:


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
            image = cv2.resize(image, (800, 500))

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
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )

        # Release the video capture object and close all windows
        # cap.release()
        #cv2.destroyAllWindows()
        return


# # Sit ups

# In[16]:


def sit_ups_rep(cap, total_reps):

    # Initialize the video capture object
    # cap = cv2.VideoCapture(video_capture_number)

    # Initialize variables
    prev_reps = 0
    reps = 0
    start_situp = False
    countdown = 0
    progress = 0
    stage = "down"
    halfway = False
    range_flag = False
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while reps < total_reps:    
            ret, image = cap.read()
            image = cv2.resize(image, (800, 500))
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
                landmarks = results.pose_landmarks.landmark
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                angle_knee = calculate_angle(left_hip, left_knee, left_heel)
                angle_body = calculate_angle(left_shoulder, left_hip, left_knee)
                
                if (angle_body < 80 and angle_body > 50) and stage == "down": #Half-way there (Used for checking bad situps)
                    halfway = True
                
                if angle_body < 40 and stage == "down": #Complete situp
                    stage = "up"
                    range_flag = True
                    
                if angle_body > 90 and angle_knee < 60: #Resting position;to check if situp was done properly
                    stage = "down"
                    
                    if halfway: #Check if a rep was attempted
                        if range_flag: #Check if a proper rep was performed
                            reps += 1
                            feedback = "Good repetition!"
                        else:
                            feedback = "Did not perform sit up completely."
                        range_flag = False #Reset vars
                        halfway = False

                if angle_knee > 70: #Triggers anytime the legs are not tucked in
                    feedback = "Keep legs tucked in closer"
                
                #Percentage of success of pushup
                per = np.interp(angle_body, (130, 40), (0, 100))

                #Bar to show Pushup progress
                bar = np.interp(angle_body, (130, 40), (50, 380))
                
                if reps != prev_reps:
                    prev_reps = reps
                    progress = int(reps / total_reps * BAR_WIDTH)


                # Draw the progress bar
                progress_bar(image, progress)
                display_reps(image, angle_body, reps, stage)
                draw_performance_bar(image, per, bar)

                if angle_body > 100 or angle_knee>60:
                # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
                elif angle_body < 40 or angle_knee>60:
                    # Red color for incorrect pose
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0 , 0), thickness=2, circle_radius=2), connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

                # Show the image
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )

        # Release the video capture object and close all windows
        # cap.release()
        #cv2.destroyAllWindows()
        return


# # Push Ups

# In[17]:


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
            image = cv2.resize(image, (800, 500))

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
                ret, jpeg = cv2.imencode(".jpg", image)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )

        # Release the video capture object and close all windows
        #cap.release()
        #cv2.destroyAllWindows()
        return


# # Main function

# In[31]:


def function_map():
    function_mapping = {"bicepCurls": bicep_curl_rep, "squats": squat_rep, "pushups": pushups_rep, "crunches":crunches_rep,
                       "jumpingjacks":jumping_jacks_rep, "RightKneeTouches":right_knee_touch_rep, 
                        "LeftKneeTouches":left_knee_touch_rep, "shoulderPress":shoulder_press_rep, "armRaise":arm_raise_rep, "hammerCurl":hammer_curl_rep, "SitUps":sit_ups_rep,
                       "walk": walk}
    return function_mapping

def routine(set = 0):  
    workout_routine = ["walk", "jumpingjacks"]
    if set == 1:
        workout_routine.extend(["bicepCurls", "shoulderPress", "RightKneeTouches", "squats"])
    elif set == 2:
        workout_routine.extend(["crunches", "pushups", "hammerCurl", "armRaise", "SitUps" ])
    #workout_routine = ["bicepCurls", "squats", "jumpingjacks", "pushups", "crunches","jumpingjacks","armRaise","shoulderPress","LeftKneeTouches", "hammerCurl", "RightKneeTouches", "SitUps", "walk"]
    #workout_routine = ["hammerCurl"]
    return workout_routine


# In[32]:


def difficulty(level):
    
    if level == "Hard":
        return 3
    elif level == "Medium":
        return 2
    else:
        return 1

def level_multiplier(level, base):
    return level * base

#def predefined_workout():
#    return total_reps, cooldown_period


# In[33]:


def main(cap,difficulty_level,set):
    
    #Please enter the difficulty level
    
    if (not difficulty_level):
        return
    # difficulty_level = input("Easy, Medium or Hard: ")
    # set = int(input("Set 1 or Set 2:"))
    level = difficulty(difficulty_level)
    total_reps = level_multiplier (level, 5)
    cooldown_period = level_multiplier (level, 5)
    function_mapping = function_map()
    workout_routine = routine(set)
    camera_position = adjust_camera_position(cap)
    for i in camera_position:
        yield(i)
    picture = display_picture("start")
    for i in picture:
        yield(i)
    audio.text_to_speech("You have chosen " + difficulty_level + " level of difficulty. Let's start, you got this !")
    
    for exercise in workout_routine:
        picture = display_picture(exercise)
        for i in picture:
            yield(i)
        audio.text_to_speech("Next Exercise is " + exercise + ", The exercise starts in ")
        for i in range(2,0,-1):
            audio.text_to_speech(str(i))
            time.sleep(1)
        output = function_mapping[exercise](cap,total_reps)
        for i in output:
            yield(i)
    picture = display_picture("end")
    for i in picture:
        yield(i)
    audio.text_to_speech("Well done, that was a great workout")