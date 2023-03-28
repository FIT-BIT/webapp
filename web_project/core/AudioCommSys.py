#!/usr/bin/env python
# coding: utf-8

# # Import Dependencies

# In[1]:


import speech_recognition as sr
from gtts import gTTS
import os
from io import BytesIO
from playsound import playsound
from datetime import datetime
import time
language = 'en'


# # Speech to text

# In[2]:


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please say something....")
        audio = recognizer.listen(source, timeout=2)
        try:
            print("You said: \n" + recognizer.recognize_google(audio))
            return (recognizer.recognize_google(audio))
        except Exception as e:
            print("Error: " + str(e))


# # Text to speech

# In[3]:


def text_to_speech(text):
    output = gTTS(text=text, lang=language, slow=False)
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    filename = "output"+date_string+"audio.mp3"
    output.save(filename)
    time.sleep(0.2)
    playsound(filename)
    os.remove(filename)
    time.sleep(0.2)


# # Main Method

# In[4]:


def main():  
    text_to_speech("Testing, Testing, Testing")


# In[5]:


if __name__ == "__main__":
    main()

