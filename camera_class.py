import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
from gpiozero import CPUTemperature

from collections import deque
import pytz
from datetime import datetime
from threading import Thread
import time
import sys
import cv2
import numpy as np
import io

class Camera:
    def __init__(self,):
        IRPin = 36
        # GPIO Stuff
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IRPin, GPIO.OUT)
        GPIO.output(IRPin, GPIO.LOW)

        time.sleep(2)

    def fill_queue(self, deque):
        while(1):
            camera = PiCamera()
            camera.framerate = 3
            camera.vflip = True
            camera.hflip = True
            camera.resolution = (2592, 1944)
            camera.exposure_mode = 'sports'
            stream = io.BytesIO()
            for i, frame in enumerate(camera.capture_continuous(stream, format="jpeg", use_video_port=True)):
                stream.seek(0)
                data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(data, 1)
                deque.append(
                    (datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), image))
                #deque.pop()
                print("Quelength: " + str(len(deque)) + "\tStreamsize: " + str(sys.getsizeof(stream)))
                if i == 60:
                    print("Loop ended, starting over.")
                    camera.close()
                    del camera
                    break