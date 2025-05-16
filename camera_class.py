from picamera2 import Picamera2
from libcamera import controls, Transform
from collections import deque
import pytz
from datetime import datetime
import time, sys, gc
import cv2
import numpy as np

CAM_X = 1920
CAM_Y = 1080

CAM_HFLIP = 1
CAM_VFLIP = 1

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()

        video_cfg = self.picam2.create_video_configuration(
            main={"size": (CAM_X, CAM_Y), "format": "RGB888"},
            controls={"FrameRate": 6}  # Still queue at 2 FPS if desired
            transform=Transform(hflip=CAM_HFLIP, vflip=CAM_VFLIP) 
        )
        self.picam2.configure(video_cfg)
        self.picam2.start()

        time.sleep(2)

    def _restart_camera(self):
        self.picam2.stop()
        self.picam2.close()
        gc.collect()
        self.__init__()

    def fill_queue(self, q: deque):
        i = 0
        tz = pytz.timezone("Europe/Berlin")
        last_enqueue_time = time.time()

        while True:
            rgb = self.picam2.capture_array("main")
            now = time.time()
            if now - last_enqueue_time >= 0.5:  # 2 FPS target
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
                q.append((timestamp, frame))
                last_enqueue_time = now

                print(f"Quelength: {len(q)}\tFrame shape: {frame.shape}")
                i += 1

                if i >= 60:
                    print("Loop ended, restarting camera resources…")
                    self._restart_camera()
                    i = 0

