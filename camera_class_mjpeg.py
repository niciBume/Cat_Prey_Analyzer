from collections import deque
import pytz
from datetime import datetime
import time, gc
import cv2
import numpy as np

MJPEG_STREAM_URL = "http://localhost:9000/mjpg"  # Replace with your MJPEG stream URL

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(MJPEG_STREAM_URL)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open MJPEG stream: {MJPEG_STREAM_URL}, restarting camera")
        time.sleep(2)

    def _restart_camera(self):
        self.cap.release()
        gc.collect()
        self.__init__()

    def fill_queue(self, q: deque):
        i = 0
        tz = pytz.timezone("Europe/Berlin")
        last_enqueue_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from MJPEG stream, restarting camera")
                time.sleep(1)
                self._restart_camera()
                continue

            now = time.time()
            if now - last_enqueue_time >= 0.5:  # 0.5 sec = 2 FPS
                timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
                q.append((timestamp, frame))
                last_enqueue_time = now

                print(f"Quelength: {len(q)}\tFrame shape: {frame.shape}")
                i += 1

                if i >= 60:
                    raise RuntimeError("Loop ended, restarting camera resources, restarting camera")
                    self._restart_camera()
                    i = 0

