import threading
import logging
import time
import cv2
import numpy as np
import pytz
import gc
import importlib
from datetime import datetime
from collections import deque
import config

# Conditionally import Picamera2 if available
try:
    from picamera2 import Picamera2, Transform
    LIBCAMERA_AVAILABLE = True
except ImportError:
    LIBCAMERA_AVAILABLE = False

class Camera:
    def __init__(self, q: deque):
        self.q = q
        self._stop_event = threading.Event()
        self.previous_gray = None
        self.motion_threshold = config.MOTION_THRESHOLD
        self.motion_scores = []

        # Determine camera type based on CAMERA_URL
        self.camera_type = self._detect_camera_type()
        self.capture = None

        # Initialize the appropriate camera source
        if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
            self.picam2 = Picamera2()
            video_cfg = self.picam2.create_video_configuration(
                main={"size": (config.CAM_X, config.CAM_Y), "format": "RGB888"},
                controls={"FrameRate": 6},
                transform=Transform(hflip=config.CAM_HFLIP, vflip=config.CAM_VFLIP)
            )
            self.picam2.configure(video_cfg)
            self.picam2.start()
            time.sleep(2)

        elif self.camera_type in ["mjpeg", "rtsp", "usb", "video"]:
            self.capture = cv2.VideoCapture(CAMERA_URL)
            if self.camera_type == "usb":
                # USB webcam settings
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_X)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_Y)
                self.capture.set(cv2.CAP_PROP_FPS, 6)
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open stream: {CAMERA_URL}")
        else:
            raise ValueError("Unsupported camera source or missing dependencies")

        # Start capture loop in separate thread
        self.capture_thread = threading.Thread(target=self._fill_queue_loop, daemon=True)
        self.capture_thread.start()

    def _detect_camera_type(self):
        if not CAMERA_URL:
            return "libcamera"
        if CAMERA_URL.startswith("rtsp://"):
            return "rtsp"
        if CAMERA_URL.startswith("http://") or CAMERA_URL.startswith("https://"):
            return "mjpeg"
        if CAMERA_URL.isdigit():
            CAMERA_URL = int(CAMERA_URL)
            return "usb"
        if CAMERA_URL.endswith(".mp4") or CAMERA_URL.endswith(".avi"):
            return "video"
        raise ValueError("Unsupported CAMERA_URL format")

    def stop(self):
        self._stop_event.set()
        self.capture_thread.join()
        if self.camera_type in ["mjpeg", "rtsp", "usb", "video"]:
            self.capture.release()

    def _restart_camera(self):
        if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
            self.picam2.stop()
            self.picam2.close()
        elif self.capture:
            self.capture.release()
        gc.collect()
        self.__init__(self.q)

    def _update_motion_average(self, score, window_size=5):
        self.motion_scores.append(score)
        if len(self.motion_scores) > window_size:
            self.motion_scores.pop(0)
        return sum(self.motion_scores) / len(self.motion_scores)

    def _fill_queue_loop(self):
        i = 0
        tz = pytz.timezone("Europe/Berlin")
        enqueue_interval = 1.0
        last_config_reload = 0
        config_reload_interval = 5  # seconds

        while not self._stop_event.is_set():
            now = time.time()

            if now - last_config_reload > config_reload_interval:
                importlib.reload(config)
                motion_thresholds = config.MOTION_THRESHOLDS
                enqueue_intervals = config.ENQUEUE_INTERVALS
                self.motion_threshold = config.MOTION_THRESHOLD
                last_config_reload = now

                logging.debug(
                    "Reloaded config: thresholds=%s, intervals=%s, motion_threshold=%.2f",
                    motion_thresholds, enqueue_intervals, self.motion_threshold
                )

            # Capture frame from camera
            if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
                rgb = self.picam2.capture_array("main")
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.capture.read()
                if not ret:
                    logging.warning("Failed to read from stream")
                    time.sleep(1)
                    continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion detection via frame differencing
            if self.previous_gray is not None:
                diff = cv2.absdiff(self.previous_gray, gray)
                non_zero_count = np.sum(diff > 15)
                diff_score = non_zero_count / diff.size * 100
                avg_motion = self._update_motion_average(diff_score)
                logging.debug("Motion Score: %.2f%% (smoothed: %.2f%%)", diff_score, avg_motion)

                if avg_motion > motion_thresholds["high"]:
                    enqueue_interval = enqueue_intervals["max"]
                elif avg_motion > motion_thresholds["medium"]:
                    enqueue_interval = enqueue_intervals["fast"]
                elif avg_motion > motion_thresholds["low"]:
                    enqueue_interval = enqueue_intervals["medium"]
                else:
                    enqueue_interval = enqueue_intervals["slow"]

                if avg_motion < self.motion_threshold:
                    logging.debug("Low motion (%.2f%%), skipping frame.", avg_motion)
                    time.sleep(enqueue_interval)
                    continue

            self.previous_gray = gray.copy()

            # Timestamp and enqueue
            timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
            if len(self.q) < config.MAX_QUEUE_LEN:
                self.q.append((timestamp, frame))
                logging.debug("Enqueued frame. Queue length: %d", len(self.q))
            else:
                logging.warning("Queue is full. Frame dropped.")

            i += 1
            time.sleep(enqueue_interval)

            # Restart logic every 60 iterations to prevent camera freezes
            if i >= 60:
                logging.info("Loop ended, restarting camera resources…")
                self._restart_camera()
                break

