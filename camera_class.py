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
import argparse

# Conditionally import Picamera2 if available
try:
    from picamera2 import Picamera2, Transform
    LIBCAMERA_AVAILABLE = True
except ImportError:
    LIBCAMERA_AVAILABLE = False

class Camera:
    def __init__(self, q: deque, camera_url=None):
        self.q = q
        self._stop_event = threading.Event()
        self.previous_gray = None

        # Default motion threshold fallback
        self.motion_threshold = getattr(config, 'MOTION_THRESHOLD', 2.5)
        self.motion_scores = []

        # Use the passed camera_url directly
        self.camera_url = camera_url

        # Determine camera type based on camera_url
        self.camera_type = self._detect_camera_type()
        self.capture = None

        # Initialize the appropriate camera source
        self._initialize_camera()

        # Start capture loop in separate thread
        self.capture_thread = threading.Thread(target=self._fill_queue_loop, daemon=True)
        self.capture_thread.start()

    def _detect_camera_type(self):
        if not self.camera_url:
            logging.info("Using internal PiCamera2.")
            return "libcamera"
        if isinstance(self.camera_url, int) or (isinstance(self.camera_url, str) and self.camera_url.isdigit()):
            self.camera_url = int(self.camera_url)
            logging.info("Using USB Camera.")
            return "usb"
        if self.camera_url.startswith("rtsp://"):
            logging.info("Using RTSP camera stream.")
            return "rtsp"
        if self.camera_url.startswith("http://") or self.camera_url.startswith("https://"):
            logging.info("Using MJPEG camera stream.")
            return "mjpeg"
        if self.camera_url.endswith(".mp4") or self.camera_url.endswith(".avi"):
            logging.info("Using avi/mp4 video file.")
            return "video"
        raise ValueError("Unsupported CAMERA_URL format")

    def _initialize_camera(self):
        retries = 5
        delay = 2
        for attempt in range(retries):
            try:
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
                else:
                    self.capture = cv2.VideoCapture(self.camera_url)
                    if self.camera_type == "usb":
                        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_X)
                        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_Y)
                        self.capture.set(cv2.CAP_PROP_FPS, 6)
                    if not self.capture.isOpened():
                        raise RuntimeError(f"Failed to open stream: {self.camera_url}")
                return
            except Exception as e:
                logging.warning("Attempt %d to initialize camera failed: %s", attempt + 1, e)
                time.sleep(delay)
        raise RuntimeError(f"Unable to initialize camera after {retries} attempts")

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
        self._initialize_camera()

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

        last_forced_enqueue = time.time()
        forced_enqueue_interval = getattr(config, 'FORCED_ENQUEUE_INTERVAL', 60)
        first_frame = True
        read_fail_count = 0
        max_read_failures = 10

        while not self._stop_event.is_set():
            now = time.time()

            if now - last_config_reload > config_reload_interval:
                importlib.reload(config)
                motion_thresholds = getattr(config, 'MOTION_THRESHOLDS', {
                    "low": 2.0,
                    "medium": 5.0,
                    "high": 10.0
                })
                enqueue_intervals = getattr(config, 'ENQUEUE_INTERVALS', {
                    "slow": 1.5,
                    "medium": 1.0,
                    "fast": 0.5,
                    "max": 0.1
                })
                self.motion_threshold = getattr(config, 'MOTION_THRESHOLD', 2.5)
                forced_enqueue_interval = getattr(config, 'FORCED_ENQUEUE_INTERVAL', 60)
                last_config_reload = now

                logging.debug(
                    "Reloaded config: thresholds=%s, intervals=%s, motion_threshold=%.2f, forced_interval=%.1f",
                    motion_thresholds, enqueue_intervals, self.motion_threshold, forced_enqueue_interval
                )

            # Capture frame from camera
            if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
                rgb = self.picam2.capture_array("main")
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                read_fail_count = 0
            else:
                ret, frame = self.capture.read()
                if not ret or frame is None or frame.size == 0:
                    read_fail_count += 1
                    logging.warning("Failed to read frame (%d/%d)", read_fail_count, max_read_failures)
                    time.sleep(0.5)
                    if read_fail_count >= max_read_failures:
                        logging.error("Too many failed reads. Restarting camera stream.")
                        self._restart_camera()
                        read_fail_count = 0
                    continue
                else:
                    read_fail_count = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion detection via frame differencing
            should_enqueue = False
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

                should_enqueue = avg_motion >= self.motion_threshold

            if first_frame or (now - last_forced_enqueue > forced_enqueue_interval):
                should_enqueue = True
                logging.debug("Forcing enqueue due to time/first-frame condition.")

            self.previous_gray = gray.copy()

            if should_enqueue:
                timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
                if len(self.q) < config.MAX_QUEUE_LEN:
                    self.q.append((timestamp, frame))
                    logging.debug("Enqueued frame. Queue length: %d", len(self.q))
                else:
                    logging.warning("Queue is full. Frame dropped.")
                first_frame = False
                last_forced_enqueue = now
            else:
                logging.debug("Low motion, skipping frame.")

            i += 1
            time.sleep(enqueue_interval)

            # Restart logic every 60 iterations to prevent camera freezes
            if i >= 60:
                logging.info("Loop ended, restarting camera resources…")
                self._restart_camera()
                break

