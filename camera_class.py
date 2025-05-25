import threading
import logging
import time
import cv2
import numpy as np
import pytz
import gc
import importlib
import subprocess
import shlex
import sys
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
    def __init__(self, q=None, camera_url=None):
        self.q = q if q is not None else deque(maxlen=config.MAX_QUEUE_LEN)
        self.camera_url = camera_url
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
        self.ffmpeg_process = None

        # FPS monitoring
        self.last_frame_time = None
        self.fps = 0.0
        self.fps_log_interval = getattr(config, 'FPS_LOG_INTERVAL', 5)
        self.last_fps_log_time = time.time()

        # Initialize the appropriate camera source
        self._initialize_camera()

        # Start capture loop in separate thread
        self.capture_thread = threading.Thread(target=self._fill_queue_loop, daemon=True)
        self.capture_thread.start()

    def get_fps(self):
        return self.fps

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
                elif self.camera_type == "rtsp" or self.camera_type == "mjpeg":
                    cmd = f"ffmpeg -loglevel warning -rtsp_transport tcp -i {shlex.quote(self.camera_url)} -f rawvideo -pix_fmt bgr24 -"
                    self.ffmpeg_process = subprocess.Popen(
                        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
                    )
                else:
                    self.capture = cv2.VideoCapture(self.camera_url)
                    if self.camera_type == "usb":
                        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_X)
                        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_Y)
                        self.capture.set(cv2.CAP_PROP_FPS, 6)
                    if not self.capture.isOpened():
                        raise RuntimeError(f"Failed to open stream: {self.camera_url}")
                logging.debug("Camera initialized successfully.")
                return
            except Exception as e:
                logging.warning("Attempt %d to initialize camera failed: %s", attempt + 1, e)
                time.sleep(delay)
        raise RuntimeError(f"Unable to initialize camera after {retries} attempts")

    def stop(self):
        self._stop_event.set()
        self.capture_thread.join()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        if self.capture:
            self.capture.release()

    def _restart_camera(self):
        logging.debug("Restarting camera...")
        if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
            self.picam2.stop()
            self.picam2.close()
        elif self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        elif self.capture:
            self.capture.release()
        gc.collect()
        self.motion_scores.clear()
        self.previous_gray = None
        self._initialize_camera()
        self._update_config_cache()
        logging.debug("Camera restarted and config cache updated.")

    def _update_motion_average(self, score, window_size=5):
        self.motion_scores.append(score)
        if len(self.motion_scores) > window_size:
            self.motion_scores.pop(0)
        return sum(self.motion_scores) / len(self.motion_scores)

    def _read_rtsp_frame(self):
        width = config.CAM_X
        height = config.CAM_Y
        frame_size = width * height * 3
        raw_frame = self.ffmpeg_process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            logging.warning("Incomplete frame received from ffmpeg. Expected %d bytes, got %d.", frame_size, len(raw_frame))
            return None
        try:
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            return frame
        except Exception as e:
            logging.warning("Error reshaping frame: %s", e)
            return None

    def _update_config_cache(self):
        importlib.reload(config)
        self.motion_thresholds = getattr(config, 'MOTION_THRESHOLDS', {
            "low": 2.0,
            "medium": 5.0,
            "high": 10.0
        })
        self.enqueue_intervals = getattr(config, 'ENQUEUE_INTERVALS', {
            "slow": 1.5,
            "medium": 1.0,
            "fast": 0.5,
            "max": 0.1
        })
        self.motion_threshold = getattr(config, 'MOTION_THRESHOLD', 2.5)
        self.forced_enqueue_interval = getattr(config, 'FORCED_ENQUEUE_INTERVAL', 60)

        logging.debug(
            "Reloaded config: thresholds=%s, intervals=%s, motion_threshold=%.2f, forced_interval=%.1f",
            self.motion_thresholds, self.enqueue_intervals, self.motion_threshold, self.forced_enqueue_interval
        )

    def _fill_queue_loop(self):
        i = 0
        tz = pytz.timezone("Europe/Berlin")
        enqueue_interval = 1.0
        last_config_reload = 0
        config_reload_interval = 5

        self._update_config_cache()

        last_forced_enqueue = time.time()
        self.forced_enqueue_interval = getattr(config, 'FORCED_ENQUEUE_INTERVAL', 60)
        initial_frames_sent = 0
        initial_frames_count = 5

        while not self._stop_event.is_set():
            now = time.time()

            if now - last_config_reload > config_reload_interval:
                self._update_config_cache()
                last_config_reload = now

            try:
                if self.camera_type == "libcamera" and LIBCAMERA_AVAILABLE:
                    rgb = self.picam2.capture_array("main")
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                elif self.camera_type == "rtsp" or self.camera_type == "mjpeg":
                    frame = self._read_rtsp_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                else:
                    ret, frame = self.capture.read()
                    if not ret:
                        raise RuntimeError("Failed to read from camera")

                current_time = time.time()
                if self.last_frame_time is not None:
                    delta = current_time - self.last_frame_time
                    if delta > 0:
                        self.fps = 1.0 / delta
                self.last_frame_time = current_time
                if self.fps_log_interval and (current_time - self.last_fps_log_time) > self.fps_log_interval:
                    logging.info("Current FPS for %s camera: %.2f", self.camera_type, self.fps)
                    self.last_fps_log_time = current_time

                logging.debug("Frame successfully read: shape=%s", frame.shape)
            except Exception as e:
                logging.warning("Frame read failed: %s. Restarting camera…", e)
                self._restart_camera()
                continue

            self.node_live_img = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_hash = np.mean(frame)
            logging.debug("Frame mean pixel value: %.2f", frame_hash)

            should_enqueue = False
            if self.previous_gray is None:
                self.previous_gray = gray.copy()
                initial_frames_sent = 0  # Reset after restart
                last_forced_enqueue = now  # Trigger initial forced enqueue
                logging.debug("First frame after restart, skipping motion detection.")
                continue

            diff = cv2.absdiff(self.previous_gray, gray)
            non_zero_count = np.sum(diff > 5)
            diff_score = non_zero_count / diff.size * 100
            avg_motion = diff_score
            logging.debug("Motion Score: %.2f%% (diff only)", avg_motion)
            logging.debug("Non-zero diff count: %d / %d", non_zero_count, diff.size)

            if avg_motion > self.motion_thresholds["high"]:
                enqueue_interval = self.enqueue_intervals["max"]
            elif avg_motion > self.motion_thresholds["medium"]:
                enqueue_interval = self.enqueue_intervals["fast"]
            elif avg_motion > self.motion_thresholds["low"]:
                enqueue_interval = self.enqueue_intervals["medium"]
            else:
                enqueue_interval = self.enqueue_intervals["slow"]

            should_enqueue = avg_motion >= self.motion_threshold

            if initial_frames_sent < initial_frames_count or (now - last_forced_enqueue > self.forced_enqueue_interval):
                should_enqueue = True
                logging.debug("Forcing enqueue due to startup or interval.")

            if not should_enqueue and len(self.q) == 0 and (now - last_forced_enqueue > self.forced_enqueue_interval / 2):
                should_enqueue = True
                logging.debug("Emergency enqueue: queue is empty too long.")

            logging.debug("Motion Score: %.2f%%, should_enqueue: %s, queue_length: %d", avg_motion, should_enqueue, len(self.q))

            self.previous_gray = gray.copy()
            logging.debug("Using enqueue_interval: %.2fs", enqueue_interval)

            if should_enqueue:
                timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
                if len(self.q) < config.MAX_QUEUE_LEN:
                    self.q.append((timestamp, frame))
                    logging.debug("Enqueued frame. Queue length: %d Frame shape: %s", len(self.q), frame.shape)
                    initial_frames_sent += 1
                    last_forced_enqueue = now
                else:
                    logging.warning("Queue is full: %s, Frame dropped.", len(self.q))
            else:
                logging.debug("Low motion, skipping frame.")

            i += 1

            if now - getattr(self, "_last_debug_log_time", 0) > 0.5:
                sys.stdout.write(
                    f"\r[DEBUG] FPS: {self.fps:.2f} | Motion: {avg_motion:.2f}% | Enqueue: {should_enqueue} | Interval: {enqueue_interval:.2f}s | Queue: {len(self.q):2d}   "
                )
                sys.stdout.flush()
                self._last_debug_log_time = now

            time.sleep(enqueue_interval)

            if i >= 60:
                logging.info("Refreshing camera resources after 60 cycles.")
                self._restart_camera()
                self.last_frame_time = None
                self.last_fps_log_time = time.time()
                self.previous_gray = None
                self.node_live_img = None
                i = 0
