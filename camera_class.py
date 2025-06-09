#camera_class.py

"""
Cat Prey Analyzer - Camera and Frame Queue Logic Summary

- Purpose:
  - Handles all camera hardware interaction, frame acquisition, and queueing for downstream analysis.
  - Implements motion detection and periodic/heartbeat frame capture in a single threaded loop.
  - Feeds frames to the main analysis pipeline via a thread-safe queue.

- Queueing Logic:
  - Main loop (fill_queue) captures frames, detects motion, and enqueues frames on motion events.
  - Periodic/heartbeat logic: If no motion for a configurable interval, enqueues a frame to ensure recency ("heartbeat").
  - Queue is pre-filled at startup to ensure immediate availability to consumers (e.g., bot requests).
  - Pausing/resuming: Queue can be paused and cleared on system or user command.

- Camera Support:
  - Supports multiple camera types (USB, PiCam via libcamera, etc.).
  - Handles orientation (horizontal/vertical flip), error recovery, and camera restarts.
  - Converts frames to grayscale for efficient motion detection.

- Logging:
  - Logs frame capture, queue events, motion detection, and heartbeat actions.
  - Warnings for dropped frames, errors for camera failures and exceptions.

- All queuing and acquisition logic is centralized here, ensuring reliable and recent images for analysis and user requests.
"""

from datetime import datetime
import cv2
import time
import logging
import gc
import config
from threading import Event, Lock

# Conditionally import Picamera2 if available
try:
    from picamera2 import Picamera2
    from libcamera import Transform
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = Transform = None                        # type: ignore


class Camera:
    def __init__(self, q, camera_url):
        self.q = q
        self.pause_event = Event()
        self.pause_duration = 0.0
        self._pause_lock = Lock()
        self.sleep_interval = getattr(config, "SLEEP_INTERVAL", 0.25)
        self.queue_cycles = getattr(config, "FILL_QUEUE_CYCLES", 60)
        self.max_len = getattr(config, "MAX_QUEUE_LEN", 20)
        self.motion_threshold = getattr(config, "MOTION_THRESHOLD", 5000)  # Adjust as needed
        self.cam_x = getattr(config, "CAM_WIDTH", 640)
        self.cam_y = getattr(config, "CAM_HEIGHT", 480)
        self.flip_overrides = getattr(config, "CAMERA_FLIP_OVERRIDES", {})
        self.camera_url = camera_url
        self.camera_type = self._detect_camera_type()
        self.cap = None
        self.picam2 = None
        self._load_flip_overrides()
        if self.motion_threshold < 0:
            raise ValueError(f"Invalid motion_threshold: {self.motion_threshold}")
        if self.max_len <= 0:
            raise ValueError(f"Invalid MAX_QUEUE_LEN: {self.max_len}")
        if self.sleep_interval <= 0:
            raise ValueError(f"Invalid SLEEP_INTERVAL: {self.sleep_interval}")

        logging.info(
            f"Motion threshold is set to {self.motion_threshold} "
            f"({'low' if self.motion_threshold < 3000 else 'medium' if self.motion_threshold < 7000 else 'high'})"
        )

        # Initialize hardware
        self._initialize_camera()

    def _load_flip_overrides(self):
        key = str(self.camera_url)
        override = self.flip_overrides.get(key)

        if not override and self.camera_type == "usb":
            override = self.flip_overrides.get(f"usb:{self.camera_url}")

        override = override or self.flip_overrides.get("default", {})
        self.hflip = override.get("hflip", getattr(config, "CAM_HFLIP", False))
        self.vflip = override.get("vflip", getattr(config, "CAM_VFLIP", False))
        logging.info(f"Camera flip config: hflip={self.hflip}, vflip={self.vflip}")

    def _detect_camera_type(self):
        if not self.camera_url:
            if PICAMERA_AVAILABLE:
                logging.info("Using internal PiCamera2.")
                return "libcamera"
            else:
                raise RuntimeError("No camera URL provided and PiCamera2 is not available.")
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

        raise ValueError(f"Unsupported CAMERA_URL format: {self.camera_url}")

    def _initialize_camera(self):
        if self.camera_type == "libcamera":
            if not PICAMERA_AVAILABLE:
                raise RuntimeError("camera_type 'libcamera' selected but Picamera2 is not available.")
            if Picamera2 is None or Transform is None:
                raise RuntimeError("Picamera2 modules are not properly loaded.")
            self.picam2 = Picamera2()
            video_cfg = self.picam2.create_video_configuration(
                main={"size": (self.cam_x, self.cam_y), "format": "RGB888"},
                controls={"FrameRate": 6},
                transform=Transform(hflip=self.hflip, vflip=self.vflip)
            )
            self.picam2.configure(video_cfg)
            self.picam2.start()
            time.sleep(2)
            logging.info("PiCamera2 initialized.")
        elif self.camera_type in {"usb", "rtsp", "mjpeg", "video"}:
            self.cap = cv2.VideoCapture(self.camera_url)
            if self.camera_type == "usb":
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_x)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_y)
                self.cap.set(cv2.CAP_PROP_FPS, 6)
            if self.camera_type == "mjpeg":
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.camera_type == "rtsp":
                logging.debug("RTSP stream may need time to buffer. Sleeping briefly...")
                time.sleep(0.2)
            if not self.cap.isOpened():
                logging.error(f"Failed to open stream: {self.camera_url}, restarting camera")
                time.sleep(1)
                self._restart_camera()
            else:
                fps_reported = self.cap.get(cv2.CAP_PROP_FPS)
                logging.debug(f"Video stream {self.camera_type} opened successfully with: {self.camera_url}.")
                logging.debug(f"Camera FPS (reported): {fps_reported:.2f}")

    def _restart_camera(self):
        logging.warning("Restarting camera...")
        if self.camera_type == "libcamera" and self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.debug("Camera resources released.")
        gc.collect()
        self._initialize_camera()

    def fill_queue(self):
        i = 0
        last_enqueue_time = time.time()
        self.last_motion_enqueue_time = time.time() # Track last motion-based enqueue
        heartbeat_interval = 60  # seconds, adjust as needed
        logging.info(f"Starting queuing loop with {self.sleep_interval:.2f}s between frames ...")
        prev_gray = None  # Store the previous grayscale frame

        # Pre-fill queue with enough frames to exceed fps_offset
        num_prefill = getattr(self, "fps_offset", 2) + 1
        for _ in range(num_prefill):
            # Usual frame capture logic:
            if self.camera_type == "libcamera" and self.picam2:
                rgb = self.picam2.capture_array("main")
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                if self.hflip or self.vflip:
                    code = -1 if self.hflip and self.vflip else (1 if self.hflip else 0)
                    frame = cv2.flip(frame, code)
            timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%c.%f")
            if len(self.q) < self.max_len:
                self.q.append((timestamp, frame))
            time.sleep(self.sleep_interval)  # Small delay to avoid identical frames


        while True:
            try:
                # Pause and clear queue if pause_event is set
                if self.pause_event.is_set():
                    with self._pause_lock:
                        if len(self.q):
                            logging.info(f"Pausing queue and clearing all frames [{len(self.q)}]")
                            self.q.clear()
                        logging.info(f"Queueing paused for {self.pause_duration} seconds.")
                        pause_duration = self.pause_duration
                        self.pause_duration = 0.0
                    time.sleep(pause_duration)
                    self.pause_event.clear()
                    continue

                # Capture frame
                if self.camera_type == "libcamera" and self.picam2:
                    rgb = self.picam2.capture_array("main")
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        logging.warning("Failed to read frame from camera. Restarting...")
                        time.sleep(0.2)
                        self._restart_camera()
                        continue

                    # Apply flip if configured
                    if self.hflip or self.vflip:
                        code = -1 if self.hflip and self.vflip else (1 if self.hflip else 0)
                        frame = cv2.flip(frame, code)

                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                motion_detected = False
                if prev_gray is not None:
                    # Compute absolute difference between current and previous frame
                    frame_delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    motion_pixels = cv2.countNonZero(thresh)

                    if motion_pixels > self.motion_threshold:
                        motion_detected = True
                        logging.debug(f"Motion detected: {motion_pixels} changed pixels.")
                    #else: # this one is really verbose..
                    #    logging.debug(f"No significant motion: {motion_pixels} changed pixels.")

                prev_gray = gray  # Update previous frame
                now = time.time()

                # 1. Motion-based enqueuing
                if motion_detected and (now - last_enqueue_time >= self.sleep_interval):
                    timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%c.%f")
                    if len(self.q) < self.max_len:
                        self.q.append((timestamp, frame))
                        logging.debug(f"[{self.camera_type.upper()}] Enqueued frame at {timestamp} | Queue length: {len(self.q)}")
                    else:
                        logging.warning("Queue is full (%d), dropping frame.", self.max_len)
                    last_enqueue_time = now

                    i += 1
                    if i >= self.queue_cycles:
                        logging.info(f"Refreshing camera after {self.queue_cycles} frames.")
                        self._restart_camera()
                        i = 0

                # 2. Heartbeat/periodic enqueuing (if no motion for heartbeat_interval)
                elif (now - self.last_motion_enqueue_time) > heartbeat_interval and (now - last_enqueue_time >= self.sleep_interval):
                    timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%c.%f")
                    if len(self.q) < self.max_len:
                        self.q.append((timestamp, frame))
                        logging.info(f"🌙 Heartbeat: Enqueued frame at {timestamp} | Queue length: {len(self.q)} [quiet]")
                    else:
                        logging.warning("Queue is full (%d), dropping frame.", self.max_len)
                    last_enqueue_time = now
                    self.last_motion_enqueue_time = now  # treat as activity for next interval

                time.sleep(0.01)

            except Exception as e:
                logging.error("Exception in fill_queue: %s", e)
                self._restart_camera()
                time.sleep(1)

