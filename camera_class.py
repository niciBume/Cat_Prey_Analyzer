# camera_class.py

"""
Cat Prey Analyzer - Camera Acquisition & Frame Queue Logic

Purpose:
    - Handles all camera hardware interaction, frame acquisition, and buffering for downstream analysis.
    - Implements motion detection and periodic (heartbeat) frame capture in a robust, pause-aware loop.
    - Supplies frames to the main analysis pipeline via a thread/process-safe queue.

Features:
    - Motion-triggered frame capture with user-configurable sensitivity (motion threshold).
    - Heartbeat capture: Ensures a fresh frame is queued even in absence of motion.
    - Queue pre-fill on startup for zero-latency user requests.
    - Pausing: Can pause and clear the frame queue on system/user command (e.g., during catflap opening).
    - Multi-camera support: USB cams, PiCam (libcamera), RTSP/MJPEG/IP cams, and local video files.
    - Orientation and error handling: Flip, restart, and recover on camera errors.
    - Detailed logging of all capture, queue, and motion events.

Integration:
    - Intended to run as a subprocess, controlled by the main cascade.py.
    - Receives inter-process signals/events for pausing and shutdown.

How to Tune:
    - Adjust motion sensitivity, heartbeat interval, and queue size in config.py.
    - Camera selection and overrides via CAMERA_OVERRIDES in config.py.
"""

import os
import cv2
import time
import gc
import sys
import config
import traceback
import logging
from datetime import datetime

# Conditionally import Picamera2 if available
try:
    from picamera2 import Picamera2
    from libcamera import Transform
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = Transform = None

class Camera:
    def __init__(self, q, camera_id, shutdown_flag, pause_event=None, pause_duration=None, log_level_str="INFO"):
        self.q = q
        self.camera_id = camera_id
        self.shutdown_flag = shutdown_flag
        if pause_event is None or pause_duration is None:
            raise ValueError("pause_event and pause_duration must be provided")
        self.pause_event = pause_event
        self.pause_duration = pause_duration
        self.restart_attempts = 0
        self.max_restart_attempts = 5
        self.max_frame_failures = getattr(config, "MAX_FRAME_FAILURES", 5)
        self.queue_cycles = getattr(config, "FILL_QUEUE_CYCLES", 60)
        self.fps_offset = getattr(config, "DEFAULT_FPS_OFFSET", 2)
        #self.last_enqueue_time = None
        self.heartbeat_interval = getattr(config, "HEARTBEAT_INTERVAL", 60)  # seconds
        self.motion_threshold = getattr(config, "MOTION_THRESHOLD", 5000)  # Adjust as needed
        if self.motion_threshold < 0:
            raise ValueError(f"Invalid motion_threshold: {self.motion_threshold}")
        self.max_queue_len = getattr(config, "MAX_QUEUE_LEN", 20)
        if self.max_queue_len <= 0:
            raise ValueError(f"Invalid MAX_QUEUE_LEN: {self.max_queue_len}")
        self.sleep_interval = getattr(config, "SLEEP_INTERVAL", 0.5)
        if self.sleep_interval <= 0:
            raise ValueError(f"Invalid SLEEP_INTERVAL: {self.sleep_interval}")
        self.cap = None
        self.picam2 = None
        threshold_category = 'low' if self.motion_threshold < 3000 else 'medium' if self.motion_threshold < 7000 else 'high'
        logging.info(f"Motion threshold is set to {self.motion_threshold} / ({threshold_category})")

        # Load config, fallback to default
        cam_cfg = config.CAMERA_OVERRIDES.get(camera_id, config.CAMERA_OVERRIDES['default'])
        self.base_url = cam_cfg.get('url')
        self.cam_x = cam_cfg.get('cam_width', config.CAMERA_OVERRIDES['default']['cam_width'])
        self.cam_y = cam_cfg.get('cam_height', config.CAMERA_OVERRIDES['default']['cam_height'])
        self.hflip = cam_cfg.get('hflip', config.CAMERA_OVERRIDES['default']['hflip'])
        self.vflip = cam_cfg.get('vflip', config.CAMERA_OVERRIDES['default']['vflip'])

        # Compose URL with credentials if needed
        self.camera_url = self._compose_url_with_creds()
        self.camera_type = self._detect_camera_type()
        logging.info(f"Camera settings: camera_id={self.camera_id}, base_url={self.base_url}, type={self.camera_type}, width={self.cam_x}, height={self.cam_y}, hflip={self.hflip}, vflip={self.vflip}")

    def _compose_url_with_creds(self):
        if not self.base_url:
            return None
        # Only add credentials if needed
        if self.base_url.startswith("rtsp://") or self.base_url.startswith("http://"):
            user = os.getenv(f"{self.camera_id.upper()}_USER")
            pw = os.getenv(f"{self.camera_id.upper()}_PASS")
            if user and pw:
                # Insert credentials into url after protocol
                proto, rest = self.base_url.split("://", 1)
                return f"{proto}://{user}:{pw}@{rest}"
        return self.base_url

    def start_hardware(self):
        self._initialize_camera()

    def _detect_camera_type(self):
        if not self.camera_url:
            if PICAMERA_AVAILABLE:
                logging.info("Using internal PiCamera2")
                return "libcamera"
            raise RuntimeError("No camera URL provided and PiCamera2 is not available!")
        if isinstance(self.camera_url, int) or (isinstance(self.camera_url, str) and self.camera_url.isdigit()):
            self.camera_url = int(self.camera_url)
            logging.info("Using USB Camera")
            return "usb"

        if self.camera_url.startswith("rtsp://"):
            logging.info("Using RTSP camera stream")
            return "rtsp"

        if self.camera_url.startswith("http://") or self.camera_url.startswith("https://"):
            logging.info("Using MJPEG camera stream")
            return "mjpeg"

        if self.camera_url.endswith(".mp4") or self.camera_url.endswith(".avi"):
            logging.info("Using avi/mp4 video file")
            return "video"

        raise ValueError(f"Unsupported CAMERA_URL format: {self.camera_url}")

    def _gstreamer_pipeline(self):
        if self.camera_type == "rtsp":
            return (
                f'rtspsrc location={self.camera_url} latency=200 ! '
                'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
                'video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
            )
        elif self.camera_type == "mjpeg":
            return (
                f'uridecodebin uri={self.camera_url} ! '
                'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
            )
        elif self.camera_type == "video":
            return (
                f'filesrc location={self.camera_url} ! decodebin ! '
                'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
            )
        elif self.camera_type == "usb":
            device = self.camera_url if isinstance(self.camera_url, str) else f"/dev/video{self.camera_url}"
            return (
                f'v4l2src device={device} ! '
                f'video/x-raw, width={self.cam_x}, height={self.cam_y}, framerate=30/1 ! '
                'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
            )
        else:
            raise ValueError(f"Unsupported camera type: {self.camera_type}")

    def _initialize_camera(self):
        if self.camera_type == "libcamera":
            if not PICAMERA_AVAILABLE:
                raise RuntimeError("camera_type 'libcamera' selected but Picamera2 is not available!")
            if Picamera2 is None or Transform is None:
                raise RuntimeError("Picamera2 modules are not properly loaded!")
            self.picam2 = Picamera2()
            video_cfg = self.picam2.create_video_configuration(
                main={"size": (self.cam_x, self.cam_y), "format": "RGB888"},
                controls={"FrameRate": 6},
                transform=Transform(hflip=self.hflip, vflip=self.vflip)
            )
            self.picam2.configure(video_cfg)
            self.picam2.start()
            time.sleep(2)
            logging.info("PiCamera2 initialized")
            return

        try:
            if config.USE_GSTREAMER:
                pipeline = self._gstreamer_pipeline()
                logging.info(f"Opening camera with GStreamer pipeline: {pipeline}")
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.camera_url)
                if self.camera_type == "usb":
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_x)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_y)
                    self.cap.set(cv2.CAP_PROP_FPS, 6)
                elif self.camera_type == "mjpeg":
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                elif self.camera_type == "rtsp":
                    logging.debug("RTSP stream may need time to buffer. Sleeping briefly…")
                    time.sleep(0.05)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera stream: {self.camera_url}")

            # Prime buffer (GStreamer only)
            if config.USE_GSTREAMER:
                for _ in range(10):
                    self.cap.read()

            logging.info(f"Camera initialized with {'GStreamer' if config.USE_GSTREAMER else 'OpenCV'} backend")

        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            if self.restart_attempts < self.max_restart_attempts:
                self.restart_attempts += 1
                logging.error(f"Restart attempt {self.restart_attempts}/{self.max_restart_attempts}")
                time.sleep(1)
                self._restart_camera()
            else:
                raise RuntimeError(f"Failed to initialize camera after {self.max_restart_attempts} attempts") from e

    def _restart_camera(self):
        logging.warning("Restarting camera…")
        if self.camera_type == "libcamera" and self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.debug("Camera resources released")
        gc.collect()
        self._initialize_camera()
        self.restart_attempts = 0

    def _process_motion_detection(self, frame, prev_gray):
        if frame is None:
            logging.warning("Received empty frame for motion detection!")
            return prev_gray, False

        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False
        motion_pixels = 0

        if prev_gray is not None:
            # Compute absolute difference between current and previous frame
            frame_delta = cv2.absdiff(prev_gray, gray)
            # Threshold the delta image
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            # Dilate to fill in holes, making motion regions more solid
            thresh = cv2.dilate(thresh, None, iterations=2)
            # Count the number of changed pixels
            motion_pixels = cv2.countNonZero(thresh)
            if motion_pixels > self.motion_threshold:
                motion_detected = True
                logging.debug(f"Motion detected: {motion_pixels} changed pixels (threshold: {self.motion_threshold})")
            #else:
                #logging.debug(f"No significant motion: {motion_pixels} changed pixels (threshold: {self.motion_threshold})")
        else:
            logging.debug("No previous frame for motion detection; skipping motion calculation")

        return gray, motion_detected

    def _capture_frame(self):
        if self.camera_type == "libcamera" and self.picam2:
            rgb = self.picam2.capture_array("main")
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.debug("RETURNING None, 'Not ret or frame is None'")
                return None
            if self.hflip or self.vflip:
                code = -1 if self.hflip and self.vflip else (1 if self.hflip else 0)
                frame = cv2.flip(frame, code)
        return frame

    def main_capture_loop(self):
        consec_failures = 0
        i = 0
        self.last_enqueue_time = time.time()
        logging.debug(f"MAIN_CAPTURE_LOOP STARTED in PID {os.getpid()}")
        logging.info(f"Starting queuing loop with {self.sleep_interval:.2f}s between frames")

        # --- Prefill queue and initialize prev_gray ---
        prev_gray = None
        last_frame = None
        num_prefill = self.fps_offset + 1
        logging.debug("Starting PREFILL loop")
        for _ in range(num_prefill):
            frame = self._capture_frame()
            if frame is None:
                logging.error("[PREFILL]: Frame capture failed (frame is None)!")
                consec_failures += 1
                if consec_failures >= self.max_frame_failures:
                    logging.error(f"[PREFILL]: Too many consecutive frame failures ({self.max_frame_failures}), exiting camera process for restart!")
                    sys.exit(13)
                time.sleep(1)
                continue
            else:
                if len(self.q) < self.max_queue_len:
                    #timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%Y_%m_%d_%H-%M-%S.%f")
                    now = datetime.now(config.TIMEZONE_OBJ)
                    timestamp = int(now.timestamp()*1000.0)
                    timestamp_nice = now.strftime("%Y_%m_%d %H-%M-%S.%f")
                    self.q.append((timestamp, frame))
                    self.last_enqueue_time = time.time()
                    last_frame = frame
                    logging.debug(f"[PREFILL]: Enqueued frame at {timestamp_nice} | Queue ID={id(self.q)} length: {len(self.q)}")
                    consec_failures = 0

            time.sleep(self.sleep_interval)
        if last_frame is not None:
            prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            logging.debug("Initialized prev_gray from last prefill frame")

        logging.debug("DONE PREFILL - entering main loop")

        # --- Main capture loop ---
        while not self.shutdown_flag.is_set():
            try:
                # Handle pause event
                if self.pause_event.is_set():
                    pause_secs = self.pause_duration.value
                    logging.info(f"Pausing queue for {pause_secs} seconds and clearing all frames [{len(self.q)}]")
                    if len(self.q):
                        self.q[:] = []
                    slept = 0
                    while slept < pause_secs and not self.shutdown_flag.is_set():
                        time.sleep(min(0.5, pause_secs - slept))
                        slept += 0.5
                    self.pause_event.clear()
                    continue

                # Grab a frame
                frame = self._capture_frame()
                if frame is None:
                    logging.error("Frame capture failed (frame is None) in main loop!")
                    consec_failures += 1
                    if consec_failures >= self.max_frame_failures:
                        logging.error(f"Too many consecutive frame failures ({self.max_frame_failures}), exiting camera process for restart!")
                        sys.exit(13)
                    time.sleep(1)
                    continue
                else:
                    consec_failures = 0

                # Try motion detection
                try:
                    gray, motion_detected = self._process_motion_detection(frame, prev_gray)
                except Exception as e:
                    logging.error(f"Motion detection failed: {e}\n{traceback.format_exc()}")
                    gray, motion_detected = None, False
                prev_gray = gray

                # Motion or heartbeat: queue frame
                heartbeat_due = (time.time() - self.last_enqueue_time) > self.heartbeat_interval
                #timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%Y_%m_%d_%H-%M-%S.%f")
                now = datetime.now(config.TIMEZONE_OBJ)
                timestamp = int(now.timestamp()*1000.0)
                timestamp_nice = now.strftime("%Y_%m_%d %H-%M-%S.%f")

                if motion_detected:
                    # Only enqueue motion frame
                    if len(self.q) < self.max_queue_len:
                        self.q.append((timestamp, frame))
                        self.last_enqueue_time = time.time()
                        logging.debug(f"[MOTION] Enqueued frame at {timestamp_nice} | Queue ID={id(self.q)} length: {len(self.q)}")
                    else:
                        logging.warning(f"Queue is full {self.max_queue_len}, dropping motion frame!")

                elif heartbeat_due:
                    # Only enqueue heartbeat if no motion was detected
                    if len(self.q) < self.max_queue_len:
                        self.q.append((timestamp, frame))
                        self.last_enqueue_time = time.time()
                        logging.info(f"🌙 [HEARTBEAT] Enqueued frame at {timestamp_nice} | Queue ID={id(self.q)} length: {len(self.q)} [quiet]")
                    else:
                        logging.warning(
                            f"### THIS SHOULDN'T HAPPEN ### Queue is full {self.max_queue_len}, dropping heartbeat frame! "
                            f"It means that the queue processing is not working for more than {self.heartbeat_interval}s, "
                            f"or your system is very slow..."
                        )

                # Sleep in small increments to allow shutdown responsiveness
                slept = 0
                while slept < self.sleep_interval and not self.shutdown_flag.is_set():
                    time.sleep(min(0.1, self.sleep_interval - slept))
                    slept += 0.1

                i += 1
                if self.queue_cycles > 0 and i >= self.queue_cycles:
                    logging.info(f"Refreshing camera after {self.queue_cycles} frames")
                    self._restart_camera()
                    i = 0

            except Exception as e:
                logging.error(f"Exception in fill_queue {type(e).__name__}: {e}\n{traceback.format_exc()}")
                self._restart_camera()
                slept = 0
                while slept < 1.0 and not self.shutdown_flag.is_set():
                    time.sleep(min(0.1, 1.0 - slept))
                    slept += 0.1
