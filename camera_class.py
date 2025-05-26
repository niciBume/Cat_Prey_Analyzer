from datetime import datetime
import cv2
import time
import pytz
import logging
import gc
import sys
import config

# Conditionally import Picamera2 if available
try:
    from picamera2 import Picamera2
    from libcamera import controls, Transform
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = Transform = None                        # type: ignore


class Camera:
    def __init__(self, q, camera_url):
        self.q = q
        self.sleep_interval = getattr(config, "SLEEP_INTERVAL", 0.25)
        self.queue_cycles = getattr(config, "FILL_QUEUE_CYCLES", 60)
        self.max_len = getattr(config, "MAX_QUEUE_LEN", 20)
        self.camera_url = camera_url
        self.camera_type = self._detect_camera_type()
        self.cap = None
        self.picam2 = None

        # Camera geometry and flip config
        self.cam_x = getattr(config, "CAM_WIDTH", 640)
        self.cam_y = getattr(config, "CAM_HEIGHT", 480)
        self.flip_overrides = getattr(config, "CAMERA_FLIP_OVERRIDES", {})
        self._load_flip_overrides()

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
        if self.camera_type == "libcamera":
            if not PICAMERA_AVAILABLE:
                raise RuntimeError("camera_type 'libcamera' selected but Picamera2 is not available.")
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
                raise RuntimeError(f"Failed to open stream: {self.camera_url}")
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
        tz = pytz.timezone("Europe/Berlin")
        last_enqueue_time = time.time()

        logging.info(f"Starting queuing loop with {self.sleep_interval:.2f}s between frames ...")
        while True:
            try:
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
                    if self.hflip and self.vflip:
                        frame = cv2.flip(frame, -1)
                    elif self.hflip:
                        frame = cv2.flip(frame, 1)
                    elif self.vflip:
                        frame = cv2.flip(frame, 0)

                now = time.time()
                if now - last_enqueue_time >= self.sleep_interval:
                    timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H-%M-%S.%f")
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

                if time.time() - getattr(self, "_last_debug_log_time", 0) > 0.5:
                    sys.stdout.write(
                        f"\r[DEBUG] last_enqueue_time: {last_enqueue_time:.2f}s | Queue: {len(self.q):2d}   "
                    )
                    sys.stdout.flush()
                    self._last_debug_log_time = time.time()

                time.sleep(0.01)

            except Exception as e:
                logging.error("Exception in fill_queue: %s", e)
                self._restart_camera()
                time.sleep(1)


