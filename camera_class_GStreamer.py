import os
import cv2
import time
import gc
import sys
import config
import multiprocessing
import traceback
import logging
from datetime import datetime

try:
    from picamera2 import Picamera2
    from libcamera import Transform
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = Transform = None                        # type: ignore

class Camera:
    def __init__(self, q, camera_key, shutdown_flag, pause_event=None, pause_duration=None, log_level_str="INFO"):
        self.q = q
        self.camera_key = camera_key
        self.shutdown_flag = shutdown_flag
        self.pause_event = pause_event or multiprocessing.Manager().Event()
        self.pause_duration = pause_duration or multiprocessing.Manager().Value('d', 0.0)
        self.restart_attempts = 0
        self.max_restart_attempts = 5
        self.queue_cycles = getattr(config, "FILL_QUEUE_CYCLES", 60)
        self.fps_offset = getattr(config, "DEFAULT_FPS_OFFSET", 2)
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
        cam_cfg = config.CAMERA_OVERRIDES.get(camera_key, config.CAMERA_OVERRIDES['default'])
        self.base_url = cam_cfg.get('url')
        self.cam_x = cam_cfg.get('cam_width', config.CAMERA_OVERRIDES['default']['cam_width'])
        self.cam_y = cam_cfg.get('cam_height', config.CAMERA_OVERRIDES['default']['cam_height'])
        self.hflip = cam_cfg.get('hflip', config.CAMERA_OVERRIDES['default']['hflip'])
        self.vflip = cam_cfg.get('vflip', config.CAMERA_OVERRIDES['default']['vflip'])

        self.camera_url = self._compose_url_with_creds()
        self.camera_type = self._detect_camera_type()
        logging.info(f"Camera settings: camera_key={self.camera_key}, base_url={self.base_url}, type={self.camera_type}, width={self.cam_x}, height={self.cam_y}, hflip={self.hflip}, vflip={self.vflip}")

    def _compose_url_with_creds(self):
        if not self.base_url:
            return None
        if self.base_url.startswith("rtsp://") or self.base_url.startswith("http://"):
            user = os.getenv(f"{self.camera_key.upper()}_USER")
            pw = os.getenv(f"{self.camera_key.upper()}_PASS")
            if user and pw:
                proto, rest = self.base_url.split("://", 1)
                return f"{proto}://{user}:{pw}@{rest}"
        return self.base_url

    def start_hardware(self):
        self._initialize_camera()

    def _detect_camera_type(self):
        if not self.camera_url:
            if PICAMERA_AVAILABLE:
                logging.info("Using internal PiCamera2.")
                return "libcamera"
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
        else:
            try:
                gst_pipeline = self._gstreamer_pipeline()
                logging.debug(f"Opening camera with GStreamer pipeline: {gst_pipeline}")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera with pipeline: {gst_pipeline}")
                # Prime the pipeline: discard first 10 frames to avoid grey/noise/buffered frames
                for _ in range(10):
                    self.cap.read()
                logging.info(f"GStreamer pipeline opened successfully and primed.")
            except Exception as e:
                logging.error(f"Failed to open camera stream with GStreamer: {e}")
                if self.restart_attempts < self.max_restart_attempts:
                    self.restart_attempts += 1
                    logging.error(f"Restart attempt {self.restart_attempts}/{self.max_restart_attempts}")
                    time.sleep(1)
                    self._restart_camera()
                else:
                    raise RuntimeError(f"Failed to initialize camera after {self.max_restart_attempts} attempts")

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
        self.restart_attempts = 0

    def _process_motion_detection(self, frame, prev_gray):
        if frame is None:
            logging.warning("Received empty frame for motion detection.")
            return prev_gray, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False
        motion_pixels = 0

        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_pixels = cv2.countNonZero(thresh)
            if motion_pixels > self.motion_threshold:
                motion_detected = True
                logging.debug(f"Motion detected: {motion_pixels} changed pixels (threshold: {self.motion_threshold}).")
            else:
                logging.debug(f"No significant motion: {motion_pixels} changed pixels (threshold: {self.motion_threshold}).")
        else:
            logging.debug("No previous frame for motion detection; skipping motion calculation.")

        return gray, motion_detected

    def _capture_frame(self):
        if self.camera_type == "libcamera" and self.picam2:
            rgb = self.picam2.capture_array("main")
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            logging.debug("Reading from cv2.VideoCapture...")
            ret, frame = self.cap.read()
            logging.debug(f"cv2.VideoCapture.read() returned ret={ret}, frame is {'not None' if frame is not None else 'None'}")
            if not ret or frame is None:
                logging.debug("RETURNING None, 'Not ret or frame is None'")
                return None
            if self.hflip or self.vflip:
                code = -1 if self.hflip and self.vflip else (1 if self.hflip else 0)
                frame = cv2.flip(frame, code)
        return frame

    def main_capture_loop(self):
        consec_failures = 0
        MAX_FRAME_FAILURES = 5

        i = 0
        self.last_heartbeat_enqueue_time = time.time()
        logging.debug(f"MAIN_CAPTURE_LOOP STARTED in PID {os.getpid()}")
        logging.info(f"Starting queuing loop with {self.sleep_interval:.2f}s between frames ...")

        prev_gray = None
        last_frame = None
        num_prefill = self.fps_offset + 1
        logging.debug("Starting PREFILL loop")
        for idx in range(num_prefill):
            frame = self._capture_frame()
            if frame is None:
                logging.error("[PREFILL]: Frame capture failed (frame is None)!")
                consec_failures += 1
                if consec_failures >= MAX_FRAME_FAILURES:
                    logging.error(f"[PREFILL]: Too many consecutive frame failures ({MAX_FRAME_FAILURES}), exiting camera process for restart.")
                    sys.exit(13)
                time.sleep(1)
                continue
            else:
                if len(self.q) < self.max_queue_len:
                    timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%Y_%m_%d_%H-%M-%S.%f")
                    self.q.append((timestamp, frame))
                    last_frame = frame
                    logging.debug(f"[PREFILL]: Enqueued frame at {timestamp} | Queue ID={id(self.q)} length: {len(self.q)}")
                    consec_failures = 0

            time.sleep(self.sleep_interval)
        if last_frame is not None:
            prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            logging.debug("Initialized prev_gray from last prefill frame.")

        logging.debug("DONE PREFILL - entering main loop")

        while not self.shutdown_flag.is_set():
            logging.debug(f"Queue type: {type(self.q)}, queue id: {id(self.q)}, length: {len(self.q)}")
            try:
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

                now = time.time()
                frame = self._capture_frame()
                if frame is None:
                    logging.error("Frame capture failed (frame is None) in main loop.")
                    consec_failures += 1
                    if consec_failures >= MAX_FRAME_FAILURES:
                        logging.error(f"Too many consecutive frame failures ({MAX_FRAME_FAILURES}), exiting camera process for restart.")
                        sys.exit(13)
                    time.sleep(1)
                    continue
                else:
                    consec_failures = 0
                logging.debug(f"Enqueued frame at {datetime.now(config.TIMEZONE_OBJ).strftime('%Y_%m_%d_%H-%M-%S.%f')} | Queue ID={id(self.q)} length: {len(self.q)}")

                try:
                    gray, motion_detected = self._process_motion_detection(frame, prev_gray)
                except Exception as e:
                    logging.error(f"Motion detection failed: {e}\n{traceback.format_exc()}")
                    gray, motion_detected = None, False
                prev_gray = gray

                logging.debug(f"Motion detected: {motion_detected}")

                timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%Y_%m_%d_%H-%M-%S.%f")
                if motion_detected:
                    if len(self.q) < self.max_queue_len:
                        self.q.append((timestamp, frame))
                        logging.info(f"[MOTION] Enqueued frame at {timestamp} | Queue ID={id(self.q)} length: {len(self.q)}")
                    else:
                        logging.warning(f"Queue is full {self.max_queue_len}, dropping frame.")
                elif (now - self.last_heartbeat_enqueue_time) > self.heartbeat_interval:
                    if len(self.q) < self.max_queue_len:
                        self.q.append((timestamp, frame))
                        logging.info(f"🌙 [HEARTBEAT] Enqueued frame at {timestamp} | Queue ID={id(self.q)} length: {len(self.q)} [quiet]")
                    else:
                        logging.warning(f"Queue is full {self.max_queue_len}, dropping frame.")
                    self.last_heartbeat_enqueue_time = now

                logging.debug(f"Queue IDs: {[id(f) for _, f in self.q]}, queue length={len(self.q)}")

                slept = 0
                while slept < self.sleep_interval and not self.shutdown_flag.is_set():
                    time.sleep(min(0.1, self.sleep_interval - slept))
                    slept += 0.1

                i += 1
                if self.queue_cycles > 0 and i >= self.queue_cycles:
                    logging.info(f"Refreshing camera after {self.queue_cycles} frames.")
                    self._restart_camera()
                    i = 0

            except Exception as e:
                logging.error(f"Exception in fill_queue {type(e).__name__}: {e}\n{traceback.format_exc()}")
                self._restart_camera()
                slept = 0
                while slept < 1.0 and not self.shutdown_flag.is_set():
                    time.sleep(min(0.1, 1.0 - slept))
                    slept += 0.1
