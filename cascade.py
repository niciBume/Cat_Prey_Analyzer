# cascade.py

"""
Cat Prey Analyzer - Main Application Orchestration and Analysis Pipeline

- Core Controller:
  - Orchestrates camera acquisition, frame analysis, event detection, and bot/user integration.
  - Manages the main queue handler, which processes frames as they become available from the camera queue.
  - Runs detection algorithms (e.g., prey detection, cat recognition) and triggers corresponding actions or notifications.

- Bot Integration:
  - Handles Telegram (or other) bot commands, such as `/sendlivepic`, status queries, and user-initiated actions.
  - Pulls latest frames from the main queue for live image requests, ensuring frames are always recent due to camera_class logic.

- Catflap & Peripheral Control:
  - Integrates with Sure Petcare (Surepy) and/or Home Assistant for smart catflap control.
  - Pauses/resumes camera queue and manages lock state during catflap operations.

- Startup & Fault Tolerance:
  - Initializes all components, launches threads for camera and bot.
  - Monitors and recovers from errors in camera or analysis loops.

- Logging:
  - Provides info, warning, and error logs for all major operations.
  - Reports status to the bot for user transparency.

- This file is the main entry point and nervous system of the analyzer, connecting all subsystems and user interfaces.
"""

import config
import sys, gzip, shutil, os, cv2, time, csv, telegram, requests, argparse, asyncio, aiohttp, pydantic, pprint, json, jwt, subprocess
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Event, Lock, Thread
from multiprocessing import Process
from telegram.ext import Updater, CommandHandler
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, List
from model_stages import PC_Stage, FF_Stage, Eye_Stage, Haar_Stage, CC_MobileNet_Stage
from camera_class import Camera
from surepy import Surepy
from surepy.enums import LockState
from surepy.entities.devices import Flap

# Set up argument parser
parser = argparse.ArgumentParser(
description="""\
Cat Prey Analyzer - Smart Cat Flap Monitor

This tool uses camera input and machine learning to detect
whether a cat is bringing prey and manage homeassistant catflap control.
It communicates with the user and can be controlled through telegram app.

Create a [hidden] .src file and 'source' it before firing cascade.py,
containing your secrets [edit config.py].
You can also tweak the rest of the values there for better performance.

""",
    formatter_class=argparse.RawTextHelpFormatter
)

CAMERA_URL = None
parser.add_argument("--log", default="INFO", help="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)")
parser.add_argument(
        "--camera-url",
        type=str,
        help="""Set camera input source:
          - libcamera (default, if parameter not set)
          - MJPEG stream (http://...)
          - RTSP stream (rtsp://...)
          - USB webcam (CAMERA_URL is digit)
          - Video file (if URL is a file, ending in avi/mp4)
        """,
        )

args = parser.parse_args()
if args.camera_url:
    CAMERA_URL = args.camera_url

# Create a RotatingFileHandler
log_handler = RotatingFileHandler(
    config.LOG_FILENAME, maxBytes=config.MAX_LOG_SIZE, backupCount=config.BACKUP_COUNT
)

# Optional: compress old log files after rotation
class GzipRotator:
    def __call__(self, source, dest):
        with open(source, 'rb') as f_in, gzip.open(dest + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(source)

log_handler.rotator = GzipRotator()
log_handler.namer = lambda name: name

# Set format and log level
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                              datefmt='%x-%X')
log_handler.setFormatter(formatter)

# Apply to root logger
logger = logging.getLogger()

# Parse log level dynamically from --log
log_level_str = args.log.upper()
log_level = getattr(logging, log_level_str, None)
if not isinstance(log_level, int):
    raise ValueError(f"Invalid log level: {args.log}")
logger.setLevel(log_level)

logger.addHandler(log_handler)

logging.info('\n\n   ##### Starting CatPreyAnalyzer #####   \n')
logging.info('Configured logging.')
logging.info(f"  Rotating log when it grows bigger than {config.MAX_LOG_SIZE/1024/1024} MB")

if CAMERA_URL:
    logging.info("Using following CAMERA_URL: %s", CAMERA_URL)

import asyncio

# ── Helper to know whether to try Surepy at all ──
def use_surepy():
    """Return True if Surepy is configured and client/ID are set."""
    required_attrs = ["SP_DEVICE_ID", "SP_EMAIL", "SP_PASSWORD"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            logging.debug(f"❌ Surepy config missing attribute: {attr}")
            return False
        if getattr(config, attr) in (None, "", 0):
            logging.debug(f"❌ Surepy config attribute {attr} is empty or zero.")
            return False
    return True

# ── Helper to know whether to try hassio at all ──
def use_ha():
    """Return True if all HA config attributes are set."""
    required_attrs = ["HA_UNLOCK_WEBHOOK", "HA_LOCK_OUT_WEBHOOK", "HA_LOCK_ALL_WEBHOOK", "HA_REST_URL"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            logging.debug(f"❌ HA config missing attribute: {attr}")
            return False
        if not getattr(config, attr):
            logging.debug(f"❌ HA config attribute {attr} is empty.")
            return False
    return True

bot_instance = None
cat_cam_py = str(Path(os.getcwd()).parents[0])
logging.debug('CatCamPy: %s', cat_cam_py)
logging.info(f"Using {config.TIMEZONE_OBJ} as timezone")
USE_SUREPET = True if (use_surepy() or use_ha()) else False

if USE_SUREPET:
    logging.info(f"Use surepy module for locking")
else:
    logging.info(f"Use Homeassistant for locking")


class Sequential_Cascade_Feeder():
    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), 'log')
        logging.debug('Log Dir: %s', self.log_dir)
        self.event_nr = 0
        self.base_cascade = Cascade()
        self.fps_offset = getattr(config, "DEFAULT_FPS_OFFSET", 2)
        self.MAX_PROCESSES = 7
        self.EVENT_FLAG = False
        self.event_objects = []
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.event_reset_threshold = 6
        self.event_reset_counter = 0
        self.cumulus_points = 0
        self.cumulus_prey_threshold = -10
        self.cumulus_no_prey_threshold = 2.9603
        self.prey_val_hard_threshold = 0.6
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.queues_cumuli_in_event = []
        self.bot = NodeBot()
        self.processing_pool = []
        self.max_queue_len = getattr(config, "MAX_QUEUE_LEN", 20)
        self.main_deque = deque(maxlen=self.max_queue_len)
        self.camera_url = CAMERA_URL
        self.surepy_client: Optional[Surepy] = None
        self.device_cache: Optional[Flap] = None
        self.surepy_client = None

    def pause_camera_for(self, open_time: int):
        if hasattr(self, "camera"):
            with self.camera._pause_lock:
                self.camera.pause_duration = max(0.0, float(open_time - 1))
                logging.debug(f"ℹ️  Pausing camera queue for {self.camera.pause_duration:.2f}s")
            self.camera.pause_event.set()

    # ── Retry wrapper for async Surepy calls ──
    async def try_surepy_with_retries(self, coro_fn, description, retries=2, delay=2):
        """Retry async surepy state-changing calls with retries."""
        for attempt in range(1, retries + 1):
            try:
                result = await coro_fn()
                logging.debug(f"{description} (attempt {attempt}) succeeded.")
                return result
            except Exception as e:
                logging.warning(f"{description} (attempt {attempt}) failed: {e}")
                await asyncio.sleep(delay)
        logging.error(f"{description} failed after {retries} attempts.")
        return False

    # ── HTTP GET with retries (for HA fallback) ──
    def try_get_with_retries(self, url, headers, description="", retries=2, timeout=2):
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                logging.debug(f"{description} (attempt {attempt}) succeeded.")
                return resp
            except requests.RequestException as e:
                logging.warning(f"{description} (attempt {attempt}) failed: {e}")
                time.sleep(1)
        logging.error(f"{description} failed after {retries} attempts.")
        return None

    # ── HTTP POST with retries (for HA fallback) ──
    def try_post_with_retries(self, url, description, retries=2, timeout=2):
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    logging.debug(f"ℹ️  {description} (attempt {attempt}) succeeded.")
                    return True
                logging.warning(f"{description} (attempt {attempt}) failed with status {resp.status_code}.")
            except requests.RequestException as e:
                logging.warning(f"{description} (attempt {attempt}) failed: {e}")
            time.sleep(1)
        logging.error(f"{description} failed after {retries} attempts.")
        return False

    # ── Home Assistant flow ──
    def ha_flow(self, open_time: int):
        self.pause_camera_for(open_time)
        headers = {
            "Authorization": f"Bearer {config.HA_REST_TOKEN}",
            "content-type": "application/json"
        }
        response = self.try_get_with_retries(config.HA_REST_URL, headers, "Query HA catflap state")
        if not response:
            self.bot.send_text("❌ Could not query HA catflap state – aborting.")
            return False

        try:
            ha_state = response.json().get("state")
            if not ha_state:
                raise ValueError("⚠️  No 'state' in HA response JSON")
        except Exception as e:
            self.bot.send_text(f"❌ Failed to decode HA state: {e}")
            return False

        open_states = {"unlocked", "locked_in"}
        if ha_state in open_states:
            self.bot.send_text(f'ℹ️  Catflap already open inwards: [{ha_state}]')
            return True  # Nothing to do, treat as success

        closed_states = {"locked_out", "locked_all"}
        if ha_state in closed_states:
            if self.try_post_with_retries(config.HA_UNLOCK_WEBHOOK, "Unlock catflap"):
                self.bot.send_text(f'ℹ️  Catflap was [{ha_state}], unlocking for {open_time}s.')
                time.sleep(open_time)
                lock_url = (config.HA_LOCK_OUT_WEBHOOK
                            if ha_state == "locked_out"
                            else config.HA_LOCK_ALL_WEBHOOK)
                if self.try_post_with_retries(lock_url, f"ℹ️  Re-lock catflap to [{ha_state}]"):
                    self.bot.send_text(f'ℹ️  Catflap is back to previous state: [{ha_state}].')
                    return True
                else:
                    self.bot.send_text("❌ Error re-locking HA catflap.")
                    return False
            else:
                self.bot.send_text("❌ Failed to unlock HA catflap.")
                return False
        else:
            self.bot.send_text(f'⚠️  Unknown Home Assistant catflap state: [{ha_state}]')
            return False

    # ── Surepy flow ──
    async def surepy_flow(self, open_time: int):
        self.pause_camera_for(open_time)
        # 1. Get flap state
        state = await self.get_catflap_state_surepy()
        if state is None:
            self.bot.send_text("❌ Could not get state from Sure Petcare.")
            return False  # Explicit failure

        open_states = {"unlocked", "locked_in"}
        if state in open_states:
            self.bot.send_text(f'ℹ️  Catflap already open inwards: [{state}]')
            return True  # Consider this a success, nothing to do

        closed_states = {"locked_out", "locked_all"}
        if state in closed_states:
            # Try unlocking with retries
            unlock_fn = lambda: self.set_catflap_lock_state_surepy("unlocked")
            if await self.try_surepy_with_retries(unlock_fn, "Unlock catflap via Surepy"):
                self.bot.send_text(f'ℹ️  Catflap was [{state}], unlocking for {open_time}s.')
                await asyncio.sleep(open_time)
                # Restore original state with retries
                relock_fn = lambda: self.set_catflap_lock_state_surepy(state)
                if await self.try_surepy_with_retries(relock_fn, f"Re-lock catflap to [{state}] via Surepy"):
                    self.bot.send_text(f'ℹ️  Catflap is back to previous state: [{state}].')
                    return True   # <-- ADD THIS for successful sequence
                else:
                    self.bot.send_text("❌ Error re-locking catflap via Sure Petcare.")
                    return False
            else:
                self.bot.send_text("❌ Failed to unlock catflap via Sure Petcare.")
                return False
        else:
            self.bot.send_text(f'⚠️  Unknown Sure Petcare catflap state: [{state}]')
            return False

    def control_catflap(self, open_time: int = 30):
        if use_surepy():
            try:
                result = asyncio.run(self.surepy_flow(open_time))
                if not result and use_ha():
                    logging.warning("❌ Surepy failed, falling back to HA flow.")
                    self.ha_flow(open_time)
                elif not result:
                    logging.error("❌ Both Surepy and HA failed or are not configured.")
            except Exception as e:
                logging.error(f"❌ Unexpected error in Surepy flow: {e}\nFalling back to HA flow.")
                if use_ha():
                    self.ha_flow(open_time)
                else:
                    logging.error("❌ Both Surepy and HA failed or are not configured.")
        elif use_ha():
            self.ha_flow(open_time)
        else:
            logging.error("❌ No catflap integration (Surepy or HA) configured!")

    # ── Lazy-initialize a Surepy client ──
    def get_surepy_client(self) -> Surepy:
        if self.surepy_client is None:
            logging.debug("🔐 Initializing Surepy client…")
            self.surepy_client = Surepy(
                email=config.SP_EMAIL,
                password=config.SP_PASSWORD
            )
            logging.debug("ℹ️  Done initializing Surepy client…")
        return self.surepy_client

    # ── Fetch (and cache) the Flap device object ──
    async def _fetch_device(self) -> Optional[Flap]:
        logging.debug("🚪️ Fetching catflap device from Surepy..")
        try:
            client = self.get_surepy_client()
            devices: List = await client.get_devices()
            target_id = str(config.SP_DEVICE_ID)
            device = next((d for d in devices if str(d.id) == target_id), None)

            if device is None:
                logging.error(f"❌ Device ID {config.SP_DEVICE_ID} not found among Surepy devices.")
                return None

            self.device_cache = device
            return device
            logging.debug("ℹ️   Done fetching catflap device from Surepy..")

        except Exception as e:
            logging.error(f"❌ Error fetching device from Surepy: {e}")
            return None

    # ── Ask Surepy for the current lock‐state string (“lock_out”, etc.) ──
    async def get_catflap_state_surepy(self) -> Optional[str]:
        try:
            device = await self._fetch_device()
            if device is None:
                logging.error("❌ No Surepy device available to read lock state.")
                return None
            logging.debug("❔ Fetching device status from Surepy..")

            self.mode = str(device.state).lower()
            logging.info(f"🐾 Surepy catflap_state = {self.mode}")
            logging.debug("ℹ️  Done fetching device status from Surepy..")
            return self.mode

        except Exception as e:
            logging.error(f"❌ Surepy error while getting state: {e}")
            return None

    # ── Tell Surepy to change the lock state of the flap ──
    async def set_catflap_lock_state_surepy(self, state: str) -> bool:
        """Set lock state using string state, as sureha does."""
        try:
            client = self.get_surepy_client()
            device = await self._fetch_device()
            if device is None:
                logging.error("❌ Could not fetch catflap device.")
                return False

            lock_states = {
                "unlocked": client.sac.unlock,
                "locked_in": client.sac.lock_in,
                "locked_out": client.sac.lock_out,
                "locked_all": client.sac.lock,
            }

            state = state.lower()
            if state not in lock_states:
                logging.error(f"ℹ️  Unknown lock state '{state}'")
                return False

            await lock_states[state](config.SP_DEVICE_ID)
            logging.info(f"ℹ️  Set lock state to '{state}' via surepy.sac")
            return True

        except Exception as e:
            logging.error(f"❌ Surepy error setting lock state [{state}]: {e}")
            return False

    def reset_cumuli_et_al(self):
        self.EVENT_FLAG = False
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.cumulus_points = 0
        self.event_reset_counter = 0
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.cumulus_points = 0

        #Close the node_letin flag
        self.bot.node_let_in_flag = False

        self.event_objects.clear()
        self.queues_cumuli_in_event.clear()
        self.main_deque.clear()

        #terminate processes when pool too large
        if len(self.processing_pool) >= self.MAX_PROCESSES:
            logging.debug('terminating oldest processes Len: %d', len(self.processing_pool))
            for p in self.processing_pool[0:int(len(self.processing_pool)/2)]:
                p.terminate()
            logging.debug('Now processes Len: %d', len(self.processing_pool))

    def log_event_to_csv(self, event_obj, queues_cumuli_in_event, event_nr):
        csv_name = 'event_log.csv'
        file_exists = os.path.isfile(os.path.join(self.log_dir, csv_name))
        with open(os.path.join(self.log_dir, csv_name), mode='a') as csv_file:
            headers = ['Event', 'Img_Name', 'Done_Time', 'Queue', 'Cumuli', 'CC_Cat_Bool', 'CC_Time', 'CR_Class', 'CR_Val', 'CR_Time', 'BBS_Time', 'HAAR_Time', 'FF_BBS_Bool', 'FF_BBS_Val', 'FF_BBS_Time', 'Face_Bool', 'PC_Class', 'PC_Val', 'PC_Time', 'Total_Time']
            writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            for i,img_obj in enumerate(event_obj):
                writer.writerow({'Event':event_nr, 'Img_Name':img_obj.img_name, 'Done_Time':queues_cumuli_in_event[i][2],
                                 'Queue':queues_cumuli_in_event[i][0],
                                 'Cumuli':queues_cumuli_in_event[i][1],'CC_Cat_Bool':img_obj.cc_cat_bool,
                                 'CC_Time':img_obj.cc_inference_time, 'CR_Class':img_obj.cr_class,
                                 'CR_Val':img_obj.cr_val, 'CR_Time':img_obj.cr_inference_time,
                                 'BBS_Time':img_obj.bbs_inference_time,
                                 'HAAR_Time':img_obj.haar_inference_time, 'FF_BBS_Bool':img_obj.ff_bbs_bool,
                                 'FF_BBS_Val':img_obj.ff_bbs_val, 'FF_BBS_Time':img_obj.ff_bbs_inference_time,
                                 'Face_Bool':img_obj.face_bool,
                                 'PC_Class':img_obj.pc_prey_class, 'PC_Val':img_obj.pc_prey_val,
                                 'PC_Time':img_obj.pc_inference_time, 'Total_Time':img_obj.total_inference_time})

    def send_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        max_prey_index = prey_vals.index(max(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logging.debug('Img_Name: %s', f_event.img_name)
            logging.debug('PC_Val: %.2f', f_event.pc_prey_val)
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[max_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => PREY IN DA HOUSE!' + ' ⚠️  🐁' + event_str
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def send_no_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        min_prey_index = prey_vals.index(min(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logging.debug('Img_Name: %s', f_event.img_name)
            logging.debug('PC_Val: %.2f', f_event.pc_prey_val)
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[min_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => No prey, cat is clean...' + ' ✅️ 🐱' + event_str
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def send_dk_message(self, event_objects, cumuli):
        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logging.debug('Img_Name: %s', f_event.img_name)
            logging.debug('PC_Val: %.2f', f_event.pc_prey_val)
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = face_events[0].output_img
        caption = 'Cumuli: ' + str(cumuli) + '❔️ => Cant say for sure...' + ' 🤷' + event_str + '\nMaybe use /letin?'
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def get_event_nr(self):
        tree = ET.parse(os.path.join(self.log_dir, 'info.xml'))
        data = tree.getroot()
        imgNr = int(data.find('node').get('imgNr'))
        data.find('node').set('imgNr', str(int(imgNr) + 1))
        tree.write(os.path.join(self.log_dir, 'info.xml'))

        return imgNr

    def queue_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()

        # Pass self.main_deque to the camera
        self.camera = Camera(self.main_deque, self.camera_url)

        # Start the camera fill loop
        camera_thread = Thread(target=self.camera.fill_queue, daemon=True)
        camera_thread.start()
        self.bot.send_text(message='ℹ️  Starting camera loop')

        while(True):
            if len(self.main_deque) > self.fps_offset:
                logging.debug("Deque type: %s | Length: %d", type(self.main_deque), len(self.main_deque))
                self.queue_worker()
            else:
                #logging.debug('Nothing to work with => Queue_length: %d', len(self.main_deque))
                time.sleep(0.15)

            #Check if user force opens the door
            if self.bot.node_let_in_flag and USE_SUREPET:
                logging.info("ℹ️  Temporary unlocking the catflap on user's behalf.")
                self.bot.send_text(message="ℹ️  Temporary unlocking the catflap on user's behalf.")
                self.control_catflap(open_time = 40)
                self.reset_cumuli_et_al()

    def queue_worker(self):
        logging.debug("Working the Queue with len: %d", len(self.main_deque))
        start_time = time.time()

        #Feed the latest image in the Queue through the cascade
        timestamp, frame = self.main_deque[self.fps_offset]
        cascade_obj = self.feed(target_img=frame, img_name=timestamp)[1]
        logging.debug('Runtime: %.2f seconds', time.time() - start_time)
        done_timestamp = datetime.now(config.TIMEZONE_OBJ).strftime("%c.%f")
        logging.debug('Timestamp at Done Runtime: %s', done_timestamp)

        overhead = datetime.strptime(done_timestamp, "%c.%f") - datetime.strptime(timestamp, "%c.%f")
        logging.debug('Overhead: %.2f seconds', overhead.total_seconds())

        #Add this such that the bot has some info
        self.bot.node_queue_info = len(self.main_deque)
        self.bot.node_live_img = frame
        self.bot.node_over_head_info = overhead.total_seconds()

        # Always delete the left part
        for _ in range(self.fps_offset + 1):
            if self.main_deque:
                self.main_deque.popleft()

        if cascade_obj.cc_cat_bool == True:
            #We are inside an event => add event_obj to list
            self.EVENT_FLAG = True
            self.event_nr = self.get_event_nr()
            self.event_objects.append(cascade_obj)

            #Last cat pic for bot
            self.bot.node_last_casc_img = cascade_obj.output_img

            self.fps_offset = 0
            #If face found add the cumulus points
            if cascade_obj.face_bool:
                self.face_counter += 1
                self.cumulus_points += (50 - int(round(100 * cascade_obj.pc_prey_val)))
                self.FACE_FOUND_FLAG = True

            logging.debug('CUMULUS: %d', self.cumulus_points)
            self.queues_cumuli_in_event.append((len(self.main_deque),self.cumulus_points, done_timestamp))

            #Check the cumuli points and set flags if necessary
            if self.face_counter > 0 and self.PATIENCE_FLAG:
                if self.cumulus_points / self.face_counter > self.cumulus_no_prey_threshold:
                    self.NO_PREY_FLAG = True
                    logging.info('NO PREY DETECTED... YOU CLEAN...')
                    p = Process(target=self.send_no_prey_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    if USE_SUREPET:
                        logging.info('😸️ Cat is clean, unlocking the catflap temporarily')
                        self.bot.send_text(message='😸️ Cat is clean, unlocking the catflap temporarily')
                        self.control_catflap(open_time = 60)
                    self.reset_cumuli_et_al()
                elif self.cumulus_points / self.face_counter < self.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logging.info('IT IS A PREY!!!!!')
                    p = Process(target=self.send_prey_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    self.reset_cumuli_et_al()
                else:
                    self.NO_PREY_FLAG = False
                    self.PREY_FLAG = False

            #Cat was found => still belongs to event => acts as dk state
            self.event_reset_counter = 0

        #No cat detected => reset event_counters if necessary
        else:
            logging.debug('NO CAT FOUND!')
            self.event_reset_counter += 1
            if self.event_reset_counter >= self.event_reset_threshold:
                # If was True => event now over => clear queue
                if self.EVENT_FLAG:
                    logging.debug('CLEARED QUEQUE BECAUSE EVENT OVER WITHOUT CONCLUSION...')
                    #TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    p = Process(target=self.send_dk_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                self.reset_cumuli_et_al()

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2 or self.face_counter > 1:
            self.PATIENCE_FLAG = True

    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        target_img = cv2.imread(os.path.join(cat_cam_py, 'CatPreyAnalyzer/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg'))
        cascade_obj = self.feed(target_img=target_img, img_name=target_img_name)[1]
        logging.debug('Runtime: %.2f seconds', time.time() - start_time)
        return cascade_obj

    def feed(self, target_img, img_name):
        target_event_obj = Event_Element(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        single_cascade = self.base_cascade.do_single_cascade(event_img_object=target_event_obj)
        single_cascade.total_inference_time = sum(filter(None, [
            single_cascade.cc_inference_time,
            single_cascade.cr_inference_time,
            single_cascade.bbs_inference_time,
            single_cascade.haar_inference_time,
            single_cascade.ff_bbs_inference_time,
            single_cascade.ff_haar_inference_time,
            single_cascade.pc_inference_time]))
        total_runtime = time.time() - start_time
        logging.debug('Total Runtime: %.2f seconds', total_runtime)

        return total_runtime, single_cascade

class Event_Element():
    def __init__(self, img_name, cc_target_img):
        self.img_name = img_name
        self.cc_target_img = cc_target_img
        self.cc_cat_bool = None
        self.cc_pred_bb = None
        self.cc_inference_time = None
        self.cr_class = None
        self.cr_val = None
        self.cr_inference_time = None
        self.bbs_target_img = None
        self.bbs_pred_bb = None
        self.bbs_inference_time = None
        self.haar_pred_bb = None
        self.haar_inference_time = None
        self.ff_haar_bool = None
        self.ff_haar_val = None
        self.ff_haar_inference_time = None
        self.ff_bbs_bool = None
        self.ff_bbs_val = None
        self.ff_bbs_inference_time = None
        self.face_box = None
        self.face_bool = None
        self.pc_prey_class = None
        self.pc_prey_val = None
        self.pc_inference_time = None
        self.total_inference_time = None
        self.output_img = None

class Cascade:
    def __init__(self):
        # Models
        self.cc_mobile_stage = CC_MobileNet_Stage()
        self.pc_stage = PC_Stage()
        self.ff_stage = FF_Stage()
        self.eyes_stage = Eye_Stage()
        self.haar_stage = Haar_Stage()

    def do_single_cascade(self, event_img_object):
        logging.debug(event_img_object.img_name)
        cc_target_img = event_img_object.cc_target_img
        original_copy_img = cc_target_img.copy()

        #Do CC
        start_time = time.time()
        dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, cc_inference_time = self.do_cc_mobile_stage(cc_target_img=cc_target_img)
        logging.debug('CC_Do Time: %.2f seconds', time.time() - start_time)
        event_img_object.cc_cat_bool = cat_bool
        event_img_object.cc_pred_bb = pred_cc_bb_full
        event_img_object.bbs_target_img = bbs_target_img
        event_img_object.cc_inference_time = cc_inference_time

        if cat_bool and bbs_target_img.size != 0:
            logging.info('Cat Detected!')
            rec_img = self.cc_mobile_stage.draw_rectangle(img=original_copy_img, box=pred_cc_bb_full, color=(255, 0, 0), text='CC_Pred')

            #Do HAAR
            haar_snout_crop, haar_bbs, haar_inference_time, haar_found_bool = self.do_haar_stage(target_img=bbs_target_img, pred_cc_bb_full=pred_cc_bb_full, cc_target_img=cc_target_img)
            rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=haar_bbs, color=(0, 255, 255), text='HAAR_Pred')

            event_img_object.haar_pred_bb = haar_bbs
            event_img_object.haar_inference_time = haar_inference_time

            if haar_found_bool and haar_snout_crop.size != 0 and self.cc_haar_overlap(cc_bbs=pred_cc_bb_full, haar_bbs=haar_bbs) >= 0.1:
                inf_bb = haar_bbs
                face_bool = True
                snout_crop = haar_snout_crop

            else:
                # Do EYES
                bbs_snout_crop, bbs, eye_inference_time = self.do_eyes_stage(eye_target_img=bbs_target_img,
                                                                             cc_pred_bb=pred_cc_bb_full,
                                                                             cc_target_img=cc_target_img)
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=bbs, color=(255, 0, 255), text='BBS_Pred')
                event_img_object.bbs_pred_bb = bbs
                event_img_object.bbs_inference_time = eye_inference_time

                # Do FF for Haar and EYES
                bbs_dk_bool, bbs_face_bool, bbs_ff_conf, bbs_ff_inference_time = self.do_ff_stage(snout_crop=bbs_snout_crop)
                event_img_object.ff_bbs_bool = bbs_face_bool
                event_img_object.ff_bbs_val = bbs_ff_conf
                event_img_object.ff_bbs_inference_time = bbs_ff_inference_time

                inf_bb = bbs
                face_bool = bbs_face_bool
                snout_crop = bbs_snout_crop

            event_img_object.face_bool = face_bool
            event_img_object.face_box = inf_bb

            if face_bool:
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=inf_bb, color=(255, 255, 255), text='INF_Pred')
                logging.info('Face Detected!')

                #Do PC
                pred_class, pred_val, inference_time = self.do_pc_stage(pc_target_img=snout_crop)
                logging.debug(f'Prey Prediction: {pred_class}')
                logging.debug('Pred_Val: %.2f', pred_val)
                pc_str = ' PC_Pred: ' + str(pred_class) + ' @ ' + str('%.2f' % pred_val)
                color = (0, 0, 255) if pred_class else (0, 255, 0)
                rec_img = self.input_text(img=rec_img, text=pc_str, text_pos=(15, 100), color=color)

                event_img_object.pc_prey_class = pred_class
                event_img_object.pc_prey_val = pred_val
                event_img_object.pc_inference_time = inference_time

            else:
                logging.info('No Face Found...')
                ff_str = 'No_Face'
                rec_img = self.input_text(img=rec_img, text=ff_str, text_pos=(15, 100), color=(255, 255, 0))

        else:
            logging.debug('No Cat Found...')
            rec_img = self.input_text(img=original_copy_img, text='CC_Pred: NoCat', text_pos=(15, 100), color=(255, 255, 0))

        #Always save rec_img in event_img object
        event_img_object.output_img = rec_img
        return event_img_object

    def cc_haar_overlap(self, cc_bbs, haar_bbs):
        cc_area = abs(cc_bbs[0][0] - cc_bbs[1][0]) * abs(cc_bbs[0][1] - cc_bbs[1][1])
        haar_area = abs(haar_bbs[0][0] - haar_bbs[1][0]) * abs(haar_bbs[0][1] - haar_bbs[1][1])
        overlap = haar_area / cc_area
        logging.debug('Overlap: %s', overlap)
        return overlap

    def infere_snout_crop(self, bbs, haar_bbs, bbs_face_bool, bbs_ff_conf, haar_face_bool, haar_ff_conf, cc_target_img):
        #Combine BBS's if both are faces
        if bbs_face_bool and haar_face_bool:
            xmin = min(bbs[0][0], haar_bbs[0][0])
            ymin = min(bbs[0][1], haar_bbs[0][1])
            xmax = max(bbs[1][0], haar_bbs[1][0])
            ymax = max(bbs[1][1], haar_bbs[1][1])
            inf_bb = np.array([(xmin,ymin), (xmax,ymax)]).reshape((-1, 2))
            snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
            return snout_crop, inf_bb, False, True, (bbs_ff_conf + haar_ff_conf)/2

        #When they are different choose the one that is true, if none is true than there is no face
        else:
            if bbs_face_bool:
                xmin = bbs[0][0]
                ymin = bbs[0][1]
                xmax = bbs[1][0]
                ymax = bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, bbs_ff_conf
            elif haar_face_bool:
                xmin = haar_bbs[0][0]
                ymin = haar_bbs[0][1]
                xmax = haar_bbs[1][0]
                ymax = haar_bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, haar_ff_conf
            else:
                ff_conf = (bbs_ff_conf + haar_ff_conf)/2 if haar_face_bool else bbs_ff_conf
                return None, None, False, False, ff_conf

    def calc_iou(self, gt_bbox, pred_bbox):
        (x_topleft_gt, y_topleft_gt), (x_bottomright_gt, y_bottomright_gt) = gt_bbox.tolist()
        (x_topleft_p, y_topleft_p), (x_bottomright_p, y_bottomright_p) = pred_bbox.tolist()

        if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if (x_bottomright_gt < x_topleft_p):# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
            return 0.0
        if (y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
            return 0.0
        if (x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
            return 0.0
        if (y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
            return 0.0

        GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
        Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

        x_top_left = np.max([x_topleft_gt, x_topleft_p])
        y_top_left = np.max([y_topleft_gt, y_topleft_p])
        x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
        y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

        return intersection_area / union_area

    def do_cc_mobile_stage(self, cc_target_img):
        pred_cc_bb_full, cat_bool, inference_time = self.cc_mobile_stage.do_cc(target_img=cc_target_img)
        dk_bool = False if cat_bool is True else True
        if cat_bool:
            bbs_xmin = pred_cc_bb_full[0][0]
            bbs_ymin = pred_cc_bb_full[0][1]
            bbs_xmax = pred_cc_bb_full[1][0]
            bbs_ymax = pred_cc_bb_full[1][1]
            bbs_target_img = cc_target_img[bbs_ymin:bbs_ymax, bbs_xmin:bbs_xmax]
            return dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, inference_time
        else:
            return dk_bool, cat_bool, None, None, inference_time

    def do_eyes_stage(self, eye_target_img, cc_pred_bb, cc_target_img):
        snout_crop, bbs, inference_time = self.eyes_stage.do_eyes(cc_target_img, eye_target_img, cc_pred_bb)
        return snout_crop, bbs, inference_time

    def do_haar_stage(self, target_img, pred_cc_bb_full, cc_target_img):
        haar_bbs, haar_inference_time, haar_found_bool = self.haar_stage.haar_do(target_img=target_img, cc_bbs=pred_cc_bb_full, full_img=cc_target_img)
        pc_xmin = int(haar_bbs[0][0])
        pc_ymin = int(haar_bbs[0][1])
        pc_xmax = int(haar_bbs[1][0])
        pc_ymax = int(haar_bbs[1][1])
        snout_crop = cc_target_img[pc_ymin:pc_ymax, pc_xmin:pc_xmax].copy()

        return snout_crop, haar_bbs, haar_inference_time, haar_found_bool

    def do_ff_stage(self, snout_crop):
        face_bool, ff_conf, ff_inference_time = self.ff_stage.ff_do(target_img=snout_crop)
        dk_bool = False if face_bool is True else True
        return dk_bool, face_bool, ff_conf, ff_inference_time

    def do_pc_stage(self, pc_target_img):
        pred_class, pred_val, inference_time = self.pc_stage.pc_do(target_img=pc_target_img)
        return pred_class, pred_val, inference_time

    def input_text(self, img, text, text_pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return img

class NodeBot():
    def __init__(self):
        self.last_msg_id = 0
        self.bot_updater = Updater(token=config.BOT_TOKEN)
        self.bot_dispatcher = self.bot_updater.dispatcher

        """
        Removed reboot for now, clicked on it too many times by mistake :P
        You can enable it in config.py, it if you are running this on a dedicated machine
        """
        # test if dedicated machine
        if config.IS_DEDICATED:
            self.commands = ['/help', '/nodestatus', '/sendlivepic', '/sendlastcascpic', '/letin', '/REBOOT']
        else:
            self.commands = ['/help', '/nodestatus', '/sendlivepic', '/sendlastcascpic', '/letin']

        self.node_live_img = None
        self.node_queue_info = None
        self.node_status = None
        self.node_last_casc_img = None
        self.node_over_head_info = None
        self.node_let_in_flag = None

        #Init the listener
        self.init_bot_listener()

    def init_bot_listener(self):
        telegram.Bot(token=config.BOT_TOKEN).send_message(chat_id=config.CHAT_ID, text='Hi there, NodeBot is online!')
        # Add all commands to handler
        help_handler = CommandHandler('help', self.bot_help_cmd)
        self.bot_dispatcher.add_handler(help_handler)
        node_status_handler = CommandHandler('nodestatus', self.bot_send_status)
        self.bot_dispatcher.add_handler(node_status_handler)
        send_pic_handler = CommandHandler('sendlivepic', self.bot_send_live_pic)
        self.bot_dispatcher.add_handler(send_pic_handler)
        send_last_casc_pic = CommandHandler('sendlastcascpic', self.bot_send_last_casc_pic)
        self.bot_dispatcher.add_handler(send_last_casc_pic)
        letin = CommandHandler('letin', self.node_let_in)
        self.bot_dispatcher.add_handler(letin)
        reboot = CommandHandler('reboot', self.node_reboot)
        self.bot_dispatcher.add_handler(reboot)

        # Start the polling stuff
        self.bot_updater.start_polling()

    def bot_help_cmd(self, bot, update):
        bot_message = 'Following commands supported:'
        for command in self.commands:
            bot_message += '\n ' + command
        self.send_text(bot_message)

    def node_let_in(self, bot, update):
        self.node_let_in_flag = True

    def node_reboot(self, bot, update):
        # test if dedicated machine
        if config.IS_DEDICATED:
            for i in range(15):
                time.sleep(1)
                bot_message = 'Rebooting in ' + str(15-i) + ' seconds...'
                self.send_text(bot_message)
            logging.info("Telegram bot requested a reboot. Won't do that, just logging it.")
            self.send_text('REBOOTING.. See ya later Alligator!')

            ### also uncomment this if you REALLY need to reboot
            #os.system("sudo reboot") # won't do, some may call this as a service or standalone, not on a dedicated Pi..

    def bot_send_last_casc_pic(self, bot, update):
        if self.node_last_casc_img is not None:
            caption = 'Last Cascade picture:'
            self.send_img(self.node_last_casc_img, caption)
            logging.info("Sending last cascade image to bot")
        else:
            self.send_text('No casc img available yet...')

    def bot_send_live_pic(self, bot, update):
        # self.capture_new_image() # only needed if you always want the freshest picture captured from the camera directly
        # time.sleep(0.2)          # else, it will just send the last captured pic from the queue
        if self.node_live_img is not None:
            # Encode image to JPEG format
            caption = 'Last Live picture:'
            self.send_img(self.node_live_img, caption)
            logging.info("ℹ️  Sending live image to bot")
        else:
            self.send_text('No img available yet...')

    def send_img(self, img, caption, is_encoded=False):
        if not is_encoded:
            ret, jpeg = cv2.imencode('.jpg', img)
            if not ret:
                self.send_text('⚠️  Image encoding failed.')
                return
            img = jpeg.tobytes()

        # Wrap bytes in a file-like object
        image_file = BytesIO(img)
        image_file.name = 'live.jpg'  # Important for Telegram to recognize format

        telegram.Bot(token=config.BOT_TOKEN).send_photo(
            chat_id=config.CHAT_ID,
            photo=image_file,
            caption=caption
        )

    def bot_send_status(self, bot, update):
        if self.node_queue_info is not None and self.node_over_head_info is not None:
            bot_message = 'Queue length: ' + str(self.node_queue_info) + '\nOverhead: ' + str(self.node_over_head_info) + 's'
        else:
            bot_message = 'No info yet...'
        self.send_text(bot_message)

    def send_text(self, message):
        telegram.Bot(token=config.BOT_TOKEN).send_message(chat_id=config.CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)

class DummyDQueue():
    def __init__(self):
        self.target_img = cv2.imread(os.path.join(cat_cam_py, 'CatPreyAnalyzer/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg'))

    def dummy_queue_filler(self, main_deque):
        while(True):
            img_name = datetime.now(config.TIMEZONE_OBJ).strftime("%c.%f")
            main_deque.append((img_name, self.target_img))
            logging.info("Took image, que-length: %d", len(main_deque))
            time.sleep(0.4)

class Spec_Event_Handler():
    def __init__(self):
        self.img_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/debug/input')
        self.out_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/debug/output')

        self.img_list = [x for x in sorted(os.listdir(self.img_dir)) if'.jpg' in x]
        self.base_cascade = Cascade()

    def log_to_csv(self, img_event_obj):
        csv_name = img_event_obj.img_name.split('_')[0] + '_' + img_event_obj.img_name.split('_')[1] + '.csv'
        file_exists = os.path.isfile(os.path.join(self.out_dir, csv_name))
        with open(os.path.join(self.out_dir, csv_name), mode='a') as csv_file:
            headers = ['Img_Name', 'CC_Cat_Bool', 'CC_Time', 'CR_Class', 'CR_Val', 'CR_Time', 'BBS_Time', 'HAAR_Time', 'FF_BBS_Bool', 'FF_BBS_Val', 'FF_BBS_Time', 'Face_Bool', 'PC_Class', 'PC_Val', 'PC_Time', 'Total_Time']
            writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            writer.writerow({'Img_Name':img_event_obj.img_name, 'CC_Cat_Bool':img_event_obj.cc_cat_bool,
                             'CC_Time':img_event_obj.cc_inference_time, 'CR_Class':img_event_obj.cr_class,
                             'CR_Val':img_event_obj.cr_val, 'CR_Time':img_event_obj.cr_inference_time,
                             'BBS_Time':img_event_obj.bbs_inference_time,
                             'HAAR_Time':img_event_obj.haar_inference_time, 'FF_BBS_Bool':img_event_obj.ff_bbs_bool,
                             'FF_BBS_Val':img_event_obj.ff_bbs_val, 'FF_BBS_Time':img_event_obj.ff_bbs_inference_time,
                             'Face_Bool':img_event_obj.face_bool,
                             'PC_Class':img_event_obj.pc_prey_class, 'PC_Val':img_event_obj.pc_prey_val,
                             'PC_Time':img_event_obj.pc_inference_time, 'Total_Time':img_event_obj.total_inference_time})

    def debug(self):
        event_object_list = []
        for event_img in sorted(self.img_list):
            event_object_list.append(Event_Element(img_name=event_img, cc_target_img=cv2.imread(os.path.join(self.img_dir, event_img))))

        for event_obj in event_object_list:
            start_time = time.time()
            single_cascade = self.base_cascade.do_single_cascade(event_img_object=event_obj)
            single_cascade.total_inference_time = sum(filter(None, [
                single_cascade.cc_inference_time,
                single_cascade.cr_inference_time,
                single_cascade.bbs_inference_time,
                single_cascade.haar_inference_time,
                single_cascade.ff_bbs_inference_time,
                single_cascade.ff_haar_inference_time,
                single_cascade.pc_inference_time]))
            logging.debug('Total Inference Time: %s', single_cascade.total_inference_time)
            logging.debug('Total Runtime: %.2f seconds', time.time() - start_time)

            # (Write img to output dir and) log csv of each event
            #cv2.imwrite(os.path.join(self.out_dir, single_cascade.img_name), single_cascade.output_img)
            self.log_to_csv(img_event_obj=single_cascade)

import signal

def handle_exit(signum, frame):
    try:
        if bot_instance is not None:
            bot_instance.send_text(f"⚠️ CatPreyAnalyzer received signal {signum} (e.g. CTRL+C); shutting down cleanly…")
        else:
            print("No bot instance available to notify Telegram.")
    except Exception as e:
        print(f"Failed to notify bot on exit: {e}")
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Instantiate your Sequential_Cascade_Feeder and assign its bot to the global variable
    sq_cascade = Sequential_Cascade_Feeder()
    bot_instance = sq_cascade.bot  # NodeBot instance is created in Sequential_Cascade_Feeder

    try:
        sq_cascade.queue_handler()
    except Exception as e:
        # Optional: send error to bot or log
        if bot_instance is not None:
            bot_instance.send_text(f"❌ Unhandled exception: {e}")
        print(f"Unhandled exception: {e}")
        sys.exit(1)
