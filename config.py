# config.py

"""
Cat Prey Analyzer - Centralized Configuration

Purpose:
    - Houses all runtime parameters, integrations, and user-tunable settings.
    - Ensures all modules use consistent configuration for camera, analysis, integration, and logging.

Configuration Areas:
    - Camera: Source (USB, RTSP, etc.), resolution, FPS, flip, queue size, heartbeat interval, etc.
    - Detection: Motion sensitivity, frame analysis parameters, event thresholds.
    - Integrations: Telegram bot (chat ID, token), Sure Petcare (Surepy) credentials, Home Assistant endpoints.
    - Logging: Path, rotation, retention, and log verbosity.
    - Paths: Data/log directories, timezone, debug flags, feature toggles.

Best Practices:
    - Edit this file for most behavior changes (camera, detection, integrations).
    - Use environment variables or a .env file for secrets (tokens/passwords).
    - Defensive defaults and validation included.
    - Read this file for descriptions on how to tune detection aggressiveness, enable integrations, or modify system behavior.

Notes:
    - All other modules import from here for consistency.
    - Safe starting values are provided; tune for your hardware/environment.
"""

import os
import locale
from dotenv import load_dotenv
try:
    import pytz
except ImportError:
    raise ImportError("pytz module is required. Install with: pip install pytz") from None

load_dotenv()

def detect_system_timezone():
    """Try to detect system timezone from /etc/timezone or /etc/localtime."""
    try:
        # Debian-based systems
        if os.path.exists("/etc/timezone"):
            with open("/etc/timezone") as f:
                tz = f.read().strip()
                if tz in pytz.all_timezones:
                    return tz

        # Systems with /etc/localtime symlink to zoneinfo
        if os.path.islink("/etc/localtime"):
            tz_path = os.readlink("/etc/localtime")
            parts = tz_path.split("/")
            if "zoneinfo" in parts:
                idx = parts.index("zoneinfo")
                tz = "/".join(parts[idx + 1:])
                if tz in pytz.all_timezones:
                    return tz
    except Exception as e:
        print(f"Failed to detect system timezone: {e}")
    return "UTC"

def _require_env(var: str) -> str:
    val = os.getenv(var)
    if val is None:
        raise RuntimeError(f"Environment variable {var} must be set.\n\nDid you forget to source your 'env' file?")
    return val

# Get system internal timezone
TIMEZONE = globals().get("TIMEZONE", None)  # default: None

# Validate TIMEZONE or fall back to system
if TIMEZONE not in pytz.all_timezones:
    sys_tz = detect_system_timezone()
    if TIMEZONE is not None:
        print(f"⚠️  Warning: TIMEZONE='{TIMEZONE}' in config.py is not valid. Falling back to system timezone.")
    TIMEZONE = sys_tz

# Warn if system timezone is UTC (often a sign it's unset)
if TIMEZONE == "UTC":
    print("""⚠️  TIMEZONE defaulted to 'UTC'.
    You should set a proper timezone in config.py (e.g. 'Europe/Berlin'),
    or set/check for the correct system timezone. Please use a canonical TZ identifier from:
    https://en.wikipedia.org/wiki/List_of_tz_database_time_zones""")
# Or set timezone manually in the following step


### START EDITABLE VARS ###

#TIMEZONE = "Europe/Berlin"

# Set locale
try:
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
except locale.Error:
    # Fall back to system default or C locale
    try:
        locale.setlocale(locale.LC_TIME, '')
    except locale.Error:
        locale.setlocale(locale.LC_TIME, 'C')
    print("⚠️  Warning: de_DE.UTF-8 locale not available, using system default")

# Set to True if this is a dedicated machine for this purpose
# User needs to be allowed passwordless 'sudo reboot'
IS_DEDICATED = False

# Default camera settings
CAMERA_OVERRIDES = {
    "cam1": {
        "url": "http://192.168.178.22:9000/mjpg",
        "cam_width": 1600,
        "cam_height": 900,
        "hflip": False,
        "vflip": False
    },
    "cam2": {
        "url": "rtsp://192.168.178.59:8554/unicast",
        "cam_width": 1920,
        "cam_height": 1080,
        "hflip": False,
        "vflip": False
    },
    "cam3": {
        "url": "rtsp://192.168.178.58:8554/unicast", # for testing
        "cam_width": 1920,
        "cam_height": 1080,
        "hflip": False,
        "vflip": False
    },
    "default": {
        "cam_width": 640,
        "cam_height": 480,
        "hflip": True, # my pi camera module is mounted upside-down..
        "vflip": True  # my pi camera module is mounted upside-down..
    }
}

# use GStreamer or OpenCV for capturing frames
USE_GSTREAMER = True

# The heartbeat interval
HEARTBEAT_INTERVAL = 60

# Watchdog Thread for Main Analysis Loop
WATCHDOG_TIMEOUT = 120

# Maximum queue length
MAX_QUEUE_LEN = 20
if MAX_QUEUE_LEN <= 0:
    raise ValueError("MAX_QUEUE_LEN must be positive")

# Restart camera aquisition process after this many failures
MAX_FRAME_FAILURES = 5

# Sleep interval between queued frames
SLEEP_INTERVAL = 0.25
if SLEEP_INTERVAL < 0:
    raise ValueError("SLEEP_INTERVAL must be non-negative")

# Queue filling cycles
FILL_QUEUE_CYCLES = 0
if FILL_QUEUE_CYCLES < 0:
    raise ValueError("FILL_QUEUE_CYCLES must be positive")

# Process queue if longer than this number of frames
DEFAULT_FPS_OFFSET = 2
if DEFAULT_FPS_OFFSET < 0:
    raise ValueError("DEFAULT_FPS_OFFSET must be non-negative")

# Set motion threshold between frames in which queuing should happen
# How to tune it:
# - Lower values (~1000–3000) → More sensitive (even small changes cause enqueues).
# - Higher values (~7000–15000) → Less sensitive (only large movements are captured).
MOTION_THRESHOLD = 5000

# Logging setup
LOG_FILENAME = 'log/CatPreyAnalyzer.log'
MAX_LOG_SIZE = 1 * 1024 * 1024  # 1 MB
BACKUP_COUNT = 3

# Define opening time for the catflap, in seconds
OPEN_TIME = 60

CAMERA_SSH_USERNAME = "root"
CAMERA_REMOTE_COMMAND = "/system/sdcard/controlscripts/rtsp"
CAMERA_SSH_KEY_FILE = "~/.ssh/id_ed25519"

### END EDITABLE VARS ###
#
# You don't have to set things from here on down,                       #
# config.py is importing these from environment variables               #
# you can create a hidden file and import it before starting cascade.py #
# $> source .src; python3 cascade.py rtsp://192.168.1.1//unicast --log  #

# This is the actual timezone object to use elsewhere
TIMEZONE_OBJ = pytz.timezone(TIMEZONE)

# Chat ID and Bot Token according to Telegram API
CHAT_ID  = _require_env("TELEGRAM_CHAT_ID")
BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")

# Webhook for home assistant
HA_WEBHOOK = os.environ.get("HA_WEBHOOK")

# URL and TOKEN for homeassistant REST API
HA_REST_URL = os.environ.get("HA_REST_URL")
HA_REST_TOKEN = os.environ.get("HA_REST_TOKEN")

# Token and device ID for surepy
SUREPY_DEVICE_ID = os.environ.get("SUREPY_DEVICE_ID")
SUREPY_EMAIL = os.environ.get("SUREPY_EMAIL")
SUREPY_PASSWORD = os.environ.get("SUREPY_PASSWORD")
SUREPY_TOKEN = os.environ.get("SUREPY_TOKEN")
