# config.py

"""
Cat Prey Analyzer - Configuration Logic Summary

- Central configuration for all runtime parameters and user-adjustable settings.
  - Camera parameters: type, resolution, frame rate, flip options, etc.
  - Motion detection: thresholds, queue lengths, analysis intervals.
  - Integration: Telegram bot tokens, Sure Petcare credentials, Home Assistant endpoints.
  - Paths: Data/log storage, timezone, debug flags, and feature toggles.

- Usage:
  - All core modules import settings from this file for consistent behavior.
  - Changing system behavior (e.g., switching camera, updating credentials) is done here, not in code logic.

- Special Logic:
  - Defensive defaults for missing configuration.
  - Conversion/validation for environment variables and sensitive secrets.
  - Optionally loads secrets from environment or .env files for security.

- Edit this file to tune detection sensitivity, enable/disable integrations, or change bot/camera behavior.
  The used values are a safe starting point, tune to your own liking (depending on your hardware).
"""

import os, pytz, locale

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
    except Exception:
        pass
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

### START EDITABLE VARS ###

# Or set timezone manually
#TIMEZONE = "Europe/Berlin"

# set locale
locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')

# Set to True if this is a dedicated machine for this purpose
IS_DEDICATED = False

# Camera resolution and image flipping
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# Optional per-source overrides
CAMERA_FLIP_OVERRIDES = {
    "http://192.168.178.22:9000/mjpg": {"hflip": True, "vflip": False},
    "rtsp://stream:P4Vdo@192.168.178.59:8554/unicast": {"hflip": False, "vflip": False},
    "rtsp://stream:P4Vdo@192.168.178.58:8554/unicast": {"hflip": False, "vflip": False},
    "usb:0": {"hflip": False, "vflip": True},  # Simulated USB cam identifier
    "default": {"hflip": True, "vflip": True}
}

# Maximum queue length
MAX_QUEUE_LEN = 20
if MAX_QUEUE_LEN <= 0:
    raise ValueError("MAX_QUEUE_LEN must be positive")

# sleep interval between queued frames
SLEEP_INTERVAL = 0.25
if SLEEP_INTERVAL < 0:
    raise ValueError("SLEEP_INTERVAL must be non-negative")

# Queue filling cycles
FILL_QUEUE_CYCLES = 120
if FILL_QUEUE_CYCLES <= 0:
    raise ValueError("FILL_QUEUE_CYCLES must be positive")

# Process queue if longer than this number of frames
DEFAULT_FPS_OFFSET = 2
if DEFAULT_FPS_OFFSET < 0:
    raise ValueError("DEFAULT_FPS_OFFSET must be non-negative")

# Set motion threshold between frames in which queuing should happen
MOTION_THRESHOLD = 5000
"""
    How to tune it
    Lower values (~1000–3000) → More sensitive (even small changes cause enqueues).
    Higher values (~7000–15000) → Less sensitive (only large movements are captured).
"""

# Logging setup
LOG_FILENAME = 'log/CatPreyAnalyzer.log'
MAX_LOG_SIZE = 1 * 1024 * 1024  # 1 MB
BACKUP_COUNT = 3

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
