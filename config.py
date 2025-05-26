# config.py
import os


def _require_env(var: str) -> str:
    val = os.getenv(var, "CHANGE_ME")
    if val == "CHANGE_ME":
        raise RuntimeError(f"Environment variable {var} must be set.")
    return val

# Insert Chat ID and Bot Token according to Telegram API
CHAT_ID  = _require_env("TELEGRAM_CHAT_ID")
BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")

# Insert webhooks for home assistant
HA_UNLOCK_WEBHOOK = _require_env("HA_UNLOCK_WEBHOOK")
HA_LOCK_OUT_WEBHOOK = _require_env("HA_LOCK_OUT_WEBHOOK")
HA_LOCK_ALL_WEBHOOK = _require_env("HA_LOCK_ALL_WEBHOOK")

# TOKEN for HA REST
HA_REST_URL = _require_env("HA_REST_URL")
HA_REST_TOKEN = _require_env("HA_REST_TOKEN")

# Camera resolution and image flipping
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# Optional per-source overrides
CAMERA_FLIP_OVERRIDES = {
    "http://192.168.1.22:9000/mjpg": {"hflip": True, "vflip": False},
    "rtsp://stream:P4Vdo@192.168.1.59:8554/unicast": {"hflip": False, "vflip": False},
    "usb:0": {"hflip": False, "vflip": True},  # Simulated USB cam identifier
    "default": {"hflip": True, "vflip": True}
}

# Maximum queue length
MAX_QUEUE_LEN = 20

# sleep interval between queued frames
SLEEP_INTERVAL = 0.25

# Queue filling cycles
FILL_QUEUE_CYCLES = 60

# Process queue if longer than this number of frames
DEFAULT_FPS_OFFSET = 2
