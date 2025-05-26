# config.py
import os

# Insert Chat ID and Bot Token according to Telegram API
CHAT_ID = os.getenv("CAT_PREY_CHAT_ID",   "CHANGE_ME")
BOT_TOKEN = os.getenv("CAT_PREY_BOT_TOKEN", "CHANGE_ME")

# Insert webhooks for home assistant
HA_UNLOCK_WEBHOOK = os.getenv("HA_UNLOCK_WEBHOOK", "CHANGE_ME")
HA_LOCK_OUT_WEBHOOK = os.getenv("HA_LOCK_OUT_WEBHOOK", "CHANGE_ME")
HA_LOCK_ALL_WEBHOOK = os.getenv("HA_LOCK_ALL_WEBHOOK", "CHANGE_ME")

# TOKEN for HA REST
HA_REST_URL = os.getenv("HA_REST_URL", "CHANGE_ME")
HA_REST_TOKEN = os.getenv("HA_REST_TOKEN", "CHANGE_ME")

# Camera resolution and image flipping
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_HFLIP = True
CAM_HFLIP = True

# Maximum queue length
MAX_QUEUE_LEN = 20

# sleep interval between queued frames
SLEEP_INTERVAL = 0.25

# Queue filling cycles
FILL_QUEUE_CYCLES = 60

# Process queue if longer than this number of frames
DEFAULT_FPS_OFFSET = 2
