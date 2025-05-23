# config.py

# Insert Chat ID and Bot Token according to Telegram API
CHAT_ID = '1234567890' # my Telegram ID
BOT_TOKEN = '1234567890:SecretTelegramApiToken534HherhETQ' # bot Telegram ID:TOKEN

# Insert unlock/lock webhooks for home assistant
#HA_UNLOCK_WEBHOOK = "http://homeassistant.local:8123/api/webhook/-=UnlockCatFlapNow=-"
#HA_LOCK_OUT_WEBHOOK = "http://homeassistant.local:8123/api/webhook/-=LockOutCatFlapNow=-"
#HA_LOCK_ALL_WEBHOOK = "http://homeassistant.local:8123/api/webhook/-=LockOutCatFlapNow=-"

# TOKEN from home assistant REST
#HA_REST_URL = "http://homeassistant.local:8123/api/states/sensor.cat_flap_sureflap"
#HA_REST_TOKEN = "ChangeThisToYourCreated-SecretHomeAssistantREST-Token"

# Camera resolution and image flipping
CAM_X = 1920
CAM_Y = 1080
CAM_HFLIP = True
CAM_HFLIP = True

# Maximum queue length
MAX_QUEUE_LEN = 4

# Sensitivity threshold for minimal motion required to enqueue a frame
MOTION_THRESHOLD = 2.5  # or another reasonable default value

# Motion detection thresholds for adaptive queue rate
MOTION_THRESHOLDS = {
    "low": 2,        # Below this, frame is skipped
    "medium": 5,    # Above this, use medium enqueue rate
    "high": 10       # Above this, use fast enqueue rate
}

# Corresponding enqueue intervals (in seconds)
ENQUEUE_INTERVALS = {
    "slow": 1.5,
    "medium": 1.0,
    "fast": 0.5,
    "max": 0.1
}

# Queue a frame at the start and in this interval
FORCED_ENQUEUE_INTERVAL = 60

