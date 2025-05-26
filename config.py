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
CAM_HFLIP = True           # horizontal flip
CAM_VFLIP = False          # vertical flip – adjust to taste
# Maximum queue length
MAX_QUEUE_LEN = 20

# sleep interval between queued frames
SLEEP_INTERVAL = 0.25

# Queue filling cycles
FILL_QUEUE_CYCLES = 60

# Process queue if longer than this number of frames
DEFAULT_FPS_OFFSET = 2

