import sys
import logging
from os import getenv, pardir, path

sys.path.append(pardir)

# Paths
PATH_ROOT = path.abspath(path.join(path.dirname(__file__), path.pardir))
PATH_MODELS = path.join(PATH_ROOT, "models")
MODELS = {
    "RFB-640": {
        "name": "RFB-640",
        "file": "RFB-640.pth",
        "ref": path.join(PATH_MODELS,  "RFB-640.pth")
    }}
DFT_MODEL = MODELS["RFB-640"]["name"]           # Default model

# cpu or cuda:0
DEVICE = getenv("DEVICE", "cpu")
# RFB (more accurate) or slim (faster)
MODEL_TYPE = getenv("MODEL_TYPE", "RFB")
# optional value 128/160/320/480/640/1280
NETWORK_INPUT_SIZE = int(getenv("NETWORK_INPUT_SIZE", "640"))
NMS_CANDIDATE_SIZE = int(getenv("NMS_CANDIDATE_SIZE", "1500"))
THRESHOLD = float(getenv("THRESHOLD", "0.6"))

PORT = int(getenv("PORT", "7000"))
ALLOWED_CONTENT = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif', 'multipart/form-data': 'any'}

LOG_LEVEL = getenv("LOG_LEVEL", "DEBUG")
levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }


def logger(name="face-detection", log_level=LOG_LEVEL):
    log = logging.getLogger(name)
    log.setLevel(levels[log_level])

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(levels[LOG_LEVEL])
    formatter = logging.Formatter("%(asctime)s[%(name)s:%(levelname)s]: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log
