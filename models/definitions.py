"""Store definitions for local and remote models"""
import sys
from os import getenv, pardir, path
from collections import namedtuple

sys.path.append(pardir)
from src.config import PATH_MODELS

CLASS_NAMES = ["BACKGROUND", "face"]
Model = namedtuple("Model", "name location ref version speed classes description aliases")

MODELS = {
    "RFB-640": Model(
        name="RFB-640",
        location="local",
        ref=path.join(PATH_MODELS,  "RFB-640.pth"),
        version="0.0.1",
        speed="fast",
        classes=CLASS_NAMES,
        description="optimized model for fast processing",
        aliases=["RFB", "rfb"]),
    "slim-640": Model(
        name="slim-640",
        location="local",
        ref=path.join(PATH_MODELS, "slim-640.pth"),
        version="0.0.1",
        speed="ultrafast",
        classes=CLASS_NAMES,
        description="smallest model with overall optimization for speed",
        aliases=["slim"]),
}

N_MODELS = len(MODELS)

EXTRA_DEFINITIONS = {
    "fastest": "RFB-640",
    "accurate": "RFB-640",
}

PRELOAD = ["slim-640", "RFB-640"]

DFT_MODEL = getenv("DFT_MODEL", "RFB-640")
MODEL_LIST = [*MODELS.keys(), *EXTRA_DEFINITIONS.keys()]
