"""Store definitions for local and remote models"""
import re
from os import getenv, pardir, path, listdir
from collections import namedtuple
from packaging import version


CLASS_NAMES = ["BACKGROUND", "face"]
PATH_MODELS = path.dirname(__file__)
model_files = [f for f in listdir(PATH_MODELS) if (path.isfile(path.join(PATH_MODELS, f))) and f[-3:] != ".py"]

Model_def = namedtuple("Model", "name location ref ext speed classes description aliases")
model_params = ["name", "location", "ref", "ext", "speed", "classes", "description", "aliases", "version"]
Model = namedtuple("Model", model_params)

MODELS = {}
LATEST_MODEL = {}
MODELS_BASIC = {
    "RFB-640": Model_def(
        name="RFB-640",
        location="local",
        ref=path.join(PATH_MODELS,  "RFB-640"),
        ext=".pth",
        speed="fast",
        classes=CLASS_NAMES,
        description="optimized model for fast processing",
        aliases=["RFB", "rfb", "fast-accurate"]),
    "slim-640": Model_def(
        name="slim-640",
        location="local",
        ref=path.join(PATH_MODELS, "slim-640"),
        ext=".pth",
        speed="ultrafast",
        classes=CLASS_NAMES,
        description="smallest model with overall optimization for speed",
        aliases=["slim", "fastest"]),
}

# Fill model list with all versions found
version_regex = re.compile("\d\.\d\.\d")
for m in MODELS_BASIC:
    is_model = [k for k in model_files if (m + "_" in k)]
    versions = [version_regex.findall(v)[0] for v in is_model]
    versions.sort(key=lambda s: map(int, s.split('.')))
    for v in versions:
        model_ref = m + "_" + v
        MODELS[model_ref] = Model(*MODELS_BASIC[m], v)
    LATEST_MODEL[m] = v

PRELOAD = [("slim-640", "0.0.1"),
           ("RFB-640", "0.0.1")]

MODEL_NAMES_LIST = MODELS_BASIC.keys()
