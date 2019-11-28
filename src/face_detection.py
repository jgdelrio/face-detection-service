import cv2
import asyncio
import nest_asyncio
from packaging import version
from src.config import logger, THRESHOLD, DEVICE, NMS_CANDIDATE_SIZE, DFT_MODEL
from src import rfb_model
from src.build_models import *
from models.definitions import MODELS, PRELOAD, LATEST_MODEL
from src.tools import show_np_img

LOGGER = logger(name="face-detection")
unavailable_model = "The model selected is not available"
nest_asyncio.apply()
PRE_LOADED_MODELS = {}


def process_image(img, model_info, model_params, mod_image, print_label):
    return asyncio.get_event_loop().run_until_complete(
        _process_image(img, model_info, model_params, mod_image, print_label))


def get_model_reference(name, version):
    ref = name + "_" + version
    if ref in MODELS.keys():
        return MODELS[ref]
    else:
        for key, val in MODELS.items():
            if name in val.aliases and version in val.version:
                return MODELS[key]
        raise ValueError(f"Model {name} is not within the pre-defined models")


def get_lastest_version(name):
    if name in LATEST_MODEL.keys():
        return LATEST_MODEL[name]
    raise ValueError(f"Model {name} is not within the pre-defined models")


def get_full_model_path(model_info):
    return model_info.ref + "_" + model_info.version + model_info.ext


def dict_faces_prediction(boxes, labels, probs, inference_time=None, image=None):
    dict_results = {'boxes': boxes.tolist(), 'labels': labels.tolist(), 'probs': probs.tolist()}
    if inference_time:
        dict_results['inference_time'] = inference_time
    if image:
        dict_results['image'] = image.tolist()
    return dict_results


async def call_api(ref):
    return ref


async def _process_image(img, model_info=None, model_params=None, mod_image=True, print_label=False):
    if model_info is None:
        model_info = get_model_reference(DFT_MODEL)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if model_info.location == 'local':
        predictor = await load_model(model_info, model_params=model_params)
        boxes, labels, probs = predictor.predict(image, NMS_CANDIDATE_SIZE / 2, THRESHOLD)

        if mod_image:
            # Print boxes on the image
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                if print_label:
                    label = f"{probs[i]:.2f}"
                    # label = f"{CLASS_NAMES[labels[i]]}: {probs[i]:.2f}"""
                    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # cv2.putText(img, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # show_np_img(img)
            # cv2.imwrite(os.path.join(result_path, file_path), img)
            return dict_faces_prediction(boxes, labels, probs,
                                         inference_time=predictor.inference_time, image=img)
        else:
            return dict_faces_prediction(boxes, labels, probs,
                                         inference_time=predictor.inference_time)
    else:
        return call_api(model_info.ref)


async def load_model(model_ref, model_params=None):
    global PRE_LOADED_MODELS
    if model_ref.name in PRE_LOADED_MODELS.keys():
        return PRE_LOADED_MODELS[model_ref.name]
    if model_ref.name in ["slim-640", "RFB-640"]:
        net = rfb_model.create_net(len(model_ref.classes), is_test=True, device=DEVICE,
                                   reduced=model_ref.name == "slim-640")
        model = rfb_predictor(net, candidate_size=NMS_CANDIDATE_SIZE, device=DEVICE)
        net.load(get_full_model_path(model_ref))
    else:
        raise ValueError(f"Requested model {model_ref.name} is not yet supported")
    return model


async def preload_models(models):
    result = await asyncio.gather(*[load_model(get_model_reference(*m)) for m in models])
    return result

# Preload selected models to reduce loading time
PRE_LOADED_MODELS = {(name[0] + "_" + name[1]): model for name, model in zip(PRELOAD, asyncio.run(preload_models(PRELOAD)))}
