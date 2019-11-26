import cv2
import asyncio
import nest_asyncio
from src.config import logger, THRESHOLD
from src.load_models import *
from models.definitions import *
from src.tools import show_np_img

LOGGER = logger(name="face-detection")
unavailable_model = "The model selected is not available"
nest_asyncio.apply()


def process_image(img, model_info, model_params, mod_image, print_label):
    return asyncio.get_event_loop().run_until_complete(
        _process_image(img, model_info, model_params, mod_image, print_label))


def get_model_reference(name):
    if name in MODELS.keys():
        return MODELS[name]
    elif name in EXTRA_DEFINITIONS.keys():
        return MODELS[EXTRA_DEFINITIONS[name]]
    else:
        for key, val in MODELS.items():
            if name in val.aliases:
                return MODELS[key]
        raise ValueError(f"Model {name} is not within the pre-defined models")


def dict_faces_prediction(boxes, labels, probs, image=None):
    if image is None:
        return {'boxes': boxes.tolist(), 'labels': labels.tolist(), 'probs': probs.tolist()}
    else:
        return {'boxes': boxes.tolist(), 'labels': labels.tolist(), 'probs': probs.tolist(),
                'image': image.tolist()}


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
            return dict_faces_prediction(boxes, labels, probs, img)
        else:
            return dict_faces_prediction(boxes, labels, probs)
    else:
        return call_api(model_info.ref)


async def load_models(models):
    result = await asyncio.gather(*[load_model(get_model_reference(m)) for m in models])
    # loop = asyncio.get_event_loop()
    # tasks = [loop.create_task(load_model(m)) for m in models]
    # result = (main())
    return result  #loop.run_until_complete(*tasks)


# PRELOADED_MODELS = {name: model for name, model in zip(PRELOAD, asyncio.run(load_models(PRELOAD)))}