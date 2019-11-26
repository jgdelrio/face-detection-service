import sys
from os import pardir
import torch
from src.config import DEVICE, NMS_CANDIDATE_SIZE
from src import rfb_model
from src import rfb_config
from src import tools
from src.data_preprocessing import PredictionTransform

sys.path.append(pardir)


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None, iou_threshold=0.45,
                 filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = tools.Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            for i in range(1):
                self.timer.start()
                scores, boxes = self.net.forward(images)
                print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = tools.nms(box_probs, self.nms_method,
                                  score_threshold=prob_threshold,
                                  iou_threshold=self.iou_threshold,
                                  sigma=self.sigma,
                                  top_k=top_k,
                                  candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


def rfb_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    return Predictor(
        net, rfb_config.image_size, rfb_config.image_mean_test, rfb_config.image_std,
        nms_method=nms_method,
        iou_threshold=rfb_config.iou_threshold,
        candidate_size=candidate_size,
        sigma=sigma,
        device=device)


async def load_model(model_ref, model_params=None):
    if model_ref.name in ["slim-640", "RFB-640"]:
        net = rfb_model.create_net(len(model_ref.classes), is_test=True, device=DEVICE,
                                   reduced=model_ref.name == "slim-640")
        model = rfb_predictor(net, candidate_size=NMS_CANDIDATE_SIZE, device=DEVICE)
        net.load(model_ref.ref)
    else:
        raise ValueError(f"Requested model {model_ref.name} is not yet supported")
    return model
