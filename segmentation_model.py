import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

class ImageSegmentationModel:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

    def segment_image(self, image_pil):
        image_tensor = self.transform(image_pil).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(image_tensor)[0]

        masks = prediction['masks']
        labels = prediction['labels']
        scores = prediction['scores']

        # Filter masks by score
        final_masks = []
        final_labels = []
        final_scores = []
        for i in range(len(scores)):
            if scores[i] > 0.85:
                mask = masks[i, 0].mul(255).byte().cpu().numpy()
                final_masks.append(mask)
                final_labels.append(labels[i].item())
                final_scores.append(scores[i].item())

        return final_masks, final_labels, final_scores
