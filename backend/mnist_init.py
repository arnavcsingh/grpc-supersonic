import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

MODEL_ID = "farleyknight/mnist-digit-classification-2022-09-04"

class Mnist:
    def __init__(self, device=None):
        if device is None:
            if(torch.cuda.is_available()):
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print("Loading model on ", self.device)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        print("model loaded")
    def inference(self, image: Image.Image):
        image = image.convert("L")
        inputs = self.processor(images = image, return_tensors="pt")
        new_inputs = {}
        for key, val in inputs.items():
            new_inputs[key] = val.to(self.device)
        inputs = new_inputs
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
            return pred, confidence
    def inference_bytes(self, img_bytes:bytes):
        image = Image.open(io.BytesIO(img_bytes))
        return self.inference(image)
