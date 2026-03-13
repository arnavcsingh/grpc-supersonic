import torch
from torch._export.converter import TS2EPConverter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io

MODEL_ID = "farleyknight/mnist-digit-classification-2022-09-04"


class MnistWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


class Mnist:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print("Loading model on", self.device)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        print("model loaded")
    def inference(self, image: Image.Image):
        image = image.convert("L")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
            return pred, confidence

    def inference_bytes(self, img_bytes: bytes):
        image = Image.open(io.BytesIO(img_bytes))
        return self.inference(image)

    def _get_dummy_input(self, device):
        img = Image.new("L", (28, 28))
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].to(device)

    def export_torchscript(self, path="mnist.pt"):
        wrapper = MnistWrapper(self.model).to(self.device)
        wrapper.eval()
        dummy = self._get_dummy_input(self.device)
        scripted = torch.jit.trace(wrapper, dummy)
        scripted = torch.jit.freeze(scripted)
        scripted.save(path)
        print("TorchScript model saved to", path)

    def export_export(self, path="mnist.pte"):
        from torch.export import export
        wrapper = MnistWrapper(self.model).cpu()
        dummy = self._get_dummy_input("cpu")
        ep = export(wrapper, (dummy,))
        torch.export.save(ep, path)
        print("Exported model saved to", path)

if __name__ == "__main__":
    m = Mnist()
    export_torchscript("mnist.pt")
    ts_model = torch.jit.load("mnist.pt")
    ts_model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    converter = TS2EPConverter(ts_model, (dummy,), {})
    ep = converter.convert()
    torch.export.save(ep, "mnist.pt2")
    print("Saved mnist.pt2")