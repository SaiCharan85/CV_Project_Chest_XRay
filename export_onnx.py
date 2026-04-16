import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE  = 224
ONNX_PATH = "pneumonia_detector.onnx"

model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# use older export path explicitly
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
print(f"Exported ONNX model to {ONNX_PATH}")