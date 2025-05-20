import sys
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image_size = (256, 256)
num_classes = 21
onnx_model_path = "unet_final.onnx"

# Using program arguments for flexibility
if len(sys.argv) != 2:
    print("Usage: python fnc_runtime.py <image_path>")
    sys.exit(1)
image_path = sys.argv[1]

# Load ONNX model
print("[INFO] Loading ONNX model...")
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Load and Preprocess Input Image
print("[INFO] Loading image...")
image = Image.open(image_path).convert("RGB").resize(image_size)
image_np = np.array(image).astype(np.float32) / 255.0
image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

# Run the model
print("[INFO] Running inference...")
outputs = session.run(["output"], {"input": image_np})[0]
preds = np.argmax(outputs, axis=1)[0]  # Shape: (H, W)

# Decode segmentation mask
label_colors = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)

def decode_segmap(mask):
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)
    for l in range(num_classes):
        idx = mask == l
        r[idx], g[idx], b[idx] = label_colors[l]
    return np.stack([r, g, b], axis=2)

decoded_mask = decode_segmap(preds)

# Show the input image and predicted mask
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(decoded_mask)
plt.title("Predicted Mask")
plt.axis('off')

plt.tight_layout()
plt.show()
