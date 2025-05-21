import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Config ---
model_path = "unet_ade20k_final.onnx"
image_path = "image2.jpg"  # ← CHANGE THIS to your test image path
image_size = (256, 256)
num_classes = 150

# --- ADE20K Colormap (Simplified) ---
def get_ade20k_colormap():
    import matplotlib
    cmap = np.zeros((150, 3), dtype=np.uint8)
    base = matplotlib.cm.get_cmap('tab20', 150)
    for i in range(150):
        r, g, b, _ = base(i)
        cmap[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap

# --- Load + preprocess image ---
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

original_img = Image.open(image_path).convert("RGB")
input_tensor = transform(original_img).unsqueeze(0).numpy()  # (1, 3, H, W)

# --- Run ONNX model ---
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
outputs = session.run(None, {"input": input_tensor})
output = outputs[0]  # (1, num_classes, H, W)

# --- Postprocess prediction ---
pred = np.argmax(output[0], axis=0)  # (H, W)
cmap = get_ade20k_colormap()

# --- Colorize segmentation ---
color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
for cls_id in range(num_classes):
    color_mask[pred == cls_id] = cmap[cls_id]

# --- Overlay: blend input image + segmentation mask ---
input_resized = original_img.resize(image_size)
overlay = Image.blend(input_resized, Image.fromarray(color_mask), alpha=0.5)

# --- Display ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(input_resized)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(color_mask)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
