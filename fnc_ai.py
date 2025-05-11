import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_type = "mac_mps"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "cuda"
else:
    device = torch.device("cpu")
    device_type = "cpu"

print(f"[INFO] Using device: {device} ({device_type})")

# --- Configuration ---
num_classes = 21
image_size = (256, 256)
batch_size = 2 if device_type == "mac_mps" else 4
epochs = 10
num_workers = 0 if device_type == "mac_mps" else os.cpu_count() - 2

# --- Dataset ---
class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform
        base_dir = os.path.join(root_dir, 'VOC2012_train_val')
        with open(os.path.join(base_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')) as f:
            image_ids = [line.strip() for line in f.readlines()]
        self.images = [os.path.join(base_dir, 'JPEGImages', f'{img_id}.jpg') for img_id in image_ids]
        self.masks = [os.path.join(base_dir, 'SegmentationClass', f'{img_id}.png') for img_id in image_ids]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask[mask == 255] = 0
        mask = torch.clamp(mask, 0, num_classes - 1)
        return image, mask

def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.int64))

# --- Transforms ---
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.NEAREST),
    transforms.Lambda(mask_to_tensor)
])

# --- Data Loaders ---
train_dataset = PascalVOCDataset('./dataset/VOC2012_train_val', 'train', image_transform, mask_transform)
val_dataset = PascalVOCDataset('./dataset/VOC2012_train_val', 'val', image_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return F.interpolate(self.final(d1), size=image_size, mode='bilinear', align_corners=False)

# --- Loss Functions ---
def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

# --- Evaluation with Live Visualization ---
def decode_segmap(mask, num_classes):
    label_colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ])
    r, g, b = np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)
    for l in range(num_classes):
        idx = mask == l
        r[idx], g[idx], b[idx] = label_colors[l]
    return np.stack([r, g, b], axis=2)

def evaluate(model, loader, criterion):
    model.eval()
    total_iou, total_loss = 0, 0
    total_correct, total_pixels = 0, 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    plt.ion()

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, desc='Evaluating')):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            print("Predicted class distribution:", torch.bincount(preds[0].flatten(), minlength=num_classes))

            total_iou += mean_iou(outputs, masks)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

            for cls in range(num_classes):
                cls_mask = (masks == cls)
                class_correct[cls] += ((preds == cls) & cls_mask).sum()
                class_total[cls] += cls_mask.sum()

            img = TF.to_pil_image(images[0].cpu())
            gt_mask = decode_segmap(masks[0].cpu().numpy(), num_classes)
            pred_mask = decode_segmap(preds[0].cpu().numpy(), num_classes)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img); axs[0].set_title("Input")
            axs[1].imshow(gt_mask); axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_mask); axs[2].set_title("Prediction")
            for ax in axs: ax.axis('off')
            plt.tight_layout(); plt.pause(0.001); plt.close(fig)

    print(f"[Validation] Loss: {total_loss / len(loader):.4f}")
    print(f"[Validation] mIoU: {(total_iou / len(loader)):.4f}")
    print(f"[Validation] Pixel Acc: {total_correct / total_pixels:.4f}")
    print(f"[Validation] Mean Class Acc: {(class_correct / class_total.clamp(min=1)).mean():.4f}")

def mean_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        ious.append((intersection / union) if union > 0 else torch.tensor(1.0, device=pred.device))
    return torch.mean(torch.stack(ious))

# --- Training ---
def train(model, train_loader, val_loader, epochs):
    class_weights = torch.tensor([0.1] + [1.0] * (num_classes - 1), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        if epoch == epochs - 1:
            print("[INFO] Final model evaluation on validation set")
            evaluate(model, val_loader, criterion)

        torch.save(model.state_dict(), f'unet_epoch_{epoch+1}.pth')

# --- ONNX Export ---
def export_onnx(model, export_path='unet_final.onnx'):
    model.eval()
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    torch.onnx.export(
        model, dummy_input, export_path,
        export_params=True, opset_version=11,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[ONNX] Model exported to {export_path}")

# --- Main ---
if __name__ == "__main__":
    print(f"[INFO] Loaded {len(train_dataset)} training samples.")
    print(f"[INFO] Loaded {len(val_dataset)} validation samples.")
    model = UNet(num_classes=num_classes).to(device)
    train(model, train_loader, val_loader, epochs=epochs)
    torch.save(model.state_dict(), 'unet_final.pth')
    export_onnx(model, 'unet_final.onnx')