import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing

# --- Configuration ---
usable_cores = max(multiprocessing.cpu_count() - 2, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 21
image_size = (256, 256)
batch_size = 4
epochs = 10

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
        mask = torch.clamp(mask, 0, num_classes - 1)
        return image, mask

def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.int64))

# --- Transforms ---
image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.NEAREST),
    transforms.Lambda(mask_to_tensor)
])

# --- Data Loaders ---
train_dataset = PascalVOCDataset('./dataset/VOC2012_train_val', 'train', image_transform, mask_transform)
val_dataset = PascalVOCDataset('./dataset/VOC2012_train_val', 'val', image_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=usable_cores)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=usable_cores)

# --- Model ---
class ImprovedFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)

# --- Losses ---
def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

# --- Metrics ---
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

def evaluate(model, loader):
    model.eval()
    total_iou = 0
    total_correct = 0
    total_pixels = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Evaluating'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            total_iou += mean_iou(outputs, masks)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

            for cls in range(num_classes):
                cls_mask = (masks == cls)
                class_correct[cls] += ((preds == cls) & cls_mask).sum()
                class_total[cls] += cls_mask.sum()

    pixel_accuracy = total_correct / total_pixels
    mean_class_accuracy = (class_correct / class_total.clamp(min=1)).mean()

    print(f"[Validation] mIoU: {total_iou / len(loader):.4f}")
    print(f"[Validation] Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"[Validation] Mean Class Accuracy: {mean_class_accuracy:.4f}")

# --- Training ---
def train(model, train_loader, val_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0]*num_classes, device=device))

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
        scheduler.step(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        if epoch == 0 or (epoch + 1) % 5 == 0:
            evaluate(model, val_loader)

        torch.save(model.state_dict(), f'improved_fcn_epoch_{epoch+1}.pth')

# --- ONNX Export ---
def export_onnx(model, export_path='improved_fcn.onnx'):
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
    print(f"[INFO] Loaded {len(train_dataset)} images for 'train' set.")
    print(f"[INFO] Loaded {len(val_dataset)} images for 'val' set.")

    model = ImprovedFCN(num_classes=num_classes).to(device)
    train(model, train_loader, val_loader, epochs=epochs)
    torch.save(model.state_dict(), 'improved_fcn_final.pth')
    export_onnx(model, 'improved_fcn.onnx')
