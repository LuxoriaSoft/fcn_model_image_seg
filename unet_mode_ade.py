import os
import time
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import random

# --- Info ------------------------------------------------------------------ #
print("[INFO] Importing libraries...")
print("[INFO] PyTorch:", torch.__version__)
print("[INFO] CUDA available:", torch.cuda.is_available())
print("[INFO] MPS  available:", torch.backends.mps.is_available())
print("[INFO] ONNX :", onnx.__version__)
print("[INFO] NumPy:", np.__version__)
print("[INFO] PIL  :", Image.__version__)

# --- TensorBoard writer ---------------------------------------------------- #
writer = SummaryWriter(log_dir="./runs/unet_ade20k")

# --- Device Setup ---------------------------------------------------------- #
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

# --- Config ---------------------------------------------------------------- #
num_classes        = 150
ignore_index       = 0
image_size         = (256, 256)
batch_size         = 1
accumulation_steps = 8
epochs             = 30
samples_per_epoch  = 2000
max_val_samples    = 500
num_workers        = 0 if device_type == "mac_mps" else max(1, os.cpu_count() - 2)

# --- Dataset --------------------------------------------------------------- #
class ADE20KDataset(Dataset):
    def __init__(self, root_dir, split="training", transform=None, mask_transform=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.images, self.masks = [], []
        pattern = os.path.join(root_dir, "images", "ADE", split, "**", "*.jpg")
        for img_path in glob.glob(pattern, recursive=True):
            seg_path = img_path.replace(".jpg", "_seg.png")
            if os.path.exists(seg_path):
                self.images.append(img_path)
                self.masks.append(seg_path)
        print(f"[INFO] {split}: Found {len(self.images)} pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return image, mask

# --- Transforms ------------------------------------------------------------ #
def mask_to_tensor(mask):
    return torch.from_numpy(np.asarray(mask, dtype=np.int64))

image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.NEAREST),
    transforms.Lambda(mask_to_tensor),
])

# --- Custom Sampler for Cycling Subsets ----------------------------------- #
class CyclingChunkSampler(Sampler):
    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_samples = len(data_source)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.num_samples, generator=g).tolist()
        chunk_start = (self.epoch * self.chunk_size) % self.num_samples
        chunk = indices[chunk_start:chunk_start + self.chunk_size]
        if len(chunk) < self.chunk_size:
            chunk += indices[:self.chunk_size - len(chunk)]  # wrap around
        return iter(chunk)

    def __len__(self):
        return self.chunk_size

# --- Load datasets --------------------------------------------------------- #
train_dataset = ADE20KDataset("./dataset/ADE20K_2021", "training", image_transform, mask_transform)
val_dataset   = ADE20KDataset("./dataset/ADE20K_2021", "validation", image_transform, mask_transform)
val_dataset   = torch.utils.data.Subset(val_dataset, list(range(min(len(val_dataset), max_val_samples))))

sampler = CyclingChunkSampler(train_dataset, samples_per_epoch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- U-Net Model ----------------------------------------------------------- #
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
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
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return F.interpolate(self.final(d1), size=image_size, mode="bilinear", align_corners=False)

# --- Loss & Metrics -------------------------------------------------------- #
def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

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

def evaluate(model, loader, criterion):
    model.eval()
    total_iou, total_loss = 0, 0
    total_correct, total_pixels = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_iou += mean_iou(outputs, masks)
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
    val_loss = total_loss / len(loader)
    val_iou = total_iou / len(loader)
    val_acc = total_correct / total_pixels
    print(f"[Validation] Loss: {val_loss:.4f} | mIoU: {val_iou:.4f} | PixelAcc: {val_acc:.4f}")
    return val_loss, val_iou, val_acc

# --- Training Loop --------------------------------------------------------- #
def train(model, train_loader, val_loader, epochs, checkpoint_path="checkpoint.pth"):
    class_weights = torch.ones(num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = global_step = 0
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        # Update sampler for the epoch
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=samples_per_epoch)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = (criterion(outputs, masks) + dice_loss(outputs, masks)) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == samples_per_epoch:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("Train/Loss", loss.item() * accumulation_steps, global_step)

        avg_loss = running_loss / samples_per_epoch
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_loss, val_iou, val_acc = evaluate(model, val_loader, criterion)
            writer.add_scalar("Val/Loss", val_loss, global_step)
            writer.add_scalar("Val/mIoU", val_iou, global_step)
            writer.add_scalar("Val/PixelAccuracy", val_acc, global_step)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
        }, checkpoint_path)
        torch.save(model.state_dict(), f"unet_ade_epoch_{epoch+1}.pth")

# --- ONNX Export ----------------------------------------------------------- #
def export_onnx(model, export_path="unet_ade20k.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    torch.onnx.export(model, dummy_input, export_path,
                      export_params=True, opset_version=11,
                      do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"[ONNX] Exported to {export_path}")

# --- Main ------------------------------------------------------------------ #
if __name__ == "__main__":
    print(f"[INFO] Loaded {len(train_dataset)} training samples")
    print(f"[INFO] Using {samples_per_epoch} samples per epoch")
    model = UNet(num_classes=num_classes).to(device)
    train(model, train_loader, val_loader, epochs=epochs)
    torch.save(model.state_dict(), "unet_ade20k_final.pth")
    export_onnx(model, "unet_ade20k_final.onnx")
    writer.close()
    print("[INFO] Training complete.")
