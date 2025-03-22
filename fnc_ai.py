import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.mask_transform = mask_transform

        self.images = []
        self.masks = []

        with open(os.path.join(root_dir, 'VOC2012_train_val', 'ImageSets', 'Segmentation', f'{image_set}.txt'), 'r') as f:
            image_ids = f.readlines()

        for image_id in image_ids:
            image_id = image_id.strip()
            image_path = os.path.join(root_dir, 'VOC2012_train_val', 'JPEGImages', f'{image_id}.jpg')
            mask_path = os.path.join(root_dir, 'VOC2012_train_val', 'SegmentationClass', f'{image_id}.png')

            self.images.append(image_path)
            self.masks.append(mask_path)

        print(f"Loaded {len(self.images)} images and masks from the dataset.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])

        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)

        if self.mask_transform:
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        mask = torch.clamp(mask, 0, 20)
        return image, mask

def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.int64))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(mask_to_tensor)
])

train_dataset = PascalVOCDataset(root_dir='./dataset/VOC2012_train_val', image_set='train',
                                 transform=transform, mask_transform=mask_transform)
val_dataset = PascalVOCDataset(root_dir='./dataset/VOC2012_train_val', image_set='val',
                               transform=transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Dice Loss function
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)  # Convert predictions to class indices (most likely class)
    
    intersection = (pred == target).sum()  # Calculate intersection for Dice coefficient
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Combined Loss (CrossEntropy + Dice Loss)
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_loss = dice_loss

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return ce + dice

# Improved FCN Model
class ImprovedFCN(nn.Module):
    def __init__(self, num_classes=21):
        super(ImprovedFCN, self).__init__()

        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Decoder (upsampling) using transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

        # Final layer to reduce the output size to 256x256
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # Decoder with skip connections (using upsampling to match the sizes)
        x = self.deconv1(x3)
        x2_upsampled = F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=False)  # Upsample x2 to match size of x
        x = x + x2_upsampled  # Skip connection
        x = self.deconv2(x)
        x1_upsampled = F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=False)  # Upsample x1 to match size of x
        x = x + x1_upsampled  # Skip connection
        x = self.deconv3(x)

        # Ensure final output size is 256x256
        x = self.final_upsample(x)

        return x

# IoU Calculation
def calculate_iou(pred, target, num_classes=21):
    iou = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        iou.append(intersection / (union + 1e-6))  # Add small epsilon to avoid division by zero

    return np.mean(iou)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedFCN(num_classes=21).to(device)

weights = torch.ones(21).to(device)
weights[0] = 0.1  # Reduce the importance of background class

criterion = CombinedLoss(weight=weights, ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=10)

# Evaluation Function
def evaluate(model, val_loader, device):
    model.eval()
    correct_pixels, total_pixels = 0, 0
    iou_list = []

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_pixels += (predicted == targets).sum().item()
            total_pixels += targets.numel()

            # Calculate IoU
            iou = calculate_iou(predicted, targets)
            iou_list.append(iou)

            # Real-time visualization
            inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            targets_np = targets.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            for i in range(inputs_np.shape[0]):
                axes[0].imshow(inputs_np[i])
                axes[0].set_title("Input Image")
                axes[1].imshow(targets_np[i], cmap="tab20")
                axes[1].set_title("Ground Truth")
                axes[2].imshow(predicted_np[i], cmap="tab20")
                axes[2].set_title("Prediction")

                plt.pause(0.0005)

    # Disable interactive mode
    plt.ioff()
    plt.close(fig)

    accuracy = correct_pixels / total_pixels
    mean_iou = np.mean(iou_list)
    print(f"Validation Accuracy: {accuracy:.4f}, Mean IoU: {mean_iou:.4f}")

# Training Loop
def train(model, train_loader, val_loader, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
        scheduler.step()

        # Evaluate the model at the end of the final epoch
        if epoch == epochs - 1:
            evaluate(model, val_loader, device)

train(model, train_loader, val_loader, device)
torch.save(model.state_dict(), 'improved_model2.pth')
