import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
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

        mask = torch.clamp(mask, 0, 20)  # Ensure mask values are in the range [0, 20]
        return image, mask


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.int64))

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(mask_to_tensor)
])

train_dataset = PascalVOCDataset(root_dir='./dataset/VOC2012_train_val', image_set='train',
                                 transform=transform, mask_transform=mask_transform)
val_dataset = PascalVOCDataset(root_dir='./dataset/VOC2012_train_val', image_set='val',
                               transform=transform, mask_transform=mask_transform)

MAX_CPU = 4
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=MAX_CPU)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=MAX_CPU)


class ImprovedFCN(nn.Module):
    def __init__(self, num_classes=21):
        super(ImprovedFCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedFCN(num_classes=21).to(device)

weights = torch.ones(21).to(device)
weights[0] = 0.1  # Reduce the importance of background class

criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def evaluate(model, val_loader, device):
    model.eval()
    correct_pixels, total_pixels = 0, 0

    plt.ion()  # Enable interactive mode for live updating
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_pixels += (predicted == targets).sum().item()
            total_pixels += targets.numel()

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

                plt.pause(0.05)  # Pause to update plot

    plt.ioff()  # Disable interactive mode
    plt.close(fig)

    accuracy = correct_pixels / total_pixels
    print(f"Validation Accuracy: {accuracy:.4f}")



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

        # Evaluate the model after each 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate(model, val_loader, device)


train(model, train_loader, val_loader, device)
torch.save(model.state_dict(), 'improved_model.pth')
