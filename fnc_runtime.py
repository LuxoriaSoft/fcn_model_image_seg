import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model class again
class SimpleFCN(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleFCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.fc(x)
        return x

# Initialize the model and load the checkpoint
model = SimpleFCN(num_classes=21).to(device)
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()  # Set the model to evaluation mode

# Define transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize image
    transforms.ToTensor(),           # Convert image to tensor
])

# Function to process and predict
def predict(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply the same transformation as during training
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)  # Get the raw model output

    # Get the predicted class (argmax over channels)
    _, predicted = torch.max(output, 1)  # Get the class with the highest probability

    print(predicted)

    # Convert predicted tensor to numpy for visualization
    predicted = predicted.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

    return predicted, image

# Function to visualize the result
def visualize(image, predicted_mask):
    # Display the image and the predicted mask
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    # Show the predicted mask with the color map (tab20b for 21 classes)
    ax[1].imshow(predicted_mask, cmap='tab20b')  # Use a colormap for visualizing the classes
    ax[1].set_title('Predicted Segmentation')
    ax[1].axis('off')

    plt.show()

# Test the program with an example image
image_path = 'image2.jpg'  # Specify the path to your input image
predicted_mask, image = predict(image_path)
visualize(image, predicted_mask)
