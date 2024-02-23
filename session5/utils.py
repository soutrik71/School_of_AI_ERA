import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),  # Apply randomly a list of transformations with a given probability
    transforms.Resize((28, 28)),  # Resize the image to the given size
    transforms.RandomRotation((-15., 15.), fill=0),  # Rotate the image by an angle
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.1307,), (0.3081,)), # Normalize a tensor image with mean and standard deviation.
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(), # PIL to tensor
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize a tensor image with mean and standard deviation.
    ])
