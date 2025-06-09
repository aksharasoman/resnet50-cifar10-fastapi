'''
Use GRAD-CAM to visualize where the model is looking in the image when making a prediction
'''
#%%
import torch
from torchvision.models import resnet50 
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#%%
# 1. Load Trained ResNet-50 model
# reinitialize the model architecture and load the state dict before applying Grad-CAM.
set_seed()
# Initialize model
model = resnet50()  # By default, no pretrained weights are used. We'll be loading our own weights

# Modify the final layer for CIFAR-10 (10 classes)
num_classes = 10
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),           # Match your best setup
    torch.nn.Linear(model.fc.in_features, num_classes)
)

# Load saved weights
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))

# %%
