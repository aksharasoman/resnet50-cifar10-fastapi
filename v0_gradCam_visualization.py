'''
Use GRAD-CAM to visualize where the model is looking in the image when making a prediction
'''
#%%
import torch
from torchvision.models import resnet50 
import random
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

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
model.load_state_dict(torch.load('dev_stage_code/best_model.pth', map_location='cpu'))

# Set to eval mode and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
# %%
# 2. Pick the Target Layer
# In ResNet-50, the typical choice is the last convolutional block:
target_layers = [model.layer4[-1]]

# 3. Prepare Input Image (CIFAR-10)


# 4. Generate Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

# Optionally, specify the class index for which you want the CAM
target_category = None  # or set to predicted class index

grayscale_cam = cam(input_tensor=input_tensor, targets=None)
grayscale_cam = grayscale_cam[0]  # For batch size = 1

# Overlay on image
visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

# 5. Save Result
Image.fromarray(visualization).save("gradcam_output.jpg")