# %%
import torch
import torchvision.models as models
import torch.nn as nn

if torch.cuda.is_available():
    print("✅ GPU is available")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU is not available")
# %%
from train import load_data

train_loader = load_data(is_train=True, batch_size=64)  # Load training data
test_loader = load_data(is_train=False, batch_size=256)  # Load testing data

# %%  Load Pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # arg. to load pretrained weights

# Freeze all the parameters (except final layer)
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for 10 output classes (CIFAR-10)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,10)

# %% 👀 Inspect model 
# print names and types of all top-level layers in resnet50 model
for name, layer in model.named_children():
    print(name, layer)

# %%
