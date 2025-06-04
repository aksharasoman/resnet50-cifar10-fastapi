# %%
import torch
import torchvision.models as models
import torch.nn as nn

from train import load_data

train_loader = load_data(is_train=True, batch_size=64)  # Load training data
test_loader = load_data(is_train=False, batch_size=256)  # Load testing data

# %%  Load Pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # arg. to load pretrained weights

# Freeze all the parameters (except final layer) “Freeze all, then unfreeze what I want.”
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for 10 output classes (CIFAR-10)
    # Why: The default ResNet-50 is trained for 1000 ImageNet classes. CIFAR-10 has only 10.
    # So: “Take existing input size, replace with 10-class output.”
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,10)

# Only the new layer's parameters will be updated during training
for param in model.fc.parameters():
    param.requires_grad = True

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

