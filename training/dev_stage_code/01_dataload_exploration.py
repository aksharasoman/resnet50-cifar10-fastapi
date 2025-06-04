# %% 
import torch, torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# %% 
# Define transformations for training and testing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), # These are the precomputed mean and std dev values for CIFAR-10 (Google “CIFAR-10 normalization values”)
                            std=(0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), # These are the precomputed mean and std dev values for CIFAR-10 (Google “CIFAR-10 normalization values”)
                            std=(0.2023, 0.1994, 0.2010))
])
'''
transforms.Compose([...])
→ It creates a pipeline of image transformations. 

ToTensor()
→ Converts images from PIL format ([0–255]) to PyTorch tensors ([0–1] floats).

Normalize(mean, std)
→ Shifts and scales each RGB channel to have mean ~0 and std ~1.
→ Helps the network converge faster and more stably.

“For transforms, use Compose. 
Always start with ToTensor, and follow with Normalize(mean, std) — dataset-specific.”
'''

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data",train=False, download=True, transform=transform_test)
'''
“From torchvision.datasets, I ask for CIFAR10, 
specify location, train/test split, download if needed, and give it a transform pipeline.”
📦 datasets.CIFAR10(...) → 📁 + ✂️ + 📥 + 🧹 = your cleaned dataset.
'''

# Create data loaders
'''
Think of the DataLoader as a courier service that delivers batches of data to your training loop.
It takes the dataset, breaks it into batches, and shuffles if needed.
shuffle=True → randomize the order (important for training to avoid learning order bias).
'''
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# %% ## 🔍 Dataset Inspection
# Check the length of the dataset
print(f"Length of training dataset: {len(train_dataset)}")
print(f"Length of test dataset: {len(test_dataset)}")
# Check the shape of the first image and its label
image, label = train_dataset[0]
print(f"Shape of first image: {image.shape}")  # torch.Size([3, 32, 32])
print(f"Label of first image: {label}")  # 6 (for example)
# Check the number of classes
num_classes = len(train_dataset.classes)  # 10
print(f"Number of classes: {num_classes}")  # 10
# Class names
classes = train_dataset.classes  # ['airplane', 'automobile', ..., 'truck']
print(f"Class names: {classes}")
# %% ## 🔍 Dataset Inspection
# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize if normalized; skip this line if not normalized
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Get a batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
print("Training Images")
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(8)))

# Repeat for test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

print("Test Images")
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(8)))

# %% 
