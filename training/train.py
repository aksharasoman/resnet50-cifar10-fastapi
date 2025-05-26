# training/train.py
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
# from utils import train_model  # assuming you have this in utils.py

def load_data(is_train = True, batch_size: int = 64) -> DataLoader:
    """
    Loads and preprocesses the CIFAR-10 training dataset.

    Args:
        is_train (bool): If True, loads the training dataset. 
                         If False, loads the test dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    # Define the transformation for training and testing
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), # precomputed mean and std dev values for CIFAR-10
                                 std=(0.2023, 0.1994, 0.2010))
        ])
    else:
        # For testing, we only normalize the images
        # without any augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010))
        ])

    data_set = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return data_loader

def main():
    """
    Main function to load data, initialize model, and start training.
    """
    train_loader = load_data(is_train=True, batch_size=64)  # Load training data
    test_loader = load_data(is_train=False, batch_size=256)  # Load testing data

    ## Model Loading
    # Load pretrained RESNET-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze all, then unfreeze what I want
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify final layer to match 10 output classes of CIFAR-10 dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,10)

    # Unfreeze only final layer for training
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Move model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    ## Call your training loop
    # train_model(model, train_loader, device=device)

if __name__ == "__main__":
    main()
