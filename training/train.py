# training/train.py
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    # if is_train:
    #     transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), # precomputed mean and std dev values for CIFAR-10
    #                              std=(0.2023, 0.1994, 0.2010))
    #     ])
    # else:
        # For testing, we only normalize the images
        # without any augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2023, 0.1994, 0.2010))
    ])

    data_set = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=is_train, num_workers=2)
    
    return data_loader

# Training Loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device = "cpu"):
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []   
     
    for epoch in tqdm(range(num_epochs)):
        model.train() # Sets the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]") # ?
        
        # Standard PyTorch training step for each batch: 
        #   forward pass → compute loss → backprop → update weights.
        for images, labels in train_loader: # Each batch is fetched from the loader
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #  clears old gradients from the previous step, so they don't accumulate during backpropagation.
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track number of correctly predicted samples for accuracy.
            _, predicted = torch.max(outputs.data, 1) # outputs contains raw scores (logits) for each class.
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
            # Progress Display:
            # loop.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        train_acc = 100 * correct / total
        # Validation
        val_acc, val_loss = evaluate(model, test_loader, criterion, device)
        print(f"\nTraining Accuracy after Epoch {epoch+1}: {train_acc:.2f}%")
        print(f"\nValidation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
        # if scheduler:
        #     scheduler.step(val_acc)  # only once per epoch
            
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

    # (Save) Plot training curves
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path='training_metrics.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'r-', label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

# Evaluation function
def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs,labels)
            total_loss += loss.item()
    avg_loss = total_loss/len(dataloader)
    return 100 * correct / total, avg_loss

def main():
    """
    Main function to load data, initialize model, and start training.
    """
    set_seed()
    print('1. Data Loading...')
    train_loader = load_data(is_train=True, batch_size=64)  # Load training data
    test_loader = load_data(is_train=False, batch_size=256)  # Load testing data

    print('2.Model Loading...')
    ## Model Loading
    # Load pretrained RESNET-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze all, then unfreeze what I want
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify final layer to match 10 output classes of CIFAR-10 dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,10)
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(num_ftrs,10)
    # )

    # Unfreeze layer4 and final layer for training
    params_train = list(model.layer4.parameters()) + list(model.fc.parameters())
    for param in params_train:
        param.requires_grad = True
        
    # Move model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_train, lr=0.0003) # only update classifier layer (final fc layer)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    print('3. Training...')
    ## Call your training loop    
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, device=device)

if __name__ == "__main__":
    main()
