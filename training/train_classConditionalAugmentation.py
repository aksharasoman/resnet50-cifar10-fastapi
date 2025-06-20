# training/train.py
'''
Apply class conditional augmentations (stronger transforms) for hard-to-classify classes, 
which are identified during error analysis following 1st stage training of the model.
Why?:
    - Focuses learning more on confusing classes (cat, dog) by introducing greater variability.
    - Keeps augmentation light for already well-learned classes (e.g., ship).
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class ClassConditionalAugDataset(Dataset):
    def __init__(self, base_dataset, target_classes, strong_transform, default_transform):
        self.base_dataset = base_dataset
        self.target_classes = target_classes  # e.g., [3, 5] for cat and dog
        self.strong_transform = strong_transform
        self.default_transform = default_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if label in self.target_classes:
            image = self.strong_transform(image)
        else:
            image = self.default_transform(image)
        return image, label
    
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
        # CIFAR-10 class indices: 3 = cat, 5 = dog
        cat_dog_indices = [3, 5]
        # Base dataset without any transform
        raw_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        
        # Default transform for all classes
        default_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010))            
        ])

        # Stronger transform for confusing classes
        strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomErasing(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010))
        ])
        # Wrapped dataset with class-conditional transform
        custom_train_dataset = ClassConditionalAugDataset(
            base_dataset=raw_train_dataset,
            target_classes=cat_dog_indices,
            strong_transform=strong_transform,
            default_transform=default_transform
        )
        data_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True)

    else: 
        # For testing, we only normalize the images
        # without any augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010))
        ])

        data_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
    return data_loader

# Training Loop
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=10, device = "cpu"):
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []   
     
    for epoch in tqdm(range(num_epochs)):
        model.train() # Sets the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
                
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
               # index of the maximum logit: corresponds to the most confident class prediction.
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        train_acc = 100 * correct / total
        # Validation
        val_acc, val_loss, ece_score  = evaluate(model, test_loader, criterion, device)
        print(f"\nAfter Epoch {epoch+1}:: Training Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}% | ECE: {ece_score:.4f}")

        if scheduler:
            scheduler.step(val_loss)  # only once per epoch
            
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

    ece_metric = MulticlassCalibrationError(num_classes=10, n_bins=15, norm='l1') # Initialize ECE metric (once, before validation loop)
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs,labels)
            total_loss += loss.item()

            all_probs.append(probs.cpu())
            all_targets.append(labels.cpu())
    val_acc = 100 * correct / total
    avg_loss = total_loss/len(dataloader)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Save all_probs and all_targets for further analysis (final epoch outputs will be obtained)
    torch.save({'probs': all_probs, 'targets': all_targets}, 'eval_probs_targets.pth') 
    
    # Compute ECE
    ece_score = ece_metric(all_probs, all_targets).item()

    return val_acc, avg_loss, ece_score

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
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs,10)
    )

    # Unfreeze layer3, layer4 and final layer for training
    params_train = list(model.layer3.parameters()) + list(model.layer4.parameters()) + list(model.fc.parameters())
    for param in params_train:
        param.requires_grad = True
        
    # Move model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(params_train, lr=0.0003) # only update classifier layer (final fc layer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print('3. Training...')
    ## Call your training loop    
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=30, device=device)

if __name__ == "__main__":
    main()
 