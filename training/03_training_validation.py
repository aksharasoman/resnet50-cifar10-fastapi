import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss() # for multi-class classification
optimizer = optim.Adam(model.fc.parameteres(), lr = 0.001) # only update classifier layer (final fc layer)

# Training Loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device = "cpu"):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train() # Sets the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]") # ?
        
        # Standard PyTorch training step for each batch: 
        #   forward pass → compute loss → backprop → update weights.
        for images, labels in loop: # Each batch is fetched from the loader
            optimizer.zero_grad() #  clears old gradients from the previous step, so they don't accumulate during backpropagation.
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track number of correctly predicted samples for accuracy.
            _, predicted = torch.max(outputs.data, 1) # outputs contains raw scores (logits) for each class.
            correct += (predicted == labels).sum().item()
            
            # Progress Display:
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)
            
        # Validation
        val_acc = evaluate(model, test_loader, device)
        print(f"\nValidation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")
        
                # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

# Evaluation function
def evaluate(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)
            


            
        
        
