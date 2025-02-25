import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt  # <-- Added for plotting

from layer import InvariantNet  # Replace with the path/name of your layer module if needed

# Load data
digits = load_digits()
X = torch.load("invariants1.pt")  # Shape (1797, 9510)
y = digits.target                # Shape (1797,)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional normalization (commented out for now)
# X_train = (X_train - X_train.mean()) / X_train.std()
# X_test  = (X_test  - X_train.mean()) / X_train.std()

# Create datasets and loaders
train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = InvariantNet()
model = model.double()
network = model.to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Training params
epochs = 64
loss_list = []
acc_list = []

# Training loop
for epoch in range(epochs):
    network.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device, dtype=torch.float64)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float64)
            labels = labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    acc_list.append(accuracy)
    print(f"Test Accuracy: {accuracy:.2f}%")

# Save model
torch.save(network.state_dict(), 'sklearn_digits.pth')

# -------------------------------
# PLOTTING TRAINING LOSS & ACCURACY
# -------------------------------
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), loss_list, 'b-', label='Training Loss')
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), acc_list, 'r-', label='Test Accuracy')
plt.title('Test Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
