import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import torch.optim as optim

from layer import InvariantNet

digits = load_digits()
data = digits.data[0]
X = torch.load("invariants.pt")  # (1797, 9510)
y = digits.target  # (1797,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#X_train = (X_train - X_train.mean()) / X_train.std()
#X_test = (X_test - X_train.mean()) / X_train.std()

train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InvariantNet()
model = model.double()
network = model.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

epochs = 64
loss_list = []
acc_list = []

for epoch in range(epochs):
    network.train()
    running_loss = 0.0
    
    for images, labels in train_loader:

        images = images.to(device, dtype=torch.float64)
        labels = labels.to(device)

        #print(images)
        #print(images.shape)
        
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


torch.save(network.state_dict(), 'sklearn_digits.pth')
