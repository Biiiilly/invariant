import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np

from layer import InvariantNet

digits = load_digits()
X = digits.images  # (1797, 8, 8)
y = digits.target  # (1797,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class DigitsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        image = self.images[idx].astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)  # [1, 8, 8]
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 3. 定义 transform（可选）
transform = transforms.Compose([
    # MNIST 原本用了 (0.5, 0.5) 的归一化，这里也可以继续用。
    # 但实际上 digits 数据的均值/方差不同，可根据需要改成适合 digits 的值
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = DigitsDataset(X_train, y_train, transform=transform)
test_dataset = DigitsDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = InvariantNet(64)
model = model.double()
network = model.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
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


torch.save(network.state_dict(), 'cnn_digits.pth')
