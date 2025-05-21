import torch
from sklearn.datasets import load_digits
from layer import InvariantNet
import torchvision.transforms as transforms
from eval import invariants_eval_combined_list


digits = load_digits()
X = torch.tensor(digits.images, dtype=torch.float32)
y = torch.tensor(digits.target, dtype=torch.long)

# Apply transformations
random_rotate = transforms.RandomRotation(degrees=(0, 360))
X_rotated = torch.stack([random_rotate(img.unsqueeze(0)).squeeze(0) for img in X])

# Convert back to original data type if needed
X_rotated = X_rotated.to(torch.double)

combined_list = []
with open("stored_data/new_combined_list_100.txt", "r") as f:
   for line in f:
    combined_list.append(line.strip())

# invariants = invariants_eval_combined_list(X, combined_list)
X = invariants_eval_combined_list(X_rotated, combined_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

invariant_model = InvariantNet().to(device)
invariant_model = invariant_model.double()
invariant_model.load_state_dict(torch.load('invariants.pth', map_location=device))
invariant_model.eval()

X = X.to(device)
y = y.to(device)

with torch.no_grad():
    outputs = invariant_model(X)
    _, predicted = torch.max(outputs, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    accuracy = correct / total * 100
    print(f"Test Accuracy on Random Rotated Dataset: {accuracy:.2f}%")
