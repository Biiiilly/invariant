import torch
from sklearn.datasets import load_digits
from layer import InvariantNet


digits = load_digits()
X = torch.load("stored_data/invariants_rot90.pt")
y = torch.tensor(digits.target, dtype=torch.long)

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
    print(f"Test Accuracy on 90 Degree Rotated Dataset: {accuracy:.2f}%")
