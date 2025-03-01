import torch
from sklearn.datasets import load_digits
import torchvision.transforms as transforms


def compute_from_combined_list(xs, combined_list):

    results = []

    for expr in combined_list:

        expr_copy = str(expr)

        try:
            val = eval(expr, {"xs": xs, "torch": torch}, {})
            results.append(val)
        except Exception as e:
            print(f"Error parsing expression: {expr_copy}")
            raise e

    return results


def invariants_eval_combined_list(X, combined_list):

    num = X.shape[0]
    output_list = []
    for i in range(num):
        x = X[i].flatten() # (8, 8)
        output = compute_from_combined_list(x, combined_list)
        output_list.append(output)

    return torch.tensor(output_list)


digits = load_digits()
X = digits.images  # (1797, 8, 8)

X_rot90 = torch.rot90(torch.tensor(X), k=1, dims=(1, 2))

combined_list = []
with open("stored_data/new_combined_list_100.txt", "r") as f:
   for line in f:
    combined_list.append(line.strip())

# invariants = invariants_eval_combined_list(X, combined_list)
invariants = invariants_eval_combined_list(X, combined_list)

torch.save(invariants, "stored_data/invariants_selected_100.pt")
