import re
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np


def compute_from_file(xs, filepath):

    pattern = re.compile(r"x(\d+)")

    def replacer(match):

        idx = int(match.group(1))
        return f"xs[{idx-1}]"  # x1 -> xs[0], x2 -> xs[1], ...

    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:

            line = line.strip().rstrip(',')
            if not line:
                continue

            expressions = line.split(',')

            for expr in expressions:
                expr_copy = str(expr)
                expr = expr.strip()
                if not expr:
                    continue

                expr = expr.replace('^', '**')
                expr = pattern.sub(replacer, expr)

                try:
                    val = eval(expr, {"xs": xs, "torch": torch}, {})
                    results.append(val)
                except Exception as e:
                    print(f"Error parsing expression: {expr_copy}")
                    raise e

    return results


def invariants_eval_random(X, n):

    num = X.shape[0]
    output_list = []
    for i in range(num):
        x = X[i].flatten() # (8, 8)
        output = compute_from_file(x, "output.txt")
        output = [x for x in output if x != 0]
        random_output = random.sample(output, n)
        output_list.append(random_output)

    return torch.tensor(output_list)


def invariants_eval_all(X):

    num = X.shape[0]
    output_list = []
    for i in range(num):
        x = X[i].flatten() # (8, 8)
        output = compute_from_file(x, "output.txt")
        output = [x for x in output if x != 0]
        output_list.append(len(output))

    return torch.tensor(output_list)


def invariants_eval_n(X, n):

    num = X.shape[0]
    output_tensor = torch.zeros((num, n))
    for i in range(num):
        x = X[i].flatten()
        output = compute_from_file(x, "output.txt")
        output = [x for x in output if x != 0]
        if len(output) >= n:
            output_selected = random.sample(output, n)
            output_tensor[i] = torch.tensor(output_selected)
        else:
            output_tensor[i, :len(output)] = torch.tensor(output)

    return output_tensor


digits = load_digits()
X = digits.images  # (1797, 8, 8)
#max_value = 9510
#invariants = invariants_eval_n(X, max_value)
#print(invariants.shape)
#print(invariants[:10, :10])


#invariants = invariants_eval_n(X, 520)
#print(invariants)
#mean_value = torch.mean(invariants.float())
#max_value = torch.max(invariants.float())
#min_value = torch.min(invariants.float())
#print(mean_value)
#print(max_value)
#print(min_value) 
#tensor(5173.7305)
#tensor(9507.)
#tensor(487.)

#torch.save(invariants, "invariants2.pt")

