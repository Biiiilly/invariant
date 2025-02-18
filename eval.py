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


def invariants_eval(X, n=64):

    num = X.shape[0]
    output_list = []
    for i in range(num):
        x = X[i].flatten() # (8, 8)
        output = compute_from_file(x, "output.txt")
        output = [x for x in output if x != 0]
        random_output = random.sample(output, n)
        output_list.append(random_output)

    return torch.tensor(output_list)


digits = load_digits()
X = digits.images  # (1797, 8, 8)
invariants = invariants_eval(X, n=64)

torch.save(invariants, "invariants.pt")

