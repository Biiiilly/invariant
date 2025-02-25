import re
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import json


def compute_nonzero_invariants(xs, filepath):

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
                    if val != 0:
                        results.append(expr)
                except Exception as e:
                    print(f"Error parsing expression: {expr_copy}")
                    raise e

    return results


# Load the digits dataset
# digits = load_digits()

# Create a dictionary mapping each digit to its corresponding images
# digit_dict = {digit: digits.images[digits.target == digit] for digit in range(10)}

# output_dict[i] is a set
# and results is another set
# I want to choose the same elements to be output_dict[i]

def compute_invariants_for_each_digit(digit_dict, filepath):

    output_dict = {i: [] for i in range(10)}

    for i in range(10):
        images = digit_dict[i] #(n, 8, 8)
        number_of_images = images.shape[0]
        for n in range(number_of_images):
            image = images[n]
            image = image.flatten()
            results = compute_nonzero_invariants(image, filepath)
            if n == 0:
                output_dict[i] = results
            else:
                output_dict[i] = [x for x in output_dict[i] if x in results]

    return output_dict


# Load the digits dataset
digits = load_digits()
digit_dict = {digit: digits.images[digits.target == digit] for digit in range(10)}
combined_dict = compute_invariants_for_each_digit(digit_dict, "output.txt")
combined_list = [elem for sublist in combined_dict.values() for elem in sublist]


with open("combined_list.txt", "w") as f:
    for item in combined_list:
        f.write(str(item) + "\n")


with open("combined_dict.json", "w") as f:
    json.dump(combined_dict, f)


#x = digits.images[0]
#x = x.flatten()
#results = compute_from_combined_list(x, combined_list)