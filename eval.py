import re
import torch

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


if __name__ == "__main__":

    xs_test = [torch.tensor(i, dtype=torch.float32) for i in range(1, 65)]

    out_list = compute_from_file(xs_test, "output.txt")
    print("Number of outputs:", len(out_list))
    
    for i, val in enumerate(out_list[:10], start=1):
        print(f"{i} => {val.item() if val.numel()==1 else val}")
