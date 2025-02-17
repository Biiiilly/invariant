import numpy as np
import os
import subprocess
import re


def get_outer_ring(matrix):

    n = len(matrix)
    if n == 0:
        return []
    if n == 1:
        return matrix[0]
    
    result = []
    result.extend(matrix[0])
    
    for i in range(1, n - 1):
        result.append(matrix[i][-1])

    result.extend(matrix[-1][::-1])

    for i in range(n - 2, 0, -1):
        result.append(matrix[i][0])
    
    return result


def outer_ring_group(outer_ring, n):

    group = []
    for i in range(n-1):
        subgroup = []
        subgroup.append(outer_ring[i])
        subgroup.append(outer_ring[i+n-1])
        subgroup.append(outer_ring[i+2*(n-1)])
        subgroup.append(outer_ring[i+3*(n-1)])
        group.append(subgroup)

    return group


def rotation_generator(n):

    ring = np.array([[i * n + j + 1 for j in range(n)] for i in range(n)])

    print(ring)

    group = []

    while True:

        k = ring.shape[0]
        if k == 1:
            break
        outer_ring = get_outer_ring(ring)
        group += outer_ring_group(outer_ring, k)
        if k == 2:
            break
        else:
            ring = ring[1:k-1, 1:k-1]

    return group


def rotation_output_to_singular(n):
    
    data = rotation_generator(n)
    list_strs = []
    for sub_list in data:
        elements = ",".join(map(str, sub_list))       
        list_strs.append(f"list({elements})")         
    middle = f"list({','.join(list_strs)})"

    return f"def GEN=list({middle});"


def run_singular_command(singular_commands):

    result = subprocess.run(
    ["C:/cygwin64/bin/bash.exe", "-c", "Singular"],
    input=singular_commands,
    capture_output=True,
    text=True
    )
    return result.stdout


def generate_invariants(n):

    num_variables = n**2
    variables = ', '.join([f'x{i}' for i in range(1, num_variables+1)])
    GEN = rotation_output_to_singular(n)

    singular_commands = f"""
    LIB "finvar.lib";
    ring F = 0, ({variables}), dp;
    {GEN}
    matrix G = invariant_algebra_perm(GEN,1);
    G;
    """

    print(singular_commands)

    result = run_singular_command(singular_commands)

    return result

