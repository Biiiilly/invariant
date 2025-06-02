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

    R = np.zeros((n**2, n**2))
    order = {i: i for i in range(n**2)}
    ring = np.array([[i * n + j for j in range(n)] for i in range(n)])
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

    group_trans = (np.array(group)).T
    a, b = group_trans.shape[0], group_trans.shape[1]
    for i in range(a):
        subgroup = group_trans[i]
        if i + 1 == a:
            k = 0
        else:
            k = i + 1
        transformed_subgroup = group_trans[k]
        for j in range(b):
            key = subgroup[j]
            element = transformed_subgroup[j]
            order[key] = element

    for i in range(n**2):
        j = order[i]
        R[j, i] = 1

    return R


def left_generator(n):

    L = np.zeros((n**2, n**2))
    order = {i: None for i in range(n**2)}

    group = []
    for i in range(n):
        subgroup = [i]
        for k in range(n-1):
            subgroup.append(i + (k+1)*n)
        group.append(subgroup)

    for i in range(n):
        subgroup = group[i]
        if i + 1 == n:
            k = 0
        else:
            k = i + 1
        transformed_subgroup = group[k]
        for j in range(n):
            key = subgroup[j]
            element = transformed_subgroup[j]
            order[key] = element

    for i in range(n**2):
        j = order[i]
        L[j, i] = 1
    
    return L


def down_generator(n):

    D = np.zeros((n**2, n**2))
    order = {i: None for i in range(n**2)}

    group = []
    for i in range(0, n**2, n):
        subgroup = [i]
        for k in range(n-1):
            subgroup.append(i + k + 1)
        group.append(subgroup)

    for i in range(n):
        subgroup = group[i]
        if i + 1 == n:
            k = 0
        else:
            k = i + 1
        transformed_subgroup = group[k]
        for j in range(n):
            key = subgroup[j]
            element = transformed_subgroup[j]
            order[key] = element

    for i in range(n**2):
        j = order[i]
        D[j, i] = 1

    return D


def numpy_to_singular_matrix(matrix, name):
    rows, cols = matrix.shape
    elements = ",".join(",".join(map(lambda x: str(int(x)), matrix[row])) for row in range(rows))
    return f"matrix {name}[{rows}][{cols}] = {elements};"


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
    R = rotation_generator(n)
    D = down_generator(n)
    L = left_generator(n)

    R_singular = numpy_to_singular_matrix(R, 'R')
    D_singular = numpy_to_singular_matrix(D, 'D')
    L_singular = numpy_to_singular_matrix(L, 'L')

    singular_commands = f"""
    LIB "finvar.lib";
    ring F=0,({variables}),dp;
    {R_singular}
    matrix REY,M=reynolds_molien(R);
    poly p(1..2);
    p(1..2)=partial_molien(M,5);
    p(1);
    """

    print(singular_commands)

    result = run_singular_command(singular_commands)

    lines = result.splitlines()
    expression = lines[-2]

    return expression
