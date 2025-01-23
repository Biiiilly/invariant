import numpy as np


def get_outer_ring(matrix):
    n = len(matrix)
    if n == 0:
        return []
    if n == 1:
        return matrix[0]
    
    result = []
    result.extend(matrix[0])
    
    for i in range(1, n - 1):
        result.append(matrix[i][0])
        result.append(matrix[i][-1])

    result.extend(matrix[-1])
    
    return result


def rotation_generator(n):

    R = np.zeros((n, n))
    order = {i: None for i in range(n**2)}
    ring = np.array([[i * n + j + 1 for j in range(n)] for i in range(n)])
    group = []

    if n % 2 != 0:
        pass
    else:
        for i in range(n-1):
            outer_ring = get_outer_ring(ring[i:n-i, i:n-i])
            


        pass

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
        L[i, j] = 1
    
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
        D[i, j] = 1

    return D
