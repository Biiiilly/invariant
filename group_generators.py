import numpy as np

def rotation_generator(n):

    R = np.zeros((n, n))
    order = {i: None for i in range(n**2)}

    if n % 2 != 0:
        pass
    else:
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
