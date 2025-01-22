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

    L = np.zeros((n, n))
    order = {i: None for i in range(n**2)}

    return L


def up_generator(n):

    U = np.zeros((n, n))
    order = {i: None for i in range(n**2)}

    return U