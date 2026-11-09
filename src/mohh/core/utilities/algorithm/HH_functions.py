

import numpy as np


def sin(x:np.ndarray) -> np.ndarray:
    return np.sin(x)


def cos(x:np.ndarray) -> np.ndarray:
    return np.cos(x)


def exp(x:np.ndarray) -> np.ndarray:
    return np.exp(x)


def log(x:np.ndarray) -> np.ndarray:
    return np.log(x)


def convolution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Arity: 2
    Convolve parent1 with kernel = parent2
    """
    return np.convolve(x, y, "same")


def one_point(parent1:np.ndarray, parent2:np.ndarray):
    """
    @param parent1 1D numpy array
    @param parent2 1D numpy array 
    """

    crossover_point = np.random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1


def masked_cross(parent1:np.ndarray, parent2:np.ndarray):
    """
    @param parent1 1D numpy array 
    @param parent2 1D numpy array 
    """
    n_var = len(parent1)

    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)

    mask = np.random.randint(0, 2, size=n_var).astype("bool")
    child1[mask] = parent1[mask]
    child1[~mask] = parent2[~mask]

    child2[~mask] = parent1[~mask]
    child2[mask] = parent2[mask]

    return child1


def swap(x):
    """
    Swap two variables from a solution
    """

    x = verify_inputs(x)

    n_var = len(x)

    i, j = np.random.choice(n_var, size=2, replace=False)

    x[i], x[j] = x[j], x[i]

    return x


def move_right(x):
    """
    Move all elements to the right.
    """
    x = verify_inputs(x)

    return np.concatenate(([x[-1]], x[:-1]))


def reverse(x):
    """
    Reverse the elements.
    """
    x = verify_inputs(x)

    return x[::-1]


def map_list(x, y):

    x = verify_inputs(x)

    y = verify_inputs(y)

    if not np.all((y >= 0) & (y < len(x))):
        raise ValueError("All elements of y must be valid indices for x and y.")

    if not np.all((x >= 0) & (x < len(y))):
        raise ValueError("All elements of x must be valid indices for x and y.")

    new_x = x[y]
    new_y = y[x]

    return new_x, new_y


def alternate_elements(x, y):
    """
    Alternate each element of both parents
    """
    x = verify_inputs(x)

    y = verify_inputs(y)

    new_x = []
    new_y = []

    for i in range(len(x)):

        if np.random.rand() < 0.5:
            new_x.extend([x[i], y[i]])
            new_y.extend([y[i], x[i]])

        else:
            new_x.extend([y[i], x[i]])
            new_y.extend([x[i], y[i]])

    new_x, new_y = np.array(new_x), np.array(new_y)

    _, indx = np.unique(new_x, return_index=True)
    new_x = new_x[np.sort(indx)]

    _, indx = np.unique(new_y, return_index=True)
    new_y = new_y[np.sort(indx)]

    return new_x, new_y


# Alternate_segments unused
def alternate_segments(x, y):
    """
    Alternate list segments.

    x and y are list of lists.
    The number of sublists must match
    """

    if len(x) != len(y):
        print("Alternate segments: No of Sublists doesn't match.")
        raise(ValueError)

    new_x = []
    new_y = []

    for i in range(len(x)):

        if np.random.rand() < 0.5:
            new_x.extend(x[i])
            new_x.extend(y[i])

            new_y.extend(y[i])
            new_y.extend(x[i])

        else:
            new_x.extend(y[i])
            new_x.extend(x[i])

            new_y.extend(x[i])
            new_y.extend(y[i])

    new_x, new_y = np.array(new_x), np.array(new_y)

    _, indx = np.unique(new_x, return_index=True)
    new_x = new_x[np.sort(new_x)]

    _, indx = np.unique(new_y, return_index=True)
    new_y = new_y[np.sort(indx)]

    return new_x, new_y


def preserve_elements(x):
    """
    Select some elements.
    """

    x = verify_inputs(x)

    k = np.random.randint(1, len(x))

    x = np.array(x)

    selected = np.random.choice(len(x), size=k, replace=False)

    result = -np.ones_like(x)
    result[selected] = x[selected]

    return result


def preserve_segments(x):
    """
    Select segments
    """

    x = verify_inputs(x)

    i = 0
    j = len(x) - 1
    while (i == 0 and j == len(x) - 1) or (i == j):
        selected = np.sort(np.random.choice(len(x), size=2, replace=False))
        i, j = selected
        j -= 1

    result = -np.ones_like(x)
    result[i:j] = x[i:j]

    return result


def fill_first_occurring(preserved, filler):
    """
    Fill preserved with filler
    """

    preserved = verify_inputs(preserved)

    filler = verify_inputs(filler)

    existing_values = set(preserved[preserved != -1])

    j = 0
    for i in range(len(preserved)):
        # nested expressions may exhaust the filler: leave -1 and let test_individual reject it
        while preserved[i] == -1 and j < len(filler):

            if filler[j] not in existing_values:
                preserved[i] = filler[j]
            j += 1

    return preserved


def order_based(fill_method, preserve_method, x, y):
    """
    fill_method/preserve_method: chosen by the grammar.
    x, y: terminals.
    """

    x = verify_inputs(x)

    y = verify_inputs(y)

    preserved_data_x = preserve_method(x)
    preserved_data_y = preserve_method(y) 

    new_x: np.ndarray = fill_method(preserved_data_x, y)
    new_y: np.ndarray = fill_method(preserved_data_y, x)

    return new_x, new_y


def collapse(x, y=None):

    return x


def verify_inputs(x):

    if isinstance(x, tuple):
        x = collapse(x[0], x[1])
    return x

