import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


# Enum for strategies
class Strategy(Enum):
    MIXED = 1
    DEFECT = 2
    COOPERATE = 3
    EXPLORATORY = 4

# Set number of repetitions with N trials each
REPETITIONS = 1000


# Computes K using the formula in the paper (Material section)
def compute_Ks(x1, x2):
    xn1 = x1 / 15
    xn2 = x2 / 15
    alpha = 0.19
    LK1 = alpha * (3 * (1 - xn1) + 7 * xn2)
    LK2 = alpha * (7 * xn1 + 3 * (1 - xn2))
    return LK1, LK2


# Returns acceptable index if out of bounds
def check_bounds(ix):
    if ix >= xcard:
        ix = xcard - 1
    if ix < 0:
        ix = 0
    return ix


def show_K_Values():
    x = list(range(15))
    values = np.zeros((15,15))
    for v in x:
        for w in x:
            values[v, w] = compute_Ks(v, w)[0]
    ax = sns.heatmap(values)
    ax.set_xlabel("v")
    ax.set_ylabel("w")
    plt.show()



'''Returns correct strategy based on which one we want
Example: we want only defect for an agent'''


def get_strategy(strategy, i, ix, k=[1, 1]):

    if strategy == Strategy.MIXED:
        # Get direction and magnitude by going where K is min
        # Add eps to denominator for numerical stability (K1.max() can be 0)
        loss = (k[i - 2] - k[i - 1]) / (k.max() + eps)
        # Get how much space to the end of the position array
        remaining_path = (xcard - ix[i - 1] - 1) if loss < 0 else ix[i - 1]
        ix[i] = ix[i - 1] + lr * loss * remaining_path
        ix[i] = check_bounds(ix[i])
    elif strategy == Strategy.COOPERATE:
        ix[i] = 9
    elif strategy == Strategy.DEFECT:
        ix[i] = 0
    elif strategy == Strategy.EXPLORATORY:
        # If last two Ks are equal, we need to explore
        if np.abs((k[i - 2] - k[i - 1])) <= eps:

            loss = -0.1 if ix[i-1] > xcard/2 else 0.1
            remaining_path = (xcard - ix[i - 1] - 1) if loss > 0 else ix[i - 1]
            ix[i] = ix[i - 1] + loss * remaining_path
            ix[i] = check_bounds(ix[i])
        else:
            # If last two Ks are not equal, we proceed with MIXED strategy
            get_strategy(Strategy.MIXED, i, ix, k)



#Variables for the final confusion matrix
data_final_ix1 = []
data_final_ix2 = []
data_starting_ix1 = []
data_starting_ix2 = []
data_final_K1 = []
data_final_K2 = []

for j in range(REPETITIONS):

    # Term for numerical stability
    eps = np.exp(-10)

    # Learning rate
    lr = np.exp(-3)

    # Number of trials
    N = 100

    # Number of positions possible (Quantization of position space)
    xcard = 10

    # Generate positions based on sample size
    x = np.linspace(0, 15, xcard)

    # Index of position for agents 1 and 2
    ix1 = np.zeros(N, dtype=int)
    ix2 = np.zeros(N, dtype=int)

    # Generate random index to start
    ix1[0] = random.randint(0, xcard + 1)
    ix2[0] = random.randint(0, xcard + 1)

    # Get positions in real space from indexes
    x1 = x[ix1[-1]]
    x2 = x[ix2[-1]]

    # Compute K and initialize array of K
    LK1, LK2 = compute_Ks(x1, x2)
    K1 = np.zeros(N)
    K1[0] = LK1
    K2 = np.zeros(N)
    K2[0] = LK2

    # Make first exploration
    ix1[1] = (ix1[0] + random.randint(-1, 1))
    ix2[1] = (ix2[0] + random.randint(-1, 1))
    ix1[1] = check_bounds(ix1[1])
    ix2[1] = check_bounds(ix2[1])
    x1 = x[ix1[1]]
    x2 = x[ix2[1]]
    LK1, LK2 = compute_Ks(x1, x2)
    K1[1] = LK1
    K2[1] = LK2

    # Main cycle
    for i in range(2, 100):
        # Get new index based on strategy, ix is passed by reference
        get_strategy(Strategy.EXPLORATORY, i, ix1, K1)
        get_strategy(Strategy.EXPLORATORY, i, ix2, K2)

        x1 = x[ix1[i]]
        x2 = x[ix2[i]]
        LK1, LK2 = compute_Ks(x1, x2)
        K1[i] = LK1
        K2[i] = LK2

    # Update variables for final plot
    data_starting_ix1.append(ix1[0])
    data_starting_ix2.append(ix2[0])
    data_final_ix1.append(ix1[-1])
    data_final_ix2.append(ix2[-1])
    data_final_K1.append(K1[-1])
    data_final_K2.append(K2[-1])

def show_ix_confusion_matrix():

    # Confusion matrix
    ix_cnf = np.zeros((xcard, xcard))
    for i in range(REPETITIONS):
        ix_cnf[data_final_ix2[i], data_final_ix1[i]] += 1

    # Divide by number of repetitions to get probability
    ix_cnf /= REPETITIONS
    ax = sns.heatmap(ix_cnf, annot=True, cmap='Blues')
    ax.set_xlabel("ix1")
    ax.set_ylabel("ix2")
    plt.show()


def enumerate_ks(K1, K2):
    ks = []
    for k in K1:
        if k not in ks:
            ks.append(k)
    for k in K2:
        if k not in ks:
            ks.append(k)
    return ks


show_ix_confusion_matrix()
