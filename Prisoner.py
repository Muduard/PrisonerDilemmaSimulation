import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# Set seed for repeatability
random.seed(10)


# Enum for strategies
class Strategy(Enum):
    SIMPLE_2MEMORY = 1
    DEFECT = 2
    COOPERATE = 3
    EXPLORATORY_2MEMORY = 4
    SIMPLE_NMEMORY = 5
    EXPLORATORY_NMEMORY = 6
    SIMPLE_NMEMORY_EXP_WEIGHTED = 7
    EXPLORATORY_NMEMORY_EXP_WEIGHTED = 8
    ORACLE_AGENT = 9

# Set number of repetitions with N trials each
REPETITIONS = 1000

# Term for numerical stability
eps = np.exp(-10)

# Learning rate
lr = np.exp(-3)

# Number of positions possible (Quantization of position space)
xcard = 10

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


def get_exploratory_direction(ix):
    return 1


def get_normal_direction(ix):
    if ix[i-1] - ix[i-2] == 0:
        direction = 0
    else:
        direction = ix[i - 1] - ix[i - 2] / np.abs(ix[i - 1] - ix[i - 2])
    return direction


# Updates ix in loss direction
def update_ix(i, ix, loss, direction, directions):

    # Get how much space to the end of the position array
    remaining_path = (xcard - ix[i - 1] - 1) if direction * loss > 0 else ix[i - 1]
    directions.append(direction * loss)
    ix[i] = ix[i - 1] + loss * direction * remaining_path
    ix[i] = check_bounds(ix[i])


# Save all previous losses
losses = []

'''Returns correct strategy based on which one we want
Example: we want only defect for an agent'''


def get_strategy(strategy, i, ix, k, directions):

    if strategy == Strategy.SIMPLE_2MEMORY:
        # Get direction and magnitude by going where K is min
        # Add eps to denominator for numerical stability (K1.max() can be 0)
        loss = (k[i - 2] - k[i - 1]) / (k.max() + eps)
        direction = get_normal_direction(ix)
        update_ix(i, ix, loss, direction, directions)
    elif strategy == Strategy.COOPERATE:
        ix[i] = 0
    elif strategy == Strategy.DEFECT:
        ix[i] = xcard - 1
    elif strategy == Strategy.EXPLORATORY_2MEMORY:
        # If last two Ks are equal, we need to explore
        if np.abs((k[i - 2] - k[i - 1])) <= eps:

            loss = 0.1 / lr if ix[i-1] > xcard/2 else 0.1 / lr
            direction = get_exploratory_direction(ix)
            update_ix(i, ix, loss, direction, directions)
        else:
            # If last two Ks are not equal, we proceed with SIMPLE strategy
            get_strategy(Strategy.SIMPLE_2MEMORY, i, ix, k, directions)
    elif strategy == Strategy.SIMPLE_NMEMORY:

        # Base case
        if len(losses) == 0:
            losses.append((k[0] - k[1]) / (k.max() + eps))
        # Weight term
        alpha = 1/len(losses)
        # Loss is the sum of all previous losses
        loss = 0
        for l_index in range(i-2):

            loss += losses[l_index]
        loss = alpha * loss
        losses.append(loss)
        direction = get_normal_direction(ix)
        update_ix(i, ix, loss, direction, directions)

    elif strategy == Strategy.EXPLORATORY_NMEMORY:
        # If last two Ks are equal, we need to explore (Checking for further equality is useless by induction)
        if np.abs((k[i - 2] - k[i - 1])) <= eps:

            loss = -0.1/lr if ix[i - 1] > xcard / 2 else 0.1/lr
            direction = get_exploratory_direction(ix)
            update_ix(i, ix, loss, direction, directions)
            losses.append(loss)
        else:
            # If last two Ks are not equal, we proceed with SIMPLE strategy
            get_strategy(Strategy.SIMPLE_NMEMORY, i, ix, k, directions)

    elif strategy == Strategy.SIMPLE_NMEMORY_EXP_WEIGHTED:

        # Base case
        if len(losses) == 0:
            losses.append((k[0] - k[1]) / (k.max() + eps))
        # Weight term
        gamma = 0.01
        # Loss is the sum of all previous losses
        loss = 0
        loss_squared = 0
        for l_index in range(i - 3):
            loss += losses[l_index]
            loss_squared += (losses[l_index] ** 2)
        denominator = 1
        if len(losses) > 2:
            denominator = np.abs(gamma * losses[i - 3] + (1 - gamma) * loss_squared)

        loss = loss / np.sqrt(denominator + eps)

        losses.append(loss)
        direction = get_normal_direction(ix)
        update_ix(i, ix, loss, direction, directions)

    elif strategy == Strategy.EXPLORATORY_NMEMORY_EXP_WEIGHTED:
        # If last two Ks are equal, we need to explore (Checking for further equality is useless by induction)
        if np.abs((k[i - 2] - k[i - 1])) <= eps:

            loss = -0.1/lr if ix[i - 1] > xcard / 2 else 0.1/lr
            direction = get_exploratory_direction(ix)
            update_ix(i, ix, loss, direction, directions)
            losses.append(loss)
        else:
            # If last two Ks are not equal, we proceed with SIMPLE strategy
            get_strategy(Strategy.SIMPLE_NMEMORY_EXP_WEIGHTED, i, ix, k, directions)
    elif strategy == Strategy.ORACLE_AGENT:
        # Get direction and magnitude by going where K is min
        # Add eps to denominator for numerical stability (K1.max() can be 0)
        loss = (k[i - 2] - k[i - 1]) / (k.max() + eps)
        direction = get_normal_direction(ix)
        update_ix(i, ix, loss, direction, directions)

# Variables for the final confusion matrix
data_final_ix1 = []
data_final_ix2 = []
data_starting_ix1 = []
data_starting_ix2 = []
data_final_K1 = []
data_final_K2 = []

for j in range(REPETITIONS):

    # Number of trials
    N = 40

    # Generate positions based on sample size
    x = np.linspace(0, 15, xcard)

    # Index of position for agents 1 and 2
    ix1 = np.zeros(N, dtype=int)
    ix2 = np.zeros(N, dtype=int)

    # Generate random index to start
    ix1[0] = random.randint(0, xcard - 1)
    ix2[0] = random.randint(0, xcard - 1)

    # Get positions in real space from indexes
    x1 = x[ix1[0]]
    x2 = x[ix2[0]]

    # Compute K and initialize array of K
    LK1, LK2 = compute_Ks(x1, x2)
    K1 = np.zeros(N)
    K1[0] = LK1
    K2 = np.zeros(N)
    K2[0] = LK2

    # Make first exploration
    ix1[1] = (ix1[0] + random.choice([-1, 1]))
    ix2[1] = (ix2[0] + random.choice([-1, 1]))

    ix1[1] = check_bounds(ix1[1])
    ix2[1] = check_bounds(ix2[1])
    x1 = x[ix1[1]]
    x2 = x[ix2[1]]
    LK1, LK2 = compute_Ks(x1, x2)
    K1[1] = LK1
    K2[1] = LK2

    # Reset losses
    losses = []
    directions = []
    # Main cycle
    for i in range(2, N):
        # Get new index based on strategy, ix is passed by reference
        # To Simulate one player game pass K1 + K2 to the agents to minimize the combined force
        get_strategy(Strategy.EXPLORATORY_2MEMORY, i, ix1, K1, directions)
        get_strategy(Strategy.EXPLORATORY_2MEMORY, i, ix2, K2, directions)

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
    #print(np.array(data_final_K1).max())
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
