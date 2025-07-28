################################################
# Title     : Policy Iteration Algorithm
# Author    : balarcode
# Version   : 1.2
# Date      : 27th July 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Policy iteration algorithm is an iterative reinforcement
#             learning algorithm for estimating an optimal policy, π_*.
#             The algorithm involves three main stages of operations
#             including initialization, policy evaluation and policy
#             improvement.
#             Policy evaluation uses in-place dynamic programming (DP)
#             algorithm. It makes the value function consistent with
#             the current policy (π) chosen.
#             Policy improvement is used to find one or more improved
#             greedy policies with a strict condition to find a better
#             policy except when the original policy is already optimal.
#             The improved greedy policy is expected to converge to an
#             optimal policy in finite number of iterations.
#             Policy iteration algorithm can be formulated in terms of
#             (1) state-value function or (2) action-value function.
#             Along with optimal policy, an optimal value function is
#             also found as an outcome of policy iteration algorithm.
#
#             In this implementation, car rental management problem is
#             chosen to define a finite Markov decision process with
#             states, actions and initial policy for every state to
#             find optimal policy or policies for an optimal value
#             function.
#
# All Rights Reserved.
################################################
# %%
# Import Packages and Libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy.stats import poisson

# %%
# Working Directory
# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
print("Files in {}: {}".format(cwd, files))
working_directory = cwd + "<Add Relative Path to Python File>"
print("Working directory is: {}".format(working_directory))

# %%
# Logging
fh = logging.FileHandler(working_directory + 'policy_iteration.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.CRITICAL) # logging.CRITICAL, logging.DEBUG, logging.INFO
logger = logging.getLogger('policy_iteration')
logger.setLevel(logging.CRITICAL)
logger.addHandler(fh)
logger.disabled = True
# logger.info(f"Function name() -- Variable1: {Value1} Variable2: {Value2}")

# %%
# Global Declarations
# Initialize parameters required to manage car rental at two locations;
# The conditions or clauses that impact the car rental management
# problem are split into two examples.
MAX_CARS_PER_LOCATION = 20
MAX_CARS_FOR_MOVE = 5
RENTAL_CREDIT_PER_CAR = 10
CAR_MOVE_COST = 2 # Used in condition-2
FREE_CARS_FOR_MOVE_TO_LOCATION_2 = 1 # Used in condition-1
CAR_PARK_COST = 4 # Used in condition-3

# Poisson distribution related parameters
# NOTE-1: Poisson distribution (probability mass function):
#         = (λ^n / n!) e^(-λ)
# NOTE-2: Number of cars requested and returned at each
#         location are Poisson random variables.
LAMBDA_REQUEST = [3, 4] # Requests at locations 1 & 2
LAMBDA_RETURN = [3, 2] # Returns at locations 1 & 2
N_UPPER_LIMIT = 12
PMF_MAPPER = {lambda_expected: [poisson.pmf(num_cars, lambda_expected) for num_cars in range(MAX_CARS_PER_LOCATION + 1)] for lambda_expected in [2, 3, 4]}

# Initialize parameters for finite MDP
GAMMA = 0.9 # Discount rate for return computation

# %%
# Class Definitions
class StateDataStructure:
    def __init__(self, s1, s2):
        """
        State is defined as the number of cars at each
        car rental location at the end of the day.
        Two states corresponding to two locations are
        defined for the finite MDP.
        """
        self.s = [s1, s2]

class RequestReturnDataStructure:
    def __init__(self, req1, preq1, req2, preq2, ret1, pret1, ret2, pret2):
        """
        The number of cars requested and returned at each
        location are Poisson random variables. Create one
        data point which is also a data structure to hold
        both request and return info at two car rental
        locations.
        preq1, preq2, pret1 & pret2: Probabilities drawn
        from a Poisson probability distribution.
        """
        self.data = [req1, preq1, req2, preq2, ret1, pret1, ret2, pret2]

# %%
# Function Definitions
def time_step(a, s, V):
    """
    Each time step in the finite MDP for car rental
    management problem is implemented in this function.
    The key functions to be performed within a single
    time step is enumerated sequentially in numbers.
    Several conditions or clauses that impact the car
    rental management problem are also mentioned.
    """
    # Fetch the states at each locations
    s1 = s[0]
    s2 = s[1]

    # 1 - State Transitions
    # First adjust the state for source location
    net_cars_moved = a
    if a > 0: # Move cars from location 1 to location 2
        s1 -= a # Reduce the number of cars at location 1 by a
        # Condition-1: One of the employees at the first location of car rental
        # rides a bus home each night and lives near the second location.
        # This leads to the transfer of one car to the second location for free.
        if (EXAMPLE == 2): net_cars_moved = max(net_cars_moved - FREE_CARS_FOR_MOVE_TO_LOCATION_2, 0)
    elif a < 0: # Move cars from location 2 to location 1
        s2 -= abs(a) # Reduce the number of cars at location 2 by a

    # Second adjust the state for destination location
    if a > 0: # Move cars from location 1 to location 2
        s2 = min(s2 + a, MAX_CARS_PER_LOCATION)
    elif a < 0: # Move cars from location 2 to location 1
        s1 = min(s1 + abs(a), MAX_CARS_PER_LOCATION)

    # 2 - Compute Expenses or Cost
    # Condition-2: Move the cars between the two locations
    # overnight, at a cost of $2 per car moved.
    expenses = abs(net_cars_moved) * CAR_MOVE_COST

    # Condition-3: If more than 10 cars are kept overnight
    # at a location (after any moving of cars), then an
    # additional cost of $4 must be incurred to use a
    # second parking lot (independent of how many cars
    # are kept there).
    if (EXAMPLE == 2): expenses += ((s1 > 10) + (s2 > 10)) * CAR_PARK_COST

    # 3 - Expected Update Operation or Computation of New Approximate Value Function
    V_s = 0 # Initialize per state in the finite MDP
    for ds in request_return_database():
        # 4 - Incorporate the Environmental Dynamics for Car Rental Management
        # Obtain requested and returned car Poisson random variables
        req1, req2, ret1, ret2 = min(ds.data[0], s1), min(ds.data[2], s2), ds.data[4], ds.data[6]

        # Probability of each time step or transition in the finite MDP
        prob = ds.data[1] * ds.data[3] * ds.data[5] * ds.data[7]

        # Net state after accounting for requests and returns
        state1 = min(s1 - req1 + ret1, MAX_CARS_PER_LOCATION)
        state2 = min(s2 - req2 + ret2, MAX_CARS_PER_LOCATION)

        # 5 - Reward
        # Condition-4: If car rental has a car available, then it is rented
        # out to receive a credit of $10 by the national company. This revenue
        # is the reward and it applies to only rental requests.
        R = (min(s1, ds.data[0]) + min(s2, ds.data[2])) * RENTAL_CREDIT_PER_CAR

        # 6 - Return
        G = R + (GAMMA * V[state1, state2])

        # Expected update to produce the new approximate value function per state in the finite MDP
        V_s += (prob * G)

    # Account for expenses or cost incurred per state in the finite MDP
    V_s -= expenses
    return V_s

def states():
    """
    Create a set or database to hold state information
    for two car rental locations.
    """
    for num_location_1 in range(MAX_CARS_PER_LOCATION + 1): # Plus 1 added to include the value of MAX_CARS_PER_LOCATION
        for num_location_2 in range(MAX_CARS_PER_LOCATION + 1):
            yield StateDataStructure(num_location_1, num_location_2)

def request_return_database():
    """
    Create a set or database to hold both request and
    return info at two car rental locations.
    """
    for req1 in range(N_UPPER_LIMIT):
        for req2 in range(N_UPPER_LIMIT):
            for ret1 in range(N_UPPER_LIMIT):
                for ret2 in range(N_UPPER_LIMIT):
                    yield RequestReturnDataStructure(req1, PMF_MAPPER.get(LAMBDA_REQUEST[0])[req1],
                                                     req2, PMF_MAPPER.get(LAMBDA_REQUEST[1])[req2],
                                                     ret1, PMF_MAPPER.get(LAMBDA_RETURN[0])[ret1],
                                                     ret2, PMF_MAPPER.get(LAMBDA_RETURN[1])[ret2])

def policy_evaluation(V, pi):
    """
    Iterative policy evaluation algorithm for estimating
    a new approximate value function, V(s).
    It uses in-place dynamic programming algorithm.
    The previous value function is obtained from previous
    iteration of policy iteration algorithm for every state, s.
    """
    function_start_time = time.time()
    theta = 0.0001 # Stopping parameter or threshold for convergence and improving estimation accuracy
    delta = 1 # Start with a value that is maximum enough to run the iterations towards convergence
    iteration = 0 # Iteration count for policy evaluation algorithm

    while delta >= theta:
        loop_start_time = time.time()
        delta = 0
        iteration += 1

        for s in states():
            v = V[s.s[0], s.s[1]] # Value function from previous iteration of policy iteration algorithm for every state, s
            a = pi[s.s[0], s.s[1]]
            V_s = time_step(a, s.s, V)
            delta = max(delta, abs(V_s - v))
            V[s.s[0], s.s[1]] = V_s # Save the new approximate value function
        print(f"policy_evaluation() iteration: {iteration}, delta: {delta}, time: {time.time() - loop_start_time} seconds")
    print(f"policy_evaluation() overall execution time: {time.time() - function_start_time} seconds")

def policy_improvement(V, pi):
    """
    Policy improvement algorithm to find an improved &
    greedy policy which eventually should converge to
    an optimal policy for the finite MDP under consideration.
    The policy is found by maximizing the value function for
    a particular action given by argmax.
    """
    function_start_time = time.time()
    policy_stable = True

    for s in states():
        old_action = pi[s.s[0], s.s[1]]
        actions = np.arange(-MAX_CARS_FOR_MOVE, (MAX_CARS_FOR_MOVE + 1)) # Negative actions transfer to location 1
        V_s_list = []

        # Keep accumulating V_s i.e. state-value functions for all actions for the current state
        for a in actions:
            if (0 <= a <= s.s[0]) or (-s.s[1] <= a <= 0):
                V_s = time_step(a, s.s, V)
                V_s_list.append(V_s)
            else:
                V_s_list.append(-1e+10) # Assign a low value to V_s for the current state due to invalid action

        # Apply argmax to the accumulated list of V_s to find the action that maximizes the value function
        new_action = actions[np.argmax(V_s_list)]
        pi[s.s[0], s.s[1]] = new_action # Save the maximizing action which should give the greedy policy (π')

        if new_action != old_action:
            policy_stable = False

    print(f"policy_improvement() overall execution time: {time.time() - function_start_time} seconds")
    return policy_stable

def policy_iteration():
    """
    Policy iteration algorithm.
    """
    # 1 - Initialization
    V  = np.zeros((MAX_CARS_PER_LOCATION + 1, MAX_CARS_PER_LOCATION + 1))
    pi = np.zeros((MAX_CARS_PER_LOCATION + 1, MAX_CARS_PER_LOCATION + 1), dtype=int)

    # Plot the initial policy (π_0)
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    plot_policy(pi, axes, iteration=0)

    policy_stable = False
    iteration = 0 # Iteration count for policy iteration algorithm
    while (not policy_stable):
        iteration += 1

        # 2 - Policy Evaluation
        policy_evaluation(V, pi)

        # 3 - Policy Improvement
        policy_stable = policy_improvement(V, pi)

        # Print the first six policies
        if (iteration < 6): plot_policy(pi, axes, iteration)
        print(f"Policy Iteration Completed for Iteration: {iteration}, Policy Stable: {policy_stable}\n")

    filename = working_directory + 'example_' + str(EXAMPLE) + '_figure_1.png'
    plt.savefig(filename)
    # plt.show() # Use it to view the generated plots runtime. Make sure to call it after plt.savefig().
    plt.close()

    # Policy is stable and hence plot the optimal policy and optimal value function
    print(f"Policy Iteration Algorithm Completed for Example-{EXAMPLE}! (Policy Stable: {policy_stable})\n")
    _, axes = plt.subplots(1, 2, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    plot_policy(pi, axes, iteration=-1)
    plot_value_function(V, axes)
    filename = working_directory + 'example_' + str(EXAMPLE) + '_figure_2.png'
    plt.savefig(filename)
    # plt.show() # Use it to view the generated plots runtime. Make sure to call it after plt.savefig().
    plt.close()

def plot_value_function(V, axes):
    """
    Plot the estimated & updated value function (V_π)
    at the end of policy iteration algorithm.
    """
    fig = sns.heatmap(np.flipud(V), cmap="Spectral", ax=axes[-1])
    fig.set_ylabel('#Cars at first location', fontsize=16)
    fig.set_yticks(list(reversed(range(MAX_CARS_PER_LOCATION + 1))))
    fig.set_xlabel('#Cars at second location', fontsize=16)
    fig.set_title('Optimal Value Function (V_*)', fontsize=16)

def plot_policy(pi, axes, iteration):
    """
    Plot the estimated & improved policy (π) for every
    iteration of policy iteration algorithm.
    """
    if (iteration == -1):
        fig = sns.heatmap(np.flipud(pi), cmap="coolwarm", ax=axes[0])
    else:
        fig = sns.heatmap(np.flipud(pi), cmap="coolwarm", ax=axes[iteration])
    fig.set_ylabel('#Cars at first location', fontsize=16)
    fig.set_yticks(list(reversed(range(MAX_CARS_PER_LOCATION + 1))))
    fig.set_xlabel('#Cars at second location', fontsize=16)
    if (iteration == -1):
        fig.set_title('Optimal Policy (π_*)', fontsize=16)
    else:
        fig.set_title('Policy (π_{})'.format(iteration), fontsize=16)

def example_1():
    """
    Example-1 uses conditions or clauses: 2 and 4.
    """
    print(f"Starting Policy Iteration Algorithm for Example-1.\n")
    global EXAMPLE
    EXAMPLE = 1
    policy_iteration()

def example_2():
    """
    Example-2 uses conditions or clauses: 1 through 4.
    """
    print(f"Starting Policy Iteration Algorithm for Example-2.\n")
    global EXAMPLE
    EXAMPLE = 2
    policy_iteration()

# %%
# Policy Iteration Algorithm
if __name__ == '__main__':
    print(f"Running Policy Iteration Algorithm for Car Rental Finite MDP ...\n")
    example_1()
    example_2()
