################################################
# Title     : Value Iteration Algorithm
# Author    : balarcode
# Version   : 1.1
# Date      : 27th July 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  :
#
# All Rights Reserved.
################################################
# %%
# Import Packages and Libraries
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set the 'Agg' backend
import matplotlib.pyplot as plt
import os
import logging

# %%
# Working Directory
# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
print("Files in {}: {}".format(cwd, files))
working_directory = cwd + "/Python/rl_value_iteration/"
print("Working directory is: {}".format(working_directory))

# %%
# Logging
fh = logging.FileHandler(working_directory + 'value_iteration.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.CRITICAL) # logging.CRITICAL, logging.DEBUG, logging.INFO
logger = logging.getLogger('value_iteration')
logger.setLevel(logging.CRITICAL)
logger.addHandler(fh)
logger.disabled = True
# logger.info(f"Function name() -- Variable1: {Value1} Variable2: {Value2}")

# %%
# Global Declarations and Configurations
# Probabilities for head appearing when coin is flipped
# Used to conduct different experiments of the gambler's problem
pH_list = [0.25, 0.4, 0.5, 0.55, 0.75]

# Initialize parameters for finite MDP
GAMMA = 1.0 # Discount rate for return computation
R = 0 # Reward is zero on all transitions except when the goal is reached
GOAL = 100 # Goal of $100 for winning the betting game
STATES = np.arange(GOAL + 1) # States including terminal states, state 0 and state 100

# %%
# Function Definitions
def bellman_optimality_update(actions, s, p, V):
    """
    Update state-value function, V according to the Bellman optimality update equation.
    """
    V_s_list = []
    for a in actions:
        # Compute the expression: { Σ p(s', r | s, a) [r + γ . v_π(s')] } per state and per action
        V_s = (p * (R + GAMMA * V[s + a])) + ((1 - p) * (R + GAMMA * V[s - a])) # HEADS + TAILS
        V_s_list.append(V_s)

    # Select the maximum value function for current state, s
    V_s_max = np.max(V_s_list)
    return V_s_max

def q_greedify_policy(actions, s, p, V, pi):
    """
    Update pi to be greedy with respect to the q-values induced by state-value function, V.
    """
    # Create a list to store q_π(s, a) induced by v_π(s) for all actions in current state, s
    V_s_list = []
    for a in actions:
        V_s = (p * (R + GAMMA * V[s + a])) + ((1 - p) * (R + GAMMA * V[s - a])) # HEADS + TAILS
        V_s_list.append(V_s)

    # Find the maximizing action over q_π(s, a)
    pi[s] = actions[np.argmax(np.round(V_s_list[1:], 5)) + 1]

def value_iteration(pH):
    """
    Value iteration algorithm that alternates between
    local policy greedifications and local truncated policy evaluations.
    """
    function_start_time = time.time()

    # 1 - Initialization
    V  = np.zeros(GOAL + 1) # V_π(s), State-value function for policy, π
    pi = np.zeros(GOAL + 1) # π(a | s)
    theta = 1e-9 # Stopping parameter or threshold for convergence and improving estimation accuracy
    delta = 1 # Start with a value that is maximum enough to run the iterations towards convergence

    # The reward is zero on all transitions except on those on which the
    # gambler reaches his goal for which the reward is +1
    V[GOAL] = 1.0

    sweeps_history = [] # Used for plotting
    iteration = 0 # Iteration count for value iteration algorithm

    while True:
        loop_start_time = time.time()
        iteration += 1

        v = V.copy()
        sweeps_history.append(v)

        for s in STATES[1:GOAL]:
            # All possible actions for the current state
            actions = np.arange(min(s, GOAL - s) + 1)

            # 2 - Greedification of Policy (Local Policy Improvement)
            q_greedify_policy(actions, s, pH, V, pi)

            # 3 - Local Truncated Policy Evaluation
            # NOTE: Policy (π) is not specified for performing expected update of value function.
            V_s_max = bellman_optimality_update(actions, s, pH, V)
            V[s] = V_s_max # Save the local new approximate value function (i.e. expected update) per state

            # 4 - Compute error
            delta = abs(V - v).max()
        print(f"value_iteration() - iteration: {iteration}, delta: {delta}, time: {time.time() - loop_start_time} seconds")

        # 5 - Evaluation for Convergence
        if delta < theta:
            sweeps_history.append(V)
            break

    print(f"Value Iteration Completed at Iteration: {iteration} for Probability: {pH}, Overall Execution Time: {time.time() - function_start_time} seconds\n")
    plot_results(pH, sweeps_history, pi)

def plot_results(pH, sweeps_history, pi):
    """
    Plot the estimated value functions for multiple
    sweeps across the state space and the estimated
    optimal policy for the finite MDP.
    """
    plt.figure(figsize=(10, 20))

    # Plot value function estimates
    plt.subplot(2, 1, 1)
    for sweep, v_s in enumerate(sweeps_history):
        if (sweep < 20):
            plt.plot(v_s, label='sweep {}'.format(sweep))
        elif (len(sweeps_history) - sweep <= 10):
            plt.plot(v_s, label='sweep {}'.format(sweep))
        else:
            plt.plot(v_s)
    plt.xlabel('States, s (Capital)', fontsize=16)
    plt.ylabel('Value Estimates - v_k(s) for k = 0, 1, 2, ...', fontsize=16)
    plt.legend(loc='best')

    # Plot optimal policy estimate
    plt.subplot(2, 1, 2)
    plt.bar(range(len(pi)), pi)
    plt.xlabel('States, s (Capital)', fontsize=16)
    plt.ylabel('Optimal Policy - π_* (Stake)', fontsize=16)

    filename = working_directory + 'example_' + str(EXAMPLE) +  '_figure_pH_' + str(pH) + '.png'
    plt.savefig(filename)
    plt.close()

def example_1():
    """
    Example-1 solves the gambler's problem.
    """
    print(f"Starting Value Iteration Algorithm for Example-1.\n")
    global EXAMPLE
    EXAMPLE = 1
    for pH in pH_list:
        value_iteration(pH)

if __name__ == '__main__':
    example_1()
