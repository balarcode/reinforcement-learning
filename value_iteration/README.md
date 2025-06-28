## Value Iteration Algorithm to Estimate Optimal Policy and Optimal Value Function

When the policy evaluation algorithm within policy iteration algorithm is truncated to only one sweep across the entire state space, it gives rise to value iteration algorithm. Value iteration algorithm combines truncated policy evaluation and policy improvement algorithms. Note that the policy improvement algorithm performs greedification of the policy, π with respect to the current value function, v_π(s). The truncation for policy evaluation can be achieved by choosing the maximum of the value functions estimated for all actions per state. Under the same conditions for a finite MDP, both multi-sweep and single-sweep policy evaluation algorithms eventually converge to the optimal value function, v* with the latter providing an advantage in terms of computational complexity and processing power required for convergence.

There are two approaches that can be taken to implement value iteration algorithm. First approach is to estimate optimal value function (v*) using iterative truncated policy evaluation algorithm followed by estimating optimal policy (π*) using policy improvement algorithm. Second approach is to perform local expected updates to both policy (π) and value function (v_k(s), k = 0, 1, 2, ...) in one iteration by alternating between local policy improvement algorithm and local truncated policy evaluation algorithm. The implemented code uses the second approach to implement value iteration algorithm.

Note that v* and v_* notations are used interchangeably. Similarly, π* and π_* notations are used interchangeably. 

### Value Estimates and Optimal Policy for Probability = 0.25

Value function estimates for one sweep across the entire state space is shown in the first figure below. The envelope of the value function estimates across all the sweeps provides the optimal value function (v*) for the finite MDP. The corresponding optimal policy (π*) is shown in the second figure below. A probability of heads occuring in a coin flip experiment is set to 0.25.

![example_1_figure_pH_0.25](results/example_1_figure_pH_0.25.png)

### Value Estimates and Optimal Policy for Probability = 0.4

![example_1_figure_pH_0.4](results/example_1_figure_pH_0.4.png)

### Value Estimates and Optimal Policy for Probability = 0.5

![example_1_figure_pH_0.5](results/example_1_figure_pH_0.5.png)

### Value Estimates and Optimal Policy for Probability = 0.55

![example_1_figure_pH_0.55](results/example_1_figure_pH_0.55.png)

### Value Estimates and Optimal Policy for Probability = 0.75

![example_1_figure_pH_0.75](results/example_1_figure_pH_0.75.png)


## Citation

Please note that the code and technical details made available are for educational purposes only. The repo is not open for collaboration.

If you happen to use the code from this repo, please cite my user name along with link to my profile: https://github.com/balarcode. Thank you!
