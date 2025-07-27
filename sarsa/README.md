## Sarsa (On-Policy Temporal Difference Control) Algorithm

The Sarsa acronym describes the data used in the update of action value function i.e. S_t, A_t, R_t+1, S_t+1, A_t+1 where 't' denotes time. Sarsa on-policy temporal difference control algorithm performs generalized policy iteration (GPI) which incorporates interaction between policy-evaluation process and policy-improvement process to eventually stabilize to obtain optimal value function and optimal policy. Before updating the action value function for Q(S_t, A_t), the Sarsa algorithm requires the knowledge of next state-action pair (S_t+1, A_t+1). That means, the RL agent has to commit to the next action (A_t+1) in next state (S_t+1) before allowing the algorithm to update the action value function. It uses the concept of temporal difference to wait for one time step to form a target from reward, R_t+1 and action value at time, t=t+1 i.e. Q(S_t+1, A_t+1) with a discount factor, γ. The resulting one step temporal difference also called as TD error is scaled with Sarsa step size, α to form an update to the action value function. This is called Sarsa prediction which performs policy evaluation. Using the GPI framework, the initial policy is further improved iteratively in every time step with respect to the updated action value function from policy evaluation. Sarsa is a sample-based algorithm to solve the Bellman equation for action values.

In example-1, a windy gridworld environment using Gymnasium is implemented and two important results are shown in the figures below. The Sarsa algorithm uses a learning rate, α = 0.5 and ε-greedy policy with ε = 0.1 in an undiscounted episodic MDP environment. The RL agent uses Sarsa algorithm to find a near-optimal policy to select actions from start state, S to reach the terminal or goal state, G. The actions allowed for the RL agent under the policy in example-1 are four of them - up, down, right and left. In the windy gridworld environment, the resultant state after taking an action will get shifted upwards by the wind strength. The wind strength values are shown along x-axis in the second figure below. A wind strength of 2 means that the resultant state gets shifted by 2 positions upwards from the next state visited after selecting an action. In the first figure below, it can be seen that the initial episodes take longer time steps to complete and as the Sarsa algorithm learns more, the curve starts to increase exponentially along y-axis taking lesser time steps to complete one episode. 

### Episodes verses Time Steps (Example-1)

![Example 1_Episodes_versus_Time_Steps](results/example_1_figure_1.png)

### Near-Optimal Policy (π* + ε) (Example-1)

![Example 1 Near_Optimal_Policy](results/example_1_figure_2.png)

### References

[1] Gymnasium API Documentation: https://gymnasium.farama.org

[2] Making Your Own Custom Environment: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

[3] Markov Decision Process (MDP): https://en.wikipedia.org/wiki/Markov_decision_process

## Citation

Please note that the code and technical details made available are for educational purposes only. The repo is not open for collaboration.

If you happen to use the code from this repo, please cite my user name along with link to my profile: https://github.com/balarcode. Thank you!
