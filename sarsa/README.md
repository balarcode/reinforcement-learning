## Sarsa (On-Policy Temporal Difference Control) Algorithm

The Sarsa acronym describes the data used in the update of action value function i.e. S_t, A_t, R_t+1, S_t+1, A_t+1 where 't' denotes time. Sarsa on-policy temporal difference control algorithm performs generalized policy iteration (GPI) which incorporates interaction between policy-evaluation process and policy-improvement process to eventually stabilize to obtain optimal value function and optimal policy. Before updating the action value function for Q(S_t, A_t), the Sarsa algorithm requires the knowledge of next state-action pair (S_t+1, A_t+1). That means, the RL agent has to commit to the next action (A_t+1) in next state (S_t+1) before allowing the algorithm to update the action value function. It uses the concept of temporal difference to wait for one time step to form a target from reward and action value at time, t=t+1 with a discount factor, γ. The resulting one step temporal difference also called as TD error is scaled with Sarsa step size, α to form an update to the action value function. This is called Sarsa prediction which performs policy evaluation. Using the GPI framework, the policy is further improved iteratively in every time step with respect to the update action value function from policy evaluation.

### References

[1] Gymnasium API Documentation: https://gymnasium.farama.org

[2] Making Your Own Custom Environment: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

[3] Markov Decision Process (MDP): https://en.wikipedia.org/wiki/Markov_decision_process

## Citation

Please note that the code and technical details made available are for educational purposes only. The repo is not open for collaboration.

If you happen to use the code from this repo, please cite my user name along with link to my profile: https://github.com/balarcode. Thank you!
