####################################################################
# Title     : Sarsa (On-Policy Temporal Difference Control) Algorithm
# Author    : balarcode
# Version   : 1.1
# Date      : 27th July 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Sarsa as it stands for State-Action-Reward-State-Action
#             quintuple is an online reinforcement learning algorithm
#             to learn action values under an ε-greedy policy.
#             It uses the concept of temporal difference to wait for
#             one time step to form a target from reward and action
#             value both at time, t=t+1 with a discount factor, γ.
#             The resulting one step temporal difference also called
#             as TD error is scaled with Sarsa step size, α to form
#             an update to the action value for one time step before,
#             i.e. at time, t=t. It's like learning a guess from
#             another guess.
#             Along with estimating action values for an MDP, the
#             Sarsa algorithm also estimates the policy as either
#             an optimal policy only if state-action pairs are visited
#             infinitely or as a near-optimal policy starting from
#             an ε-greedy policy with exploration or from a uniform
#             random policy with or without exploration starts.
#
# All Rights Reserved.
####################################################################
# %%
# Import Packages and Libraries
import gym
import numpy as np
import matplotlib
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Python Package and Library Version Information:")
print('gym:',        gym.__version__)
print('numpy:',      np.__version__)
print('matplotlib:', matplotlib.__version__)
print('logging:',    logging.__version__)

# %%
# Working Directory
# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
print("\nFiles in {}: {}".format(cwd, files))
working_directory = cwd + "/Python/reinforcement_learning/td_control_sarsa/"
print("Working directory is: {}".format(working_directory))

# %%
# Logging
fh = logging.FileHandler(working_directory + 'sarsa.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.CRITICAL) # logging.CRITICAL, logging.DEBUG, logging.INFO
logger = logging.getLogger('sarsa')
logger.setLevel(logging.CRITICAL)
logger.addHandler(fh)
logger.disabled = True
# logger.info(f"Function name() -- Variable1: {Value1} Variable2: {Value2}")

# %%
# Class Definitions
class WindyGridWorldEnv(gym.Env):
    """ Create Windy Gridworld environment using Gymnasium."""
    size = (10, 7) # Size of the windy gridworld (7 rows and 10 columns)
    S = (0, 3) # Start State
    G = (7, 3) # Terminal State or Goal State
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # Wind Strength

    stop_action = [(0, 0)]
    standard_actions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    kings_action = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]

    observation_space = gym.spaces.MultiDiscrete(size)
    reward_range = (-1, -1)

    def __init__(self, king=False, stop=False, stochastic=False):
        """Initialize WindyGridWorldEnv class."""
        self.king = king
        self.stop = stop
        self.stochastic = stochastic

        # Finite MDP related parameters
        self.state = None
        self.actions = self.standard_actions[:]
        if self.king:
            # Condition-1: Include King's actions to include diagonal moves to create eight possible actions.
            self.actions += self.kings_action
        if self.stop:
            # Condition-2: Include no movement as a stop action to create nine possible actions.
            self.actions += self.stop_action
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Plotting related parameters
        self.arrow = None
        self.ax = None

    def step(self, action):
        """Perform one time step move in an episodic task."""
        assert self.action_space.contains(action)

        # Select an action in current state to move to the next state in the windy gridworld
        delta = self.actions[action]
        state = self.state + np.array(delta)

        # Add strength of the wind to the next state to alter it
        wind = self.wind[self.state[0]]
        if self.stochastic and wind > 0:
            # Condition-3: Effect of wind is made stochastic with 1/3 probability in the direction of the wind.
            wind += np.random.choice([-1, 0, +1])
        state[1] += wind

        # Store state for the next step and calculate arrow for rendering
        state = np.clip(state, 0, self.observation_space.nvec - 1)
        self.arrow = state - self.state
        self.state = state

        # Check for terminal state
        terminated = (state == self.G).all()
        reward = -1
        truncated = False
        info = {}

        assert self.observation_space.contains(state)
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the RL agent and environment."""
        super().reset(seed=seed)
        self.state = np.array(self.S, dtype=int) # Start from Start State, S
        self.arrow = np.array((0, 0), dtype=int)
        self.ax = None
        info = { "state": None }
        return self.state, info

    def render(self, mode='human'):
        """Rendering to show results."""
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Background color by wind strength
            wind = np.vstack([self.wind] * self.size[1])
            self.ax.imshow(wind, aspect='equal', origin='lower', cmap='Wistia')
            self.ax.set_title("Near-Optimal Policy (π* + ε) Estimated by Sarsa (On-Policy Temporal Difference Control) Algorithm")

            # Annotations at start and goal states
            self.ax.annotate("G", self.G, size=25, color='blue', ha='center', va='center')
            self.ax.annotate("S", self.S, size=25, color='blue', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.wind)))
            self.ax.set_xticklabels(self.wind)
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.size[0]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.size[1]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)

        # Arrow pointing from current state to the next state
        if (self.arrow == 0).all():
            patch = mpatches.Circle(self.state, radius=0.05, color='black', zorder=1)
        else:
            patch = mpatches.FancyArrow(*(self.state - self.arrow), *self.arrow, color='black',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
        self.ax.add_patch(patch)

# %%
# Register custom environment to be detected by Gymnasium
gym.envs.registration.register(
    id='Windy-GridWorld-v0',
    entry_point=lambda king, stop, stochastic: WindyGridWorldEnv(king, stop, stochastic),
    kwargs={'king': False, 'stop': False, 'stochastic': False},
    max_episode_steps=5_000,
)

# %%
# Function Definitions
def run_episode(env, policy=None, render=False):
    """Run one episode in the windy gridworld environment following
    the estimated policy by the Sarsa algorithm and return the
    collected rewards."""
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # Initialize the state by resetting the environment
    state, state_info = env.reset()
    if render:
        env.render()

    terminal = False
    rewards = []
    while not terminal:
        state_index = np.ravel_multi_index(state, env.observation_space.nvec)
        # Select greedy action (A) at t=t in current state (S)
        action = np.argmax(policy[state_index])
        # Obtain reward (R) and move to next state (S') at t=t+1
        state, reward, terminal, truncated, info = env.step(action)
        # Save rewards in a list
        rewards.append(reward)
        if render:
            env.render()

    if render:
        plt.show()

    return rewards

def sarsa(env, num_episodes, epsilon=0.1, alpha=0.5, gamma=1.0):
    """Sarsa (On-Policy Temporal Difference Control) Algorithm."""
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # 1 - Initialization
    # Number of available actions and states
    n_action = env.action_space.n
    n_state_index = np.ravel_multi_index((env.observation_space.nvec - 1), env.observation_space.nvec) + 1

    # Initialize action values
    Q = np.zeros([n_state_index, n_action], dtype=float)

    # Initialize policy to be a uniform random policy (initial policy)
    policy = np.ones([n_state_index, n_action], dtype=float) / n_action

    # For plotting purposes
    history = [0] * num_episodes

    # Loop for each episode
    for episode in range(num_episodes):
        # Initialize the state by resetting the environment
        state, state_info = env.reset()
        state_index = np.ravel_multi_index(state, env.observation_space.nvec)

        # 2 - Take Action in Current State at Time, t=t under ε-Greedy Policy
        action = np.random.choice(n_action, p=policy[state_index])

        terminal = False
        while not terminal:
            # Advance one time step (t=t+1) in the episode and check for terminal state
            next_state, reward, terminal, truncated, info = env.step(action)
            next_state_index = np.ravel_multi_index(next_state, env.observation_space.nvec)

            # 3 - Take Next Action at Time Step, t=t+1 under ε-Greedy Policy
            next_action = np.random.choice(n_action, p=policy[next_state_index])

            # 4 - Update Q(S_t, A_t) for the Target Policy
            Q[state_index, action] += alpha * (reward + (gamma * Q[next_state_index, next_action]) - Q[state_index, action])

            # 5 - Policy Improvement
            # NOTE: Update target policy with respect to the current action value function, Q(S_t, A_t).
            eps = epsilon / (episode + 1) # Update ε = 1 / t, where 't' is the number of time steps completed so far in the episode
            policy[state_index, :] = eps / n_action # For non-greedy action
            policy[state_index, np.argmax(Q[state_index])] = 1 - eps + (eps / n_action) # For greedy action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # 6 - Save State and Action for Q Update in the Next Time Step
            state_index = next_state_index
            action = next_action
            history[episode] += 1

    return Q, policy, history

if __name__ == '__main__':
    print(f"\nRunning On-Policy Sarsa Temporal Difference Control Algorithm for Windy Gridworld MDP ...\n")
    env = gym.make('Windy-GridWorld-v0')
    Q, policy, history = sarsa(env, 8000, epsilon=0.1, alpha=0.5, gamma=1.0)

    # Plot the graph of number of episodes v/s number of accumulated time steps taken
    plt.figure()
    plt.xlabel("Time steps"); plt.xlim(0, 8_000)
    plt.ylabel("Episodes"); plt.ylim(0, 170)
    timesteps = np.cumsum([0] + history)
    plt.plot(timesteps, np.arange(len(timesteps)), color='red')
    plt.show()
    matplotlib.rcParams['figure.figsize'] = [10, 10]

    # Run one episode choosing greedy actions from the learnt ε-greedy policy
    # and plot the resulting trajectory in windy gridworld environment from
    # start state, S to terminal state, G
    rewards = run_episode(env, policy, render=True)
    print(f"Episode Length = {len(rewards)}")
