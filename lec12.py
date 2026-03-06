
import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math


def compute_discounted_return(rewards, gamma):
    """
    Compute the discounted return G = sum_{t=0}^{T-1} gamma^t * rewards[t].

    In RL, the value of a state under a policy is measured by the expected
    discounted return: the sum of rewards weighted by how far in the future
    they occur. A discount factor of 0 means only the immediate reward matters;
    a factor of 1 gives equal weight to all future rewards.

    Args:
        rewards (list of float): sequence of rewards for one episode
        gamma (float): discount factor in [0, 1]

    Returns:
        float: total discounted return

    Raises:
        ValueError: if rewards is empty
        ValueError: if gamma is not in [0, 1]
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If rewards is empty, raise ValueError("rewards list cannot be empty")
    #   - If gamma is not in [0, 1], raise ValueError("gamma must be in [0, 1]")
    #
    # Step 2: Compute G = sum_{t=0}^{T-1} gamma^t * rewards[t]
    #   Hint: use enumerate(rewards) to get (t, r) pairs.
    #
    # Step 3: Return the total.
    if rewards == []:
        raise ValueError("rewards list cannot be empty")
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")
    sum = 0
    for t, r in enumerate(rewards):
        sum += gamma ** t * r
    return sum


def compute_mc_returns(rewards, gamma):
    """
    Compute per-step Monte Carlo returns for each timestep in an episode.

    G[t] = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...
         = rewards[t] + gamma * G[t+1]   (recursive definition)

    This is the foundation of temporal credit assignment: even though a reward
    at the end of a coding task (e.g., unit-test pass) is delayed, G[0] captures
    its discounted contribution to the very first decision.

    Computed efficiently in reverse order (O(n) time, O(n) space).

    Args:
        rewards (list of float): sequence of rewards for one episode
        gamma (float): discount factor in [0, 1]

    Returns:
        list of float: G[t] for each timestep t (same length as rewards)

    Raises:
        ValueError: if rewards is empty
        ValueError: if gamma is not in [0, 1]
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If rewards is empty, raise ValueError("rewards list cannot be empty")
    #   - If gamma not in [0, 1], raise ValueError("gamma must be in [0, 1]")
    #
    # Step 2: Allocate a result list of the same length as rewards.
    #
    # Step 3: Traverse rewards in REVERSE order, maintaining a running G.
    #   At each step t:
    #     G = rewards[t] + gamma * G
    #     returns[t] = G
    #
    # Step 4: Return the returns list.
    if rewards == []:
        raise ValueError("rewards list cannot be empty")
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")
    returns= [None]*len(rewards)
    G = 0 
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def build_transition(state, action, reward, next_state, done):
    """
    Build a validated RL transition record (s, a, r, s', done).

    In sequential decision making, an agent observes a state, takes an action,
    receives a reward, and transitions to a new state. For an LLM agent:
      state      = current context window
      action     = token generation + tool call (if any)
      reward     = feedback (e.g., unit test result)
      next_state = updated context after the action
      done       = whether the episode (task) is complete

    Args:
        state: current state (must not be None)
        action: action taken (must not be None)
        reward (int | float): reward received
        next_state: next state (may be None only when done=True)
        done (bool): whether the episode terminated after this transition

    Returns:
        dict with keys: "state", "action", "reward" (float), "next_state", "done"

    Raises:
        ValueError: if state is None
        ValueError: if action is None
        ValueError: if reward is not int or float
        TypeError: if done is not bool
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs:
    #   - if state is None  → raise ValueError("state cannot be None")
    #   - if action is None → raise ValueError("action cannot be None")
    #   - if reward is not int or float → raise ValueError("reward must be numeric (int or float)")
    #   - if done is not bool → raise TypeError("done must be a bool")
    #
    # Step 2: Return a dict with keys:
    #   "state", "action", "reward" (cast to float), "next_state", "done"

    if state is None:
        raise ValueError("state cannot be None")
    if action is None:
        raise ValueError("action cannot be None")
    if type(reward) not in [int, float]:
        raise ValueError("reward must be numeric (int or float)")
    if type(done) != bool:
        raise TypeError("done must be a bool")
    
    return {"state": state, 
            "action": action, 
            "reward": float(reward),
            "next_state":  next_state,   
            "done": done}


def classify_policy(action_probs):
    """
    Classify a policy as "deterministic" or "stochastic" from its action distribution.

    The lecture distinguishes between deterministic policies (pi: S → A, always
    the same action for a given state) and stochastic policies (pi: S × A → [0,1],
    which assign probabilities to actions). LLMs implement stochastic policies:
    the autoregressive sampling process produces a probability distribution over
    the next token (action) at each step.

    A policy is "deterministic" if exactly one action has probability 1.0 and
    all others have probability 0.0. Otherwise it is "stochastic".

    Args:
        action_probs (dict): maps action names (str) to probabilities (float).
                             Must be non-empty and sum to approximately 1.0.

    Returns:
        str: "deterministic" or "stochastic"

    Raises:
        ValueError: if action_probs is empty
        ValueError: if probabilities do not sum to approximately 1.0 (tolerance 1e-6)
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If action_probs is empty, raise ValueError("action_probs cannot be empty")
    #   - Compute the sum of all values. If abs(total - 1.0) > 1e-6,
    #     raise ValueError("probabilities must sum to 1.0, got <total>")
    #
    # Step 2: Check if policy is deterministic.
    #   A policy is deterministic if max(probs) is exactly 1.0 and only ONE
    #   action has that probability.
    #
    # Step 3: Return "deterministic" or "stochastic".

    if action_probs == {}:
        raise ValueError("action_probs cannot be empty")
    probs = [n for n in action_probs.values()]
    total = sum(probs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {total}")
    is_deterministic = max(probs) == 1 and len([n for n in action_probs.values() if n == 1]) == 1
    if is_deterministic:
        return "deterministic"
    return "stochastic"



def get_rl_component(component_name):
    """
    Return the LLM-agent equivalent of a standard RL component.

    The lecture provided a concrete mapping for how RL concepts translate
    to LLM agents trained with reinforcement learning:

        RL component   →  LLM-agent equivalent
        ─────────────────────────────────────────────────────────────────
        agent          →  "LLM base model"
        environment    →  "prompt and external environment"
        action         →  "next token generation and tool calling"
        state          →  "current context"
        reward         →  "task feedback"

    Lookup is case-insensitive (e.g., "Agent" and "AGENT" both work).

    Args:
        component_name (str): one of the five RL component names above

    Returns:
        str: the corresponding LLM-agent concept (exact string from the table)

    Raises:
        ValueError: if component_name is not one of the five recognized names
    """
    # TODO: Implement this function.
    #
    # Step 1: Define the MAPPING dict with the 5 entries above.
    #
    # Step 2: Normalize the key: component_name.lower().strip()
    #
    # Step 3: If the key is not in MAPPING, raise ValueError with a message
    #         like: "Unknown component '...'. Expected one of: [...]"
    #
    # Step 4: Return MAPPING[key].

    mapping = {
        "agent": "LLM base model",
        "environment": "prompt and external environment",
        "action": "next token generation and tool calling",
        "state": "current context",
        "reward": "task feedback"
    }
    result = mapping.get(component_name.lower().strip())
    if not result:
        raise ValueError()
    return result 


def evaluate_policy(trajectories, gamma):
    """
    Estimate the value of a policy as the average discounted return
    over a set of episodes (trajectories).

    In RL, value functions quantify the "goodness" of a policy. To estimate
    V(pi), we run the policy for multiple episodes and average the discounted
    returns. This is how you compare two LLM fine-tuning runs: run each policy
    on the same benchmark tasks and compare their average returns (e.g., average
    unit-test pass rate weighted by task difficulty).

    Args:
        trajectories (list of list of float): each inner list is a reward
            sequence [r_0, r_1, ..., r_{T-1}] for one episode
        gamma (float): discount factor in [0, 1]

    Returns:
        float: mean discounted return across all trajectories

    Raises:
        ValueError: if trajectories is empty
        ValueError: if gamma is not in [0, 1]
        ValueError: if any individual trajectory is empty
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate: trajectories not empty, gamma in [0,1].
    #
    # Step 2: For each trajectory, compute its discounted return
    #         G = sum_{t=0}^{T-1} gamma^t * rewards[t]
    #         Raise ValueError if a trajectory is empty.
    #
    # Step 3: Return the average of all discounted returns.
    if trajectories == []:
        raise ValueError()
    returns = [None] * len(trajectories)
    G = 0 
    for t in reversed(range(len(trajectories))):
        G = returns[t] + gamma * G
        returns[t] = G
    return sum(returns) / len(returns)
    


def select_best_policy(policy_values):
    """
    Given a mapping from policy names to their estimated values, return the
    name of the best policy (highest value).

    The lecture noted that value functions allow us to compare policies directly:
    the policy with the higher value function is considered better.

    Args:
        policy_values (dict): maps policy name (str) to its estimated value (float)

    Returns:
        str: the name of the policy with the highest value

    Raises:
        ValueError: if policy_values is empty
    """

    max_score = 0 
    best_policy = None
    if policy_values == None:
        raise ValueError()
    for policy in policy_values.keys():
        score = policy_values[policy]
        if score > max_score:
            best_policy = policy
            max_score = score
    return best_policy


import math


def compute_ppo_ratio(old_log_prob, new_log_prob):
    """
    Compute the PPO probability ratio r = exp(new_log_prob - old_log_prob).

    This ratio measures how much the new policy has moved relative to the old
    policy for a specific action. A ratio > 1 means the new policy assigns
    higher probability to that action; < 1 means lower probability.

    In log space, the ratio becomes exp(log_new - log_old), which is numerically
    stable even for very small probabilities.

    Args:
        old_log_prob (float): log P(action | pi_old) — must be <= 0
        new_log_prob (float): log P(action | pi_new) — must be <= 0

    Returns:
        float: r = exp(new_log_prob - old_log_prob), always > 0

    Raises:
        ValueError: if old_log_prob > 0 or new_log_prob > 0
    """
    # TODO: Implement.
    # Validate both log_probs <= 0 (log of a probability is always non-positive).
    # Return math.exp(new_log_prob - old_log_prob).
    if not old_log_prob <= 0 and new_log_prob <= 0:
        raise ValueError()
    return float(math.exp(new_log_prob - old_log_prob))

def compute_ppo_objective(ratio, advantage, epsilon=0.2):
    """
    Compute the PPO clipped surrogate objective for one (action, advantage) pair.

    The PPO objective prevents dangerously large policy updates by clipping the
    probability ratio. This implements the "trust region" idea from the lecture:
    learning stays safe when the new policy doesn't deviate too far from the old one.

    Steps:
    1. Clipped ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
    2. Return min(ratio * advantage, clipped_ratio * advantage)

    Args:
        ratio (float): probability ratio r = P_new / P_old (must be > 0)
        advantage (float): advantage estimate A(s, a)
        epsilon (float): clipping parameter, must be > 0 (default 0.2)

    Returns:
        float: the PPO clipped surrogate objective value

    Raises:
        ValueError: if ratio <= 0
        ValueError: if epsilon <= 0
    """
    # TODO: Implement.
    # Step 1: Validate ratio > 0 and epsilon > 0.
    # Step 2: Compute clipped = max(1-epsilon, min(1+epsilon, ratio)).
    # Step 3: Return min(ratio * advantage, clipped * advantage).
    if not ratio > 0 and epsilon > 0:
        raise ValueError
    clipped = max(1-epsilon, min(1+epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


def compute_reinforce_loss(log_probs, returns):
    """
    Compute the REINFORCE (Monte Carlo policy gradient) loss.

    REINFORCE is the simplest policy gradient algorithm. The gradient of the
    expected return with respect to the policy parameters is:

        ∇J(θ) = E[ ∇log π(a|s) * G ]

    In practice, we implement this as a loss function for gradient descent
    (rather than ascent) by negating:

        L = -mean( log_prob[t] * G[t] )  for t = 0, ..., T-1

    This is the core training signal for RL-trained LLMs: actions that led to
    high returns (e.g., correctly passing unit tests) are made more probable;
    actions that led to low returns are made less probable.

    Args:
        log_probs (list of float): log probabilities of actions taken at each
                                   timestep, log π(a_t | s_t)
        returns (list of float): per-step discounted returns G[t], typically
                                 from compute_mc_returns()

    Returns:
        float: REINFORCE loss = -mean(log_prob[t] * G[t])

    Raises:
        ValueError: if log_probs or returns is empty
        ValueError: if log_probs and returns have different lengths
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If log_probs or returns is empty, raise ValueError.
    #   - If len(log_probs) != len(returns), raise ValueError with lengths.
    #
    # Step 2: Compute total = sum of log_prob[t] * G[t] for all t.
    #   Use zip(log_probs, returns).
    #
    # Step 3: Return -total / len(log_probs).
    #   (Negate because we minimize loss, not maximize reward.)


    if any(prob == [] for prob in log_probs):
        raise ValueError("")
    if len(log_probs) != len(returns):
        raise ValueError("")
    zipped = zip(log_probs, returns)
    result =   - sum(prob * r for prob, r in zipped) / len(log_probs)
    return result

# log_probs = [-0.5, -1.0]
# returns = [3.0, 1.0]
# zip(log_probs, returns)



def normalize_returns(returns):
    """
    Normalize a list of returns to zero mean and unit standard deviation.

    Return normalization is a standard variance-reduction technique in RL
    training. Raw returns can span very large ranges (e.g., 0 vs 1000), which
    causes numerically unstable gradients. Normalizing to zero mean and unit std
    gives the policy gradient a consistent scale across different episodes and tasks.

    This is the same operation used in practice before computing the REINFORCE
    loss or the PPO advantage estimate.

    Formula:
        mean = sum(returns) / len(returns)
        var  = sum((r - mean)^2 for r in returns) / len(returns)
        std  = sqrt(var)
        normalized[i] = (returns[i] - mean) / std

    Args:
        returns (list of float): raw per-step returns to normalize

    Returns:
        list of float: normalized returns (same length as input)

    Raises:
        ValueError: if returns is empty
        ValueError: if all returns are identical (std = 0, division by zero)
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate — raise ValueError if returns is empty.
    #
    # Step 2: Compute mean = sum(returns) / len(returns).
    #
    # Step 3: Compute variance = sum((r - mean)^2 for r in returns) / len(returns).
    #         If variance == 0.0, raise ValueError (all values identical).
    #
    # Step 4: Compute std = variance ** 0.5.
    #
    # Step 5: Return [(r - mean) / std for r in returns].
    pass