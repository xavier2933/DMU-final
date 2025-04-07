import numpy as np

def value_iteration(mdp, gamma=0.9, epsilon=1e-6, max_iterations=1000):
    """
    Computes the optimal value function for a given MDP using value iteration.

    Args:
        mdp: A dictionary representing the MDP, with the following keys:
            - 'states': A list of all possible states.
            - 'actions': A list of all possible actions.
            - 'transition_probs': A dictionary of transition probabilities, where keys are (state, action, next_state) tuples and values are probabilities.
            - 'rewards': A dictionary of rewards, where keys are (state, action, next_state) tuples and values are rewards.
        gamma: The discount factor (default: 0.9).
        epsilon: The convergence threshold (default: 1e-6).
        max_iterations: The maximum number of iterations to perform (default: 1000).

    Returns:
        A dictionary representing the optimal value function, where keys are states and values are optimal values.
    """
    values = {s: 0 for s in mdp['states']}

    for i in range(max_iterations):
        delta = 0
        for s in mdp['states']:
            v = values[s]
            max_q = float('-inf')
            for a in mdp['actions']:
                q = 0
                for s_prime in mdp['states']:
                    p = mdp['transition_probs'].get((s, a, s_prime), 0)
                    r = mdp['rewards'].get((s, a, s_prime), 0)
                    q += p * (r + gamma * values[s_prime])
                max_q = max(max_q, q)
            values[s] = max_q
            delta = max(delta, abs(v - values[s]))
        if delta < epsilon:
            break
    return values

# Create an MDP dictionary
mdp = {
    'states': ["0", "1", "2", "Ground"],      # List of all possible states
    'actions': ["transmit left", "transmit right", "downlink"],     # List of all possible actions
    'transition_probs': {
        # Satellite 0 transitions
        ("0", "transmit left", "0"): 1.0,  # Can't transmit left from leftmost satellite
        
        ("0", "transmit right", "0"): 0.1,  # Small chance of staying in place
        ("0", "transmit right", "1"): 0.8,  # High chance of successful transmission
        ("0", "transmit right", "Ground"): 0.1,  # Small chance of failure
        
        ("0", "downlink", "0"): 0.9,
        ("0", "downlink", "Ground"): .1,  # Downlink always goes to ground
        
        # Satellite 1 transitions
        ("1", "transmit left", "0"): 0.8,  # High chance of successful transmission
        ("1", "transmit left", "1"): 0.1,  # Small chance of staying in place
        ("1", "transmit left", "Ground"): 0.1,  # Small chance of failure
        
        ("1", "transmit right", "0"): 0.0,
        ("1", "transmit right", "1"): 0.1,  # Small chance of staying in place
        ("1", "transmit right", "2"): 0.8,  # High chance of successful transmission
        ("1", "transmit right", "Ground"): 0.1,  # Small chance of failure
        
        ("1", "downlink", "1"): 0.5,
        ("1", "downlink", "Ground"): 0.5,  # Downlink always goes to ground
        
        # Satellite 2 transitions
        ("2", "transmit left", "1"): 0.8,  # High chance of successful transmission
        ("2", "transmit left", "2"): 0.2,  # Small chance of staying in place
        
        ("2", "transmit right", "2"): 1.0,  # Can't transmit right from rightmost satellite

        ("2", "downlink", "2"): 0.2,
        ("2", "downlink", "Ground"): 0.8,  # Downlink always goes to ground
        
        # Ground is terminal, no transitions from ground
        # Adding these for completeness
        ("Ground", "transmit left", "Ground"): 1.0,
        ("Ground", "transmit right", "Ground"): 1.0,
        ("Ground", "downlink", "Ground"): 1.0
    },  # Dictionary mapping (state, action, next_state) to probability
    'rewards': {
        # Satellite 0 rewards
        ("0", "transmit left", "Ground"): -10.0,  # Penalty for failed action
        
        ("0", "transmit right", "0"): -1.0,  # Small penalty for staying in place
        ("0", "downlink", "Ground"): 10.0,  # Reward for successful downlink
        
        # Satellite 1 rewards
        ("1", "transmit left", "1"): -1.0,  # Small penalty for staying in place
        ("1", "transmit left", "Ground"): -10.0,  # Penalty for failure
        
        ("1", "transmit right", "1"): -1.0,  # Small penalty for staying in place
        ("1", "transmit right", "Ground"): -10.0,  # Penalty for failure
        
        ("1", "downlink", "Ground"): 10.0,  # Higher reward for middle satellite downlink
        
        # Satellite 2 rewards
        ("2", "transmit left", "2"): -1.0,  # Small penalty for staying in place
        ("2", "transmit left", "Ground"): -10.0,  # Penalty for failure
        
        ("2", "transmit right", "Ground"): -10.0,  # Penalty for failed action
        
        ("2", "downlink", "Ground"): 10.0,  # Highest reward for rightmost satellite downlink
    }
}


# Call the value iteration function with your defined MDP
optimal_values = value_iteration(mdp)

# Print the optimal values for each state
print("Optimal values:")
for state in mdp['states']:
    print(f"State {state}: {optimal_values[state]:.2f}")

# If you want to determine the optimal policy (what action to take in each state)
def get_optimal_policy(mdp, values, gamma=0.9):
    policy = {}
    for s in mdp['states']:
        best_action = None
        best_value = float('-inf')
        
        for a in mdp['actions']:
            value = 0
            for s_prime in mdp['states']:
                p = mdp['transition_probs'].get((s, a, s_prime), 0)
                r = mdp['rewards'].get((s, a, s_prime), 0)
                value += p * (r + gamma * values[s_prime])
            
            if value > best_value:
                best_value = value
                best_action = a
        
        policy[s] = best_action
    
    return policy

# Get and print the optimal policy
optimal_policy = get_optimal_policy(mdp, optimal_values)
print("\nOptimal policy:")
for state in mdp['states']:
    if state != "Ground":  # Skip Ground since it's terminal
        print(f"In state {state}, take action: {optimal_policy[state]}")
