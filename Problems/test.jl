using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat
using Statistics
using Random

"""
    MultiPacketSatelliteNetworkPOMDP

A decentralized POMDP modeling a network of satellites that need to coordinate
to transmit multiple data packets to the ground. Each satellite can pass data to neighbors
or attempt to transmit to ground with varying success probabilities.
"""

# Basic node structure for finite state controllers
struct FSCNode
    action::Int  # Index of the action to take
    transitions::Dict{String, Int}  # Observation -> next node mapping
end

struct AgentController
    nodes::Vector{FSCNode}
end

struct JointController
    controllers::Vector{AgentController}
end

struct MultiPacketSatelliteNetworkPOMDP <: POMDP{String, Tuple{Vararg{String}}, Tuple{Vararg{String}}}
    num_satellites::Int                # Number of satellites in the network
    total_packets::Int                 # Total number of data packets in the system
    max_capacity::Int                  # Maximum packets a satellite can hold
    discount_factor::Float64           # Discount factor for future rewards
    ground_tx_probs::Vector{Float64}   # Probability of successful ground transmission for each satellite
    pass_success_prob::Float64         # Probability of successful data passing between satellites
    observation_accuracy::Float64      # Probability of observing packets correctly
    successful_tx_reward::Float64      # Reward for successful ground transmission (per packet)
    unsuccessful_tx_penalty::Float64   # Penalty for unsuccessful ground transmission (per packet)
    pass_cost::Float64                 # Cost for passing data between satellites (per packet)
    congestion_penalty::Float64        # Penalty for having too many packets at one satellite
end

# Default constructor
function MultiPacketSatelliteNetworkPOMDP(;
    num_satellites = 3,
    total_packets = 5,
    max_capacity = 3,
    discount_factor = 0.9,
    ground_tx_probs = [0.3, 0.5, 0.7],
    pass_success_prob = 0.95,
    observation_accuracy = 0.8,
    successful_tx_reward = 10.0,
    unsuccessful_tx_penalty = -2.0,
    pass_cost = -1.0,
    congestion_penalty = -5.0
)
    # Ensure ground_tx_probs has the correct length
    if length(ground_tx_probs) < num_satellites
        # Extend with random values if too short
        append!(ground_tx_probs, rand(num_satellites - length(ground_tx_probs)))
    elseif length(ground_tx_probs) > num_satellites
        # Truncate if too long
        ground_tx_probs = ground_tx_probs[1:num_satellites]
    end
    
    return MultiPacketSatelliteNetworkPOMDP(
        num_satellites,
        total_packets,
        max_capacity,
        discount_factor,
        ground_tx_probs,
        pass_success_prob,
        observation_accuracy,
        successful_tx_reward,
        unsuccessful_tx_penalty,
        pass_cost,
        congestion_penalty
    )
end

# Helper function to generate all ways to distribute packets
function partition_packets(num_satellites, total_packets, max_per_sat=nothing)
    if max_per_sat === nothing
        max_per_sat = total_packets
    end
    
    partitions = []
    
    function generate_partitions(remaining, current_idx, current_partition)
        if current_idx > num_satellites
            if remaining == 0
                push!(partitions, copy(current_partition))
            end
            return
        end
        
        # Try different amounts of packets for current satellite
        for i in 0:min(remaining, max_per_sat)
            current_partition[current_idx] = i
            generate_partitions(remaining - i, current_idx + 1, current_partition)
        end
    end
    
    generate_partitions(total_packets, 1, zeros(Int, num_satellites))
    return partitions
end

# Helper function to parse state string into packet counts
function parse_state(s::String)
    if s == "all-transmitted"
        return Int[]
    else
        return [parse(Int, x) for x in split(s, "-")]
    end
end

# Helper function to calculate binomial probability
function binomial_probability(n, k, p)
    return binomial(n, k) * p^k * (1-p)^(n-k)
end

# Define state space - how packets are distributed among satellites
function POMDPs.states(m::MultiPacketSatelliteNetworkPOMDP)
    # Generate all possible distributions of packets
    data_distributions = []
    
    # Always include the initial state first (all packets at satellite 1)
    initial_distribution = zeros(Int, m.num_satellites)
    initial_distribution[1] = m.total_packets
    push!(data_distributions, join(initial_distribution, "-"))
    
    # Generate other possible distributions
    for partition in partition_packets(m.num_satellites, m.total_packets, m.max_capacity)
        # Don't add the initial state twice
        state_str = join(partition, "-")
        if state_str != data_distributions[1]
            push!(data_distributions, state_str)
        end
    end
    
    # Add terminal state for when all data is transmitted
    push!(data_distributions, "all-transmitted")
    
    return data_distributions
end

# Define agent actions
function POMDPs.actions(m::MultiPacketSatelliteNetworkPOMDP)
    # Generate all combinations of actions for all satellites
    joint_actions = Vector{NTuple{m.num_satellites, String}}()
    
    # Get individual actions for each satellite based on position
    sat_actions_list = []
    for sat_idx in 1:m.num_satellites
        sat_actions = ["wait"]  # All satellites can wait
        
        # Transmit actions for different packet counts
        for i in 1:m.max_capacity
            push!(sat_actions, "transmit-$i")
        end
        
        # Pass actions depend on satellite position
        if sat_idx > 1  # Can pass left if not leftmost
            for i in 1:m.max_capacity
                push!(sat_actions, "pass-left-$i")
            end
        end
        
        if sat_idx < m.num_satellites  # Can pass right if not rightmost
            for i in 1:m.max_capacity
                push!(sat_actions, "pass-right-$i")
            end
        end
        
        push!(sat_actions_list, sat_actions)
    end
    
    # Generate all combinations recursively
    current_actions = ["wait" for _ in 1:m.num_satellites]
    
    function generate_actions(sat_idx)
        if sat_idx > m.num_satellites
            push!(joint_actions, Tuple(current_actions))
            return
        end
        
        for action in sat_actions_list[sat_idx]
            current_actions[sat_idx] = action
            generate_actions(sat_idx + 1)
        end
    end
    
    generate_actions(1)
    return joint_actions
end

# Define observation space
function POMDPs.observations(m::MultiPacketSatelliteNetworkPOMDP)
    # Each satellite observes a packet distribution with possible errors
    # To keep things manageable, we'll use a simplified representation
    # where each observation is just a string describing packet distribution
    
    # First, define the possible observations for a single satellite
    # For simplicity, each satellite observes a distribution like "0-2-1"
    single_sat_obs = []
    
    # Allow for some error in observations (observing non-existent packets)
    max_obs_packets = m.total_packets + m.num_satellites
    
    # Generate all possible packet distribution observations
    # This could be very large, so we'll limit to valid distributions
    for partition in partition_packets(m.num_satellites, max_obs_packets, max_obs_packets)
        push!(single_sat_obs, join(partition, "-"))
    end
    
    # Joint observations are tuples of individual observations
    # For practicality, we won't enumerate all possible combinations
    # Instead, we'll generate them dynamically in the observation function
    
    # For the purpose of the POMDP interface, return a minimal set
    # The actual distribution will be handled in the observation function
    joint_obs = Vector{NTuple{m.num_satellites, String}}()
    
    # Just for interface compatibility - return a small subset
    # Generate a few sample observations
    for _ in 1:min(100, length(single_sat_obs)^m.num_satellites)
        obs_tuple = Tuple(rand(single_sat_obs) for _ in 1:m.num_satellites)
        push!(joint_obs, obs_tuple)
    end
    
    # Always include the "all-transmitted" observation
    empty_obs = join(zeros(Int, m.num_satellites), "-")
    push!(joint_obs, Tuple(fill(empty_obs, m.num_satellites)))
    
    return joint_obs
end

# Define initial state distribution
function POMDPs.initialstate(m::MultiPacketSatelliteNetworkPOMDP)
    # Initial state: all packets at the first satellite
    initial_distribution = zeros(Int, m.num_satellites)
    initial_distribution[1] = m.total_packets
    return Deterministic(join(initial_distribution, "-"))
end

# Define discount factor
function POMDPs.discount(m::MultiPacketSatelliteNetworkPOMDP)
    return m.discount_factor
end

# Define transition function
function POMDPs.transition(m::MultiPacketSatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})
    # If all data transmitted, stay in terminal state
    if s == "all-transmitted"
        return Deterministic("all-transmitted")
    end
    
    # Parse current packet distribution
    packet_counts = parse_state(s)
    
    # Process each satellite's action
    new_packet_counts = copy(packet_counts)
    next_states = String[]
    next_probs = Float64[]
    
    # Process ground transmissions first
    for sat_idx in 1:m.num_satellites
        action = a[sat_idx]
        
        if startswith(action, "transmit-")
            # Extract number of packets to transmit
            num_packets = parse(Int, split(action, "-")[2])
            
            # Can't transmit more than you have
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                # Create all possible transmission outcomes
                for success_count in 0:actual_packets
                    # Calculate probability of this outcome
                    prob = binomial_probability(actual_packets, success_count, m.ground_tx_probs[sat_idx])
                    
                    # Create new state with successful transmissions removed
                    temp_counts = copy(new_packet_counts)
                    temp_counts[sat_idx] -= success_count
                    
                    # Check if all packets transmitted
                    if sum(temp_counts) == 0
                        next_state = "all-transmitted"
                    else
                        next_state = join(temp_counts, "-")
                    end
                    
                    push!(next_states, next_state)
                    push!(next_probs, prob)
                end
                
                # Update packet count for subsequent actions
                new_packet_counts[sat_idx] -= actual_packets
            end
        end
    end
    
    # If no transmissions occurred, process packet passing
    if isempty(next_states)
        # Initialize with current state
        temp_packet_counts = copy(packet_counts)
        changes_made = false
        
        # Process each satellite's pass actions
        for sat_idx in 1:m.num_satellites
            action = a[sat_idx]
            
            if startswith(action, "pass-")
                parts = split(action, "-")
                direction = parts[2]
                num_packets = parse(Int, parts[3])
                
                # Can't pass more than you have
                actual_packets = min(num_packets, temp_packet_counts[sat_idx])
                
                if actual_packets > 0
                    changes_made = true
                    
                    # Determine target satellite
                    target_idx = -1
                    if direction == "left" && sat_idx > 1
                        target_idx = sat_idx - 1
                    elseif direction == "right" && sat_idx < m.num_satellites
                        target_idx = sat_idx + 1
                    else
                        continue  # Invalid direction
                    end
                    
                    # Calculate success probability for each packet
                    pass_prob = m.pass_success_prob
                    
                    # Generate all possible outcomes
                    for success_count in 0:actual_packets
                        # Calculate probability of this outcome
                        prob = binomial_probability(actual_packets, success_count, pass_prob)
                        
                        # Create new state with successful passes
                        temp_counts = copy(temp_packet_counts)
                        temp_counts[sat_idx] -= success_count
                        temp_counts[target_idx] += success_count
                        
                        next_state = join(temp_counts, "-")
                        push!(next_states, next_state)
                        push!(next_probs, prob)
                    end
                    
                    # Update for subsequent actions
                    # Assume all packets passed for simplicity
                    temp_packet_counts[sat_idx] -= actual_packets
                    temp_packet_counts[target_idx] += actual_packets
                end
            end
        end
        
        # If no changes were made, stay in the same state
        if !changes_made
            push!(next_states, s)
            push!(next_probs, 1.0)
        end
    end
    
    # Normalize probabilities
    if !isempty(next_probs)
        next_probs = next_probs ./ sum(next_probs)
    else
        # If no actions affected packets, stay in same state
        push!(next_states, s)
        push!(next_probs, 1.0)
    end
    
    return SparseCat(next_states, next_probs)
end

# Define observation function
function POMDPs.observation(m::MultiPacketSatelliteNetworkPOMDP, a::Tuple{Vararg{String}}, sp::String)
    # If all data transmitted, observation is simple
    if sp == "all-transmitted"
        empty_obs = join(zeros(Int, m.num_satellites), "-")
        return Deterministic(Tuple(fill(empty_obs, m.num_satellites)))
    end
    
    # Parse true packet distribution
    true_packet_counts = parse_state(sp)
    
    # For simplicity, we'll generate a small set of possible observations
    # with probabilities based on observation accuracy
    possible_obs = []
    obs_probs = []
    
    # Generate the "perfect observation" (all satellites see true distribution)
    true_obs_str = sp
    perfect_obs = Tuple(fill(true_obs_str, m.num_satellites))
    
    # Probability of perfect observation
    perfect_prob = m.observation_accuracy^m.num_satellites
    push!(possible_obs, perfect_obs)
    push!(obs_probs, perfect_prob)
    
    # Generate some noisy observations
    n_samples = min(20, m.num_satellites * 2)  # Limit number of samples
    remaining_prob = 1.0 - perfect_prob
    
    for _ in 1:n_samples
        satellite_observations = []
        
        for sat_idx in 1:m.num_satellites
            # Each satellite has some probability of observing correctly
            if rand() < m.observation_accuracy
                # Correct observation
                push!(satellite_observations, true_obs_str)
            else
                # Noisy observation
                noisy_counts = copy(true_packet_counts)
                
                # Add noise to each count
                for i in 1:length(noisy_counts)
                    # Add random noise (-1, 0, or +1)
                    noise = rand(-1:1)
                    noisy_counts[i] = max(0, noisy_counts[i] + noise)
                end
                
                push!(satellite_observations, join(noisy_counts, "-"))
            end
        end
        
        obs_tuple = Tuple(satellite_observations)
        push!(possible_obs, obs_tuple)
        push!(obs_probs, remaining_prob/n_samples)
    end
    
    return SparseCat(possible_obs, obs_probs)
end

# Define reward function
function POMDPs.reward(m::MultiPacketSatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})
    # If already in terminal state, no reward
    if s == "all-transmitted"
        return 0.0
    end
    
    # Parse packet distribution
    packet_counts = parse_state(s)
    
    total_reward = 0.0
    
    # Calculate rewards for each satellite's action
    for sat_idx in 1:m.num_satellites
        action = a[sat_idx]
        
        if startswith(action, "transmit-")
            # Extract number of packets to transmit
            num_packets = parse(Int, split(action, "-")[2])
            
            # Can't transmit more than you have
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                # Expected reward based on transmission probability
                tx_prob = m.ground_tx_probs[sat_idx]
                expected_success = actual_packets * tx_prob
                expected_failure = actual_packets * (1 - tx_prob)
                
                total_reward += expected_success * m.successful_tx_reward
                total_reward += expected_failure * m.unsuccessful_tx_penalty
            end
        elseif startswith(action, "pass-")
            parts = split(action, "-")
            num_packets = parse(Int, parts[3])
            
            # Can't pass more than you have
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                # Cost for passing packets
                total_reward += actual_packets * m.pass_cost
            end
        end
    end
    
    # Penalty for congestion (satellites with too many packets)
    for count in packet_counts
        if count > m.max_capacity
            total_reward += (count - m.max_capacity) * m.congestion_penalty
        end
    end
    
    return total_reward
end

# Helper functions for working with individual agents

# Get individual agent actions based on position
function agent_actions(m::MultiPacketSatelliteNetworkPOMDP, sat_idx::Int)
    actions = ["wait"]  # All satellites can wait
    
    # Transmit actions
    for i in 1:m.max_capacity
        push!(actions, "transmit-$i")
    end
    
    # Pass actions depend on satellite position
    if sat_idx > 1  # Can pass left if not leftmost
        for i in 1:m.max_capacity
            push!(actions, "pass-left-$i")
        end
    end
    
    if sat_idx < m.num_satellites  # Can pass right if not rightmost
        for i in 1:m.max_capacity
            push!(actions, "pass-right-$i")
        end
    end
    
    return actions
end

# General agent actions function for controller generation
function agent_actions(m::MultiPacketSatelliteNetworkPOMDP)
    # This returns a superset of all possible actions any satellite might take
    actions = ["wait"]
    
    # Add transmit actions
    for i in 1:m.max_capacity
        push!(actions, "transmit-$i")
    end
    
    # Add pass actions
    for i in 1:m.max_capacity
        push!(actions, "pass-left-$i")
        push!(actions, "pass-right-$i")
    end
    
    return actions
end

# Get observation strings for a given packet count
function agent_observations(m::MultiPacketSatelliteNetworkPOMDP)
    # For simplicity, return a subset of possible observations
    # The real observation function will handle the probabilities
    
    obs = []
    
    # Add empty network observation
    push!(obs, join(zeros(Int, m.num_satellites), "-"))
    
    # Add some common observations
    for p in partition_packets(m.num_satellites, m.total_packets, m.max_capacity)
        push!(obs, join(p, "-"))
    end
    
    return obs
end

# Function to create an initial controller for policy iteration
function create_initial_controllers(m::MultiPacketSatelliteNetworkPOMDP)
    # Create a controller for each satellite
    controllers = AgentController[]
    
    for sat_idx in 1:m.num_satellites
        # Get available actions for this satellite
        sat_actions = agent_actions(m, sat_idx)
        obs_list = agent_observations(m)
        
        # Create a simple controller with nodes for different scenarios
        nodes = FSCNode[]
        
        # Create a node for when satellite has data
        has_data_node = FSCNode(
            # Choose action based on satellite position and transmission probability
            sat_idx == m.num_satellites || m.ground_tx_probs[sat_idx] > 0.7 ? 
                findfirst(a -> a == "transmit-1", sat_actions) :  # Transmit if last or high prob
                findfirst(a -> a == "pass-right-1", sat_actions),  # Otherwise pass right
            
            # Simple transitions - stay in same state if obs shows we have data
            Dict(obs => has_packets(obs, sat_idx) ? 1 : 2 for obs in obs_list)
        )
        
        # Create a node for when satellite doesn't have data
        no_data_node = FSCNode(
            findfirst(a -> a == "wait", sat_actions),  # Wait if no data
            
            # Transition to has-data node if observation shows we have data
            Dict(obs => has_packets(obs, sat_idx) ? 1 : 2 for obs in obs_list)
        )
        
        push!(nodes, has_data_node)
        push!(nodes, no_data_node)
        
        push!(controllers, AgentController(nodes))
    end
    
    return JointController(controllers)
end

# Helper function to check if satellite has packets in an observation
function has_packets(obs::String, sat_idx::Int)
    # Parse the observation string
    try
        counts = parse_state(obs)
        if sat_idx <= length(counts)
            return counts[sat_idx] > 0
        end
    catch
        # If parsing fails, assume no packets
    end
    return false
end

# Function to create random controllers
function create_random_controllers(m::MultiPacketSatelliteNetworkPOMDP)
    # Create a controller for each satellite
    controllers = AgentController[]
    
    for sat_idx in 1:m.num_satellites
        # Get available actions for this satellite
        sat_actions = agent_actions(m, sat_idx)
        obs_list = agent_observations(m)
        
        # Create random nodes
        num_nodes = rand(2:4)  # Random number of nodes
        nodes = FSCNode[]
        
        for _ in 1:num_nodes
            # Random action
            action_idx = rand(1:length(sat_actions))
            
            # Random transitions
            transitions = Dict{String, Int}()
            for obs in obs_list
                transitions[obs] = rand(1:num_nodes)
            end
            
            push!(nodes, FSCNode(action_idx, transitions))
        end
        
        push!(controllers, AgentController(nodes))
    end
    
    return JointController(controllers)
end

# Evaluate the controller using value iteration
function evaluate_controller(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP; 
                           max_iter=2000, tolerance=1e-6)
    # Get all possible states
    all_states = POMDPs.states(prob)
    
    # Map states to indices for easier array access
    state_map = Dict(s => i for (i, s) in enumerate(all_states))
    
    # Get the number of states and number of nodes for each agent
    num_states = length(all_states)
    num_satellites = length(joint_controller.controllers)
    nodes_per_sat = [length(c.nodes) for c in joint_controller.controllers]
    
    # Create value function matrix
    # Dimensions: one for each satellite's node, plus one for state
    dims = [nodes_per_sat..., num_states]
    V = zeros(dims...)
    
    # Discount factor
    gamma = prob.discount_factor
    
    # Value iteration to compute the value function
    for iter in 1:max_iter
        # Make a copy of V for updating
        V_new = copy(V)
        
        # For each possible joint node configuration and state
        for node_indices in Iterators.product([1:n for n in nodes_per_sat]...)
            for s in 1:num_states
                # Get the current state name
                state = all_states[s]
                
                # If terminal state, value is 0
                if state == "all-transmitted"
                    V_new[node_indices..., s] = 0.0
                    continue
                end
                
                # Get joint action from current nodes
                joint_action_indices = [joint_controller.controllers[i].nodes[node_indices[i]].action 
                                      for i in 1:num_satellites]
                
                # Convert action indices to strings
                joint_action = Tuple(agent_actions(prob, i)[joint_action_indices[i]] 
                                   for i in 1:num_satellites)
                
                # Get immediate reward
                immediate_reward = POMDPs.reward(prob, state, joint_action)
                
                # Expected future reward
                future_reward = 0.0
                
                # For each possible next state
                for next_s in 1:num_states
                    next_state = all_states[next_s]
                    
                    # Get transition distribution
                    trans_distribution = POMDPs.transition(prob, state, joint_action)
                    
                    # Calculate transition probability
                    trans_prob = 0.0
                    
                    if typeof(trans_distribution) <: Deterministic
                        # Deterministic transition
                        trans_prob = (trans_distribution.val == next_state) ? 1.0 : 0.0
                    elseif typeof(trans_distribution) <: SparseCat
                        # SparseCat distribution
                        for (idx, s_val) in enumerate(trans_distribution.vals)
                            if s_val == next_state
                                trans_prob = trans_distribution.probs[idx]
                                break
                            end
                        end
                    else
                        # Default case - try to use pdf
                        try
                            trans_prob = pdf(trans_distribution, next_state)
                        catch
                            @warn "Unsupported distribution type: $(typeof(trans_distribution))"
                            trans_prob = 0.0
                        end
                    end
                    
                    # Skip if transition probability is 0
                    if trans_prob ≈ 0.0
                        continue
                    end
                    
                    # Get observation distribution
                    obs_distribution = POMDPs.observation(prob, joint_action, next_state)
                    
                    # For each possible joint observation
                    if typeof(obs_distribution) <: Deterministic
                        # For deterministic observations, just use the single observation
                        next_obs = obs_distribution.val
                        
                        # Determine next nodes for each satellite
                        next_nodes = []
                        for i in 1:num_satellites
                            current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                            next_node = current_node.transitions[next_obs[i]]
                            push!(next_nodes, next_node)
                        end
                        
                        # Future value
                        future_value = V[next_nodes..., next_s]
                        
                        # Update expected future reward
                        future_reward += trans_prob * future_value
                    else
                        # For stochastic observations
                        for (obs_idx, joint_obs) in enumerate(obs_distribution.vals)
                            obs_prob = obs_distribution.probs[obs_idx]
                            
                            # Skip if observation probability is 0
                            if obs_prob ≈ 0.0
                                continue
                            end
                            
                            # Determine next nodes for each satellite
                            next_nodes = []
                            for i in 1:num_satellites
                                current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                                
                                # Handle potential missing observations in controller
                                # This can happen if the observation function generates observations
                                # that weren't included in the initial controller
                                if haskey(current_node.transitions, joint_obs[i])
                                    next_node = current_node.transitions[joint_obs[i]]
                                else
                                    # Default to staying in same node if observation not recognized
                                    next_node = node_indices[i]
                                end
                                
                                push!(next_nodes, next_node)
                            end
                            
                            # Future value
                            future_value = V[next_nodes..., next_s]
                            
                            # Update expected future reward
                            future_reward += trans_prob * obs_prob * future_value
                        end
                    end
                end
                
                # Total value for this configuration
                V_new[node_indices..., s] = immediate_reward + gamma * future_reward
            end
        end
        
        # Check convergence
        delta = maximum(abs.(V_new - V))
        V = copy(V_new)
        
        if delta < tolerance
            # println("Value iteration converged after $(iter) iterations")
            break
        end
        
        if iter == max_iter
            @warn "Value iteration did not converge after $(max_iter) iterations"
        end
    end
    
    # Compute the value for the initial state and initial controller nodes
    initial_state = initialstate(prob).val
    initial_state_idx = findfirst(s -> s == initial_state, all_states)
    
    if initial_state_idx === nothing
        @warn "Initial state not found in state space"
        return -Inf, Dict()
    end
    
    initial_nodes = [1 for _ in 1:num_satellites]  # Start at first node for each satellite
    
    value = V[initial_nodes..., initial_state_idx]
    
    # Collect some additional information for analysis
    info = Dict(
        "state_values" => V,
        "initial_state_value" => value,
        "controller_size" => sum(nodes_per_sat)
    )
    
    return value, info
end

# Verification function to simulate and analyze the controller
function verify_satellite_controller(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP, num_episodes=1000, max_steps=50)
    total_reward = 0.0
    rewards_per_episode = []
    packets_transmitted = 0
    episodes_completed = 0
    total_steps = 0
    
    # Track statistics for each satellite
    satellite_tx_attempts = zeros(Int, prob.num_satellites)
    satellite_tx_success = zeros(Int, prob.num_satellites)
    satellite_pass_attempts = zeros(Int, prob.num_satellites)
    satellite_pass_success = zeros(Int, prob.num_satellites)
    
    # For each episode
    for episode in 1:num_episodes
        # Initialize state: all packets at first satellite
        initial_distribution = zeros(Int, prob.num_satellites)
        initial_distribution[1] = prob.total_packets
        state = join(initial_distribution, "-")
        
        # Start with first node for each satellite
        nodes = ones(Int, prob.num_satellites)
        episode_reward = 0.0
        step_count = 0
        episode_packets_transmitted = 0
        
        # Run episode until max steps or all data transmitted
        while step_count < max_steps && state != "all-transmitted"
            step_count += 1
            
            # Get current packet distribution
            packet_counts = parse_state(state)
            
            # Get actions from current nodes
            action_indices = [joint_controller.controllers[i].nodes[nodes[i]].action 
                             for i in 1:prob.num_satellites]
            
            # Convert action indices to strings
            actions = []
            for i in 1:prob.num_satellites
                sat_actions = agent_actions(prob, i)
                
                # Handle out-of-bounds indices (could happen with random controllers)
                if action_indices[i] <= length(sat_actions)
                    push!(actions, sat_actions[action_indices[i]])
                else
                    push!(actions, "wait")  # Default to wait if invalid index
                end
            end
            joint_action = Tuple(actions)
            
            # Get reward
            step_reward = POMDPs.reward(prob, state, joint_action)
            episode_reward += step_reward
            
            # Process actions and generate next state
            new_packet_counts = copy(packet_counts)
            
            # Process transmissions
            for sat_idx in 1:prob.num_satellites
                action = actions[sat_idx]
                
                if startswith(action, "transmit-")
                    num_packets = parse(Int, split(action, "-")[2])
                    actual_packets = min(num_packets, packet_counts[sat_idx])
                    
                    if actual_packets > 0
                        satellite_tx_attempts[sat_idx] += actual_packets
                        
                        # Simulate each packet transmission
                        for _ in 1:actual_packets
                            if rand() < prob.ground_tx_probs[sat_idx]
                                # Successful transmission
                                satellite_tx_success[sat_idx] += 1
                                new_packet_counts[sat_idx] -= 1
                                episode_packets_transmitted += 1
                            end
                        end
                    end
                elseif startswith(action, "pass-")
                    parts = split(action, "-")
                    direction = parts[2]
                    num_packets = parse(Int, parts[3])
                    actual_packets = min(num_packets, packet_counts[sat_idx])
                    
                    if actual_packets > 0
                        satellite_pass_attempts[sat_idx] += actual_packets
                        
                        # Determine target satellite
                        target_idx = -1
                        if direction == "left" && sat_idx > 1
                            target_idx = sat_idx - 1
                        elseif direction == "right" && sat_idx < prob.num_satellites
                            target_idx = sat_idx + 1
                        else
                            continue  # Invalid direction
                        end
                        
                        # Simulate each packet pass
                        for _ in 1:actual_packets
                            if rand() < prob.pass_success_prob
                                # Successful pass
                                satellite_pass_success[sat_idx] += 1
                                new_packet_counts[sat_idx] -= 1
                                new_packet_counts[target_idx] += 1
                            end
                        end
                    end
                end
            end
            
            # Check if all packets have been transmitted
            if sum(new_packet_counts) == 0
                state = "all-transmitted"
            else
                state = join(new_packet_counts, "-")
            end
            
            # Generate observations for each satellite
            observations = []
            
            if state == "all-transmitted"
                # All satellites observe empty network
                empty_obs = join(zeros(Int, prob.num_satellites), "-")
                observations = [empty_obs for _ in 1:prob.num_satellites]
            else
                # Each satellite observes with some noise
                for i in 1:prob.num_satellites
                    if rand() < prob.observation_accuracy
                        # Accurate observation
                        push!(observations, state)
                    else
                        # Noisy observation
                        noisy_counts = copy(new_packet_counts)
                        
                        # Add noise to counts
                        for j in 1:length(noisy_counts)
                            noise = rand(-1:1)
                            noisy_counts[j] = max(0, noisy_counts[j] + noise)
                        end
                        
                        push!(observations, join(noisy_counts, "-"))
                    end
                end
            end
            
            # Update node for each satellite
            for i in 1:prob.num_satellites
                current_node = joint_controller.controllers[i].nodes[nodes[i]]
                
                # Handle potentially missing observations in transitions
                if haskey(current_node.transitions, observations[i])
                    nodes[i] = current_node.transitions[observations[i]]
                else
                    # Default to staying in current node if observation not recognized
                    nodes[i] = nodes[i]
                end
            end
        end
        
        # Record episode statistics
        push!(rewards_per_episode, episode_reward)
        total_reward += episode_reward
        total_steps += step_count
        packets_transmitted += episode_packets_transmitted
        
        if state == "all-transmitted"
            episodes_completed += 1
        end
    end
    
    # Calculate overall statistics
    avg_reward = total_reward / num_episodes
    std_dev = std(rewards_per_episode)
    completion_rate = episodes_completed / num_episodes
    avg_steps = total_steps / num_episodes
    transmission_efficiency = packets_transmitted / (prob.total_packets * num_episodes)
    
    println("=== Multi-Packet Satellite Network Controller Verification Results ===")
    println("Average reward per episode: $avg_reward")
    println("Standard deviation: $std_dev")
    println("Episode completion rate: $(completion_rate * 100)%")
    println("Average steps per episode: $avg_steps")
    println("Transmission efficiency: $(transmission_efficiency * 100)%")
    
    println("\n=== Satellite Transmission Statistics ===")
    for i in 1:prob.num_satellites
        tx_success_rate = satellite_tx_attempts[i] > 0 ? 
            satellite_tx_success[i] / satellite_tx_attempts[i] : 0.0
        pass_success_rate = satellite_pass_attempts[i] > 0 ? 
            satellite_pass_success[i] / satellite_pass_attempts[i] : 0.0
        
        println("Satellite $i (Ground TX prob: $(prob.ground_tx_probs[i])):")
        println("  Transmission attempts: $(satellite_tx_attempts[i])")
        println("  Transmission successes: $(satellite_tx_success[i]) ($(tx_success_rate*100)%)")
        println("  Pass attempts: $(satellite_pass_attempts[i])")
        println("  Pass successes: $(satellite_pass_success[i]) ($(pass_success_rate*100)%)")
    end
    
    println("\n=== Controller Structure Analysis ===")
    for i in 1:prob.num_satellites
        println("Satellite $i controller has $(length(joint_controller.controllers[i].nodes)) nodes")
        
        # Analyze node actions
        for (j, node) in enumerate(joint_controller.controllers[i].nodes)
            sat_actions = agent_actions(prob, i)
            
            # Handle out-of-bounds index
            action_name = node.action <= length(sat_actions) ?
                sat_actions[node.action] : "invalid-action"
                
            println("  Node $j: Action = $action_name")
            println("    Transitions: $(length(node.transitions)) observation mappings")
        end
    end
    
    return avg_reward, completion_rate
end

# For running policy iteration with multi-packet satellite network
function dec_pomdp_pi(controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP)
    # Initialize
    it = 0
    epsilon = 0.01  # Desired precision
    R_max = 100.0  # Approximate maximum absolute reward (should be calculated properly)
    gamma = prob.discount_factor
    
    ctrlr_t = deepcopy(controller)
    n = length(ctrlr_t.controllers)
    
    # Initial evaluation
    V_prev, _ = evaluate_controller(ctrlr_t, prob)
    println("Initial controller value: $(V_prev)")
    
    # Set initial values
    V_curr = V_prev
    improvement = Inf  # Start with infinite improvement to ensure first iteration
    
    # Main policy iteration loop with proper stopping condition
    while it < 30 && improvement > epsilon
        # [Backup and evaluate]
        for i in 1:n
            println("Backing up agent $i...")
            new_controller = improved_exhaustive_backup(
                ctrlr_t.controllers[i],
                ctrlr_t,
                i,
                prob
            )
            
            ctrlr_t.controllers[i] = new_controller
        end
        
        # Evaluate the joint controller
        V_curr, _ = evaluate_controller(ctrlr_t, prob)
        println("After backup, controller value: $(V_curr)")
        
        # Calculate improvement (absolute difference)
        improvement = abs(V_curr - V_prev)
        println("Improvement: $(improvement)")
        
        # Update previous value for next iteration
        V_prev = V_curr
        
        it += 1
        println("Completed iteration $(it)")
        
        # Additional stopping condition based on convergence formula
        if (gamma^it * R_max) < epsilon
            println("Theoretical bound reached, algorithm converged.")
            break
        end
    end
    
    # Report reason for stopping
    if it >= 30
        println("Stopped due to maximum iterations reached.")
    elseif improvement <= epsilon
        println("Stopped due to convergence (improvement below threshold).")
    end
    
    return it, ctrlr_t
end

# Improved exhaustive backup function for multi-packet satellite network
function improved_exhaustive_backup(controller::AgentController, joint_controller::JointController, agent_idx::Int, prob::MultiPacketSatelliteNetworkPOMDP)
    original_controller = deepcopy(controller)
    
    # Extract individual agent actions
    agent_act = agent_actions(prob, agent_idx)
    
    # Get individual agent observations
    obs_list = agent_observations(prob)
    
    current_nodes = length(controller.nodes)
    candidate_nodes = Vector{FSCNode}()
    
    # Generate candidate nodes (simplified to reduce computation)
    for action_idx in 1:length(agent_act)
        # We'll create candidates with different transition patterns
        # but limit the total number to prevent combinatorial explosion
        
        # Simple pattern: go to node 1 for all observations
        transitions1 = Dict{String, Int}()
        for obs in obs_list
            transitions1[obs] = 1
        end
        push!(candidate_nodes, FSCNode(action_idx, transitions1))
        
        # Pattern: separate nodes for data vs no data observations
        transitions2 = Dict{String, Int}()
        for obs in obs_list
            # Go to node 1 if observation suggests we have data, else node 2
            transitions2[obs] = has_packets(obs, agent_idx) ? 1 : 2
        end
        
        if current_nodes >= 2
            push!(candidate_nodes, FSCNode(action_idx, transitions2))
        end
        
        # Add a few random transition patterns to increase diversity
        for _ in 1:3
            transitions_rand = Dict{String, Int}()
            for obs in obs_list
                transitions_rand[obs] = rand(1:max(2, current_nodes))
            end
            push!(candidate_nodes, FSCNode(action_idx, transitions_rand))
        end
    end
    
    # Evaluate the current controller value
    current_value, _ = evaluate_controller(joint_controller, prob)
    println("Current value before backup: $current_value")
    
    # Try each candidate node as a replacement for each existing node
    best_controller = deepcopy(controller)
    best_value = current_value
    improved = false
    
    # For each existing node in the controller
    for node_idx in 1:length(controller.nodes)
        # Try replacing with each candidate node
        for candidate in candidate_nodes
            # Create a temporary controller with this node replaced
            temp_controller = deepcopy(controller)
            
            # Replace the node
            new_nodes = Vector{FSCNode}()
            for i in 1:length(temp_controller.nodes)
                if i == node_idx
                    push!(new_nodes, candidate)
                else
                    push!(new_nodes, temp_controller.nodes[i])
                end
            end
            temp_controller = AgentController(new_nodes)
            
            # Create a temporary joint controller for evaluation
            temp_joint_controller = deepcopy(joint_controller)
            temp_joint_controller.controllers[agent_idx] = temp_controller
            
            # Evaluate this controller
            temp_value, _ = evaluate_controller(temp_joint_controller, prob)
            
            # If it's better, keep it
            if temp_value > best_value
                best_value = temp_value
                best_controller = deepcopy(temp_controller)
                improved = true
                println("Found improvement by replacing node $node_idx, new value: $temp_value")
            end
        end
    end
    
    # Try adding a new node (only if it helps and controller isn't too large)
    if !improved && length(controller.nodes) < 5  # Limit size to prevent explosion
        for candidate in candidate_nodes
            # Create a temporary controller with this node added
            temp_controller = deepcopy(controller)
            new_nodes = copy(temp_controller.nodes)
            push!(new_nodes, candidate)
            temp_controller = AgentController(new_nodes)
            
            # Create a temporary joint controller
            temp_joint_controller = deepcopy(joint_controller)
            temp_joint_controller.controllers[agent_idx] = temp_controller
            
            # Evaluate this controller
            temp_value, _ = evaluate_controller(temp_joint_controller, prob)
            
            # If it's better, keep it
            if temp_value > best_value
                best_value = temp_value
                best_controller = deepcopy(temp_controller)
                improved = true
                println("Found improvement by adding a new node, new value: $temp_value")
                # Break early when we find an improvement to save computation
                break
            end
        end
    end
    
    # Return the best controller found
    if improved
        println("Controller improved from $current_value to $best_value")
        return best_controller
    else
        println("No improvement found, returning original controller")
        return original_controller
    end
end

# Create a satellite network with 4 satellites and 5 data packets
sat_network = MultiPacketSatelliteNetworkPOMDP(
    num_satellites = 3,
    total_packets = 2,
    max_capacity = 3,
    ground_tx_probs = [0.3, 0.5, 0.6],
    discount_factor = 0.9,
    pass_success_prob = 0.95,
    observation_accuracy = 0.8,
    successful_tx_reward = 10.0,
    unsuccessful_tx_penalty = -2.0,
    pass_cost = -1.0,
    congestion_penalty = -5.0
)

# Create initial controller
initial_ctrl = create_initial_controllers(sat_network)

# Run policy iteration
iterations, final_controller = dec_pomdp_pi(initial_ctrl, sat_network)

# Verify the controller
avg_reward, completion_rate = verify_satellite_controller(final_controller, sat_network)

println("Policy iteration completed in $iterations iterations.")
println("Final controller average reward: $avg_reward")
println("Completion rate: $(completion_rate * 100)%")