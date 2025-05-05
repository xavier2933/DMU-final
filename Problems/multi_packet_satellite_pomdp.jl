using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat
using Statistics
using Random

include("../Solvers/dec_pi_packet_solver.jl")


"""
    MultiPacketSatelliteNetworkPOMDP

A decentralized POMDP modeling a network of satellites that need to coordinate
to transmit multiple data packets to the ground. Each satellite can pass data to neighbors
or attempt to transmit to ground with varying success probabilities.
"""

# structs for controllers
struct FSCNode
    action::Int
    transitions::Dict{String, Int}
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
        
        for i in 0:min(remaining, max_per_sat)
            current_partition[current_idx] = i
            generate_partitions(remaining - i, current_idx + 1, current_partition)
        end
    end
    
    generate_partitions(total_packets, 1, zeros(Int, num_satellites))
    return partitions
end

function parse_state(s::String)
    if s == "all-transmitted"
        return Int[]
    else
        return [parse(Int, x) for x in split(s, "-")]
    end
end

function binomial_probability(n, k, p)
    return binomial(n, k) * p^k * (1-p)^(n-k)
end

# Claude for elegant scalable solution
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

function POMDPs.actions(m::MultiPacketSatelliteNetworkPOMDP)
    joint_actions = Vector{NTuple{m.num_satellites, String}}()
    
    sat_actions_list = [agent_actions(m, sat_idx) for sat_idx in 1:m.num_satellites]
    
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

# Claude
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

function POMDPs.initialstate(prob::POMDP)

    initial_distribution = zeros(Int, prob.num_satellites)
    initial_distribution[1] = prob.total_packets

    return Deterministic(join(initial_distribution, "-"))
end

function POMDPs.discount(m::MultiPacketSatelliteNetworkPOMDP)
    return m.discount_factor
end

function POMDPs.transition(m::MultiPacketSatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})
    if s == "all-transmitted"
        return Deterministic("all-transmitted")
    end
    
    packet_counts = parse_state(s)
    
    new_packet_counts = copy(packet_counts)
    next_states = String[]
    next_probs = Float64[]
    
    # Process ground transmissions first
    for sat_idx in 1:m.num_satellites
        action = a[sat_idx]
        
        if startswith(action, "transmit-")
            # Extract number of packets to transmit - old, used to be able to send
            # multiple at each time step, didn't want to re-factor
            num_packets = parse(Int, split(action, "-")[2])
            
            # Can't transmit more than you have
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                # Create all possible transmission outcomes
                for success_count in 0:actual_packets
                    # Calculate probability of this outcome - Claude
                    prob = binomial_probability(actual_packets, success_count, m.ground_tx_probs[sat_idx])
                    
                    temp_counts = copy(new_packet_counts)
                    temp_counts[sat_idx] -= success_count
                    
                    if sum(temp_counts) == 0
                        next_state = "all-transmitted"
                    else
                        next_state = join(temp_counts, "-")
                    end
                    
                    push!(next_states, next_state)
                    push!(next_probs, prob)
                end
                new_packet_counts[sat_idx] -= actual_packets

            end
        end
    end
    
    if isempty(next_states)
        temp_packet_counts = copy(packet_counts)
        changes_made = false
        
        for sat_idx in 1:m.num_satellites
            action = a[sat_idx]
            
            if startswith(action, "pass-")
                parts = split(action, "-")
                direction = parts[2]
                num_packets = parse(Int, parts[3])
                
                actual_packets = min(num_packets, temp_packet_counts[sat_idx])
                
                if actual_packets > 0
                    changes_made = true
                    
                    target_idx = -1
                    if direction == "left" && sat_idx > 1
                        target_idx = sat_idx - 1
                    elseif direction == "right" && sat_idx < m.num_satellites
                        target_idx = sat_idx + 1
                    else
                        continue
                    end
                    
                    pass_prob = m.pass_success_prob
                    
                    for success_count in 0:actual_packets
                        # Calculate probability of this outcome - Claude
                        prob = binomial_probability(actual_packets, success_count, pass_prob)
                        
                        temp_counts = copy(temp_packet_counts)
                        temp_counts[sat_idx] -= success_count
                        temp_counts[target_idx] += success_count
                        
                        next_state = join(temp_counts, "-")
                        push!(next_states, next_state)
                        push!(next_probs, prob)
                    end
                    
                    temp_packet_counts[sat_idx] -= actual_packets
                    temp_packet_counts[target_idx] += actual_packets
                end
            end
        end
        
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

function POMDPs.observation(m::MultiPacketSatelliteNetworkPOMDP, a::Tuple{Vararg{String}}, sp::String)
    # Claude helped with this

    if sp == "all-transmitted"
        empty_obs = join(zeros(Int, m.num_satellites), "-")
        return Deterministic(Tuple(fill(empty_obs, m.num_satellites)))
    end
    
    true_packet_counts = parse_state(sp)
    
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

function POMDPs.reward(m::MultiPacketSatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})

    if s == "all-transmitted"
        return 0.0
    end
    
    packet_counts = parse_state(s)
    
    total_reward = 0.0
    
    for sat_idx in 1:m.num_satellites
        action = a[sat_idx]
        
        if startswith(action, "transmit-")
            num_packets = parse(Int, split(action, "-")[2])
            
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                tx_prob = m.ground_tx_probs[sat_idx]
                expected_success = actual_packets * tx_prob
                expected_failure = actual_packets * (1 - tx_prob)
                
                total_reward += expected_success * m.successful_tx_reward
                total_reward += expected_failure * m.unsuccessful_tx_penalty
            end
        elseif startswith(action, "pass-")
            parts = split(action, "-")
            num_packets = parse(Int, parts[3])
            
            actual_packets = min(num_packets, packet_counts[sat_idx])
            
            if actual_packets > 0
                total_reward += actual_packets * m.pass_cost
            end
        end
    end
    
    for count in packet_counts
        if count > m.max_capacity
            total_reward += (count - m.max_capacity) * m.congestion_penalty
        end
    end
    
    return total_reward
end

function agent_actions(m::MultiPacketSatelliteNetworkPOMDP, sat_idx::Int)
    actions = ["wait"]  
    
    push!(actions, "transmit-1")
    
    if sat_idx > 1 
        push!(actions, "pass-left-1")
    end
    
    if sat_idx < m.num_satellites
        push!(actions, "pass-right-1")
    end
    
    return actions
end


function agent_observations(m::MultiPacketSatelliteNetworkPOMDP)
    # For simplicity, return a subset of possible observations    
    obs = []
    
    push!(obs, join(zeros(Int, m.num_satellites), "-"))
    
    for p in partition_packets(m.num_satellites, m.total_packets, m.max_capacity)
        push!(obs, join(p, "-"))
    end
    
    return obs
end

function create_initial_controllers(m::MultiPacketSatelliteNetworkPOMDP)
    controllers = AgentController[]
    
    for sat_idx in 1:m.num_satellites
        sat_actions = agent_actions(m, sat_idx)
        obs_list = agent_observations(m)
        
        nodes = FSCNode[]
        
        # Create a node for when satellite has data
        has_data_node = FSCNode(
            sat_idx == m.num_satellites || m.ground_tx_probs[sat_idx] > 0.7 ? 
                findfirst(a -> a == "transmit-1", sat_actions) :
                findfirst(a -> a == "pass-right-1", sat_actions),
            Dict(obs => has_packets(obs, sat_idx) ? 1 : 2 for obs in obs_list)
        )
        
        no_data_node = FSCNode(
            findfirst(a -> a == "wait", sat_actions),
            Dict(obs => has_packets(obs, sat_idx) ? 1 : 2 for obs in obs_list)
        )
        
        push!(nodes, has_data_node)
        push!(nodes, no_data_node)
        
        push!(controllers, AgentController(nodes))
    end
    
    return JointController(controllers)
end

# claude
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

# claude
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

# claude
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
