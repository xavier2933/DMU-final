using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat

"""
    SatelliteNetworkPOMDP

A decentralized POMDP modeling a network of satellites that need to coordinate
to transmit data to the ground. Each satellite can either pass data to its neighbors
or attempt to transmit to ground with varying success probabilities.
"""
struct SatelliteNetworkPOMDP <: POMDP{String, Tuple{Vararg{String}}, Tuple{Vararg{String}}}
    num_satellites::Int                # Number of satellites in the network
    discount_factor::Float64           # Discount factor for future rewards
    ground_tx_probs::Vector{Float64}   # Probability of successful ground transmission for each satellite
    pass_success_prob::Float64         # Probability of successful data passing between satellites
    successful_tx_reward::Float64      # Reward for successful ground transmission
    unsuccessful_tx_penalty::Float64   # Penalty for unsuccessful ground transmission
    pass_cost::Float64                 # Cost for passing data between satellites
end

# Default constructor
function SatelliteNetworkPOMDP(;
    num_satellites = 3,
    discount_factor = 0.9,
    ground_tx_probs = [0.3, 0.5, 0.7],  # Example probabilities
    pass_success_prob = 0.95,
    successful_tx_reward = 10.0,
    unsuccessful_tx_penalty = -2.0,
    pass_cost = -1.0
)
    return SatelliteNetworkPOMDP(
        num_satellites,
        discount_factor,
        ground_tx_probs,
        pass_success_prob,
        successful_tx_reward,
        unsuccessful_tx_penalty,
        pass_cost
    )
end

# Define state space - satellite holding the data
function POMDPs.states(m::SatelliteNetworkPOMDP)
    states = []
    for i in 1:m.num_satellites
        push!(states, "data-at-$i")
    end
    # Add terminal state for when data is successfully transmitted
    push!(states, "data-transmitted")
    return states
end

# Define agent actions
function POMDPs.actions(m::SatelliteNetworkPOMDP)
    # Individual agent actions depend on satellite position
    joint_actions = Vector{NTuple{m.num_satellites, String}}()
    
    # Generate all combinations of actions for all satellites
    function generate_actions(current_actions, sat_idx)
        if sat_idx > m.num_satellites
            push!(joint_actions, Tuple(current_actions))
            return
        end
        
        # Available actions depend on satellite position
        available_actions = ["wait"]  # All satellites can wait
        
        if sat_idx == 1
            # First satellite can only pass right or transmit
            push!(available_actions, "transmit")
            push!(available_actions, "pass-right")
        elseif sat_idx == m.num_satellites
            # Last satellite can only pass left or transmit
            push!(available_actions, "transmit")
            push!(available_actions, "pass-left")
        else
            # Middle satellites can pass left, pass right, or transmit
            push!(available_actions, "transmit")
            push!(available_actions, "pass-left")
            push!(available_actions, "pass-right")
        end
        
        for action in available_actions
            current_actions[sat_idx] = action
            generate_actions(current_actions, sat_idx + 1)
        end
    end
    
    generate_actions(["wait" for _ in 1:m.num_satellites], 1)
    return joint_actions
end

# Define observation space
function POMDPs.observations(m::SatelliteNetworkPOMDP)
    # Observations: each satellite can observe if it has data or not
    obs_options = ["has-data", "no-data"]
    
    # Generate all combinations of observations
    joint_obs = Vector{NTuple{m.num_satellites, String}}()
    
    function generate_observations(current_obs, sat_idx)
        if sat_idx > m.num_satellites
            push!(joint_obs, Tuple(current_obs))
            return
        end
        
        for obs in obs_options
            current_obs[sat_idx] = obs
            generate_observations(current_obs, sat_idx + 1)
        end
    end
    
    generate_observations(["no-data" for _ in 1:m.num_satellites], 1)
    return joint_obs
end

# Define initial state distribution
function POMDPs.initialstate(m::SatelliteNetworkPOMDP)
    # Data always starts at satellite 1
    return Deterministic("data-at-1")
end

# Define discount factor
function POMDPs.discount(m::SatelliteNetworkPOMDP)
    return m.discount_factor
end

# Define transition function
function POMDPs.transition(m::SatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})
    # If data already transmitted, stay in terminal state
    if s == "data-transmitted"
        return Deterministic("data-transmitted")
    end
    
    # Extract the satellite number from the state
    if !startswith(s, "data-at-")
        # Invalid state, shouldn't happen
        return Deterministic(s)
    end
    
    data_sat = parse(Int, last(split(s, "-")))
    
    # Actions by the satellite holding the data
    sat_action = a[data_sat]
    
    if sat_action == "wait"
        # Data stays at the same satellite
        return Deterministic(s)
    elseif sat_action == "transmit"
        # Attempt to transmit to ground
        tx_prob = m.ground_tx_probs[data_sat]
        
        # Either successfully transmit or data stays at same satellite
        return SparseCat(["data-transmitted", s], [tx_prob, 1 - tx_prob])
    elseif sat_action == "pass-left" && data_sat > 1
        # Pass data to left neighbor
        new_sat = data_sat - 1
        
        # Either successfully pass or data stays at same satellite
        return SparseCat(["data-at-$new_sat", s], [m.pass_success_prob, 1 - m.pass_success_prob])
    elseif sat_action == "pass-right" && data_sat < m.num_satellites
        # Pass data to right neighbor
        new_sat = data_sat + 1
        
        # Either successfully pass or data stays at same satellite
        return SparseCat(["data-at-$new_sat", s], [m.pass_success_prob, 1 - m.pass_success_prob])
    else
        # Invalid action for this satellite or position
        return Deterministic(s)
    end
end

# Define observation function
function POMDPs.observation(m::SatelliteNetworkPOMDP, a::Tuple{Vararg{String}}, sp::String)
    # Create the observation vector based on the next state
    observations = ["no-data" for _ in 1:m.num_satellites]
    
    if sp == "data-transmitted"
        # If data is transmitted, no satellite has data
        return Deterministic(Tuple(observations))
    else
        # Extract the satellite number from the state
        data_sat = parse(Int, last(split(sp, "-")))
        
        # The satellite with data observes that it has data
        observations[data_sat] = "has-data"
        
        # Deterministic observation in this model
        return Deterministic(Tuple(observations))
    end
end

# Define reward function
function POMDPs.reward(m::SatelliteNetworkPOMDP, s::String, a::Tuple{Vararg{String}})
    # If already in terminal state, no reward
    if s == "data-transmitted"
        return 0.0
    end
    
    # Extract the satellite number from the state
    data_sat = parse(Int, last(split(s, "-")))
    
    # Action by the satellite holding the data
    sat_action = a[data_sat]
    
    if sat_action == "wait"
        return 0.0  # No reward for waiting
    elseif sat_action == "transmit"
        # Expected reward for transmission attempt
        tx_prob = m.ground_tx_probs[data_sat]
        return tx_prob * m.successful_tx_reward + (1 - tx_prob) * m.unsuccessful_tx_penalty
    elseif (sat_action == "pass-left" && data_sat > 1) || 
           (sat_action == "pass-right" && data_sat < m.num_satellites)
        return m.pass_cost  # Cost for passing data
    else
        return 0.0  # Invalid action, no reward
    end
end

# Helper functions for working with individual agents

# Get individual agent actions based on position
function agent_actions(m::SatelliteNetworkPOMDP, sat_idx::Int)
    actions = ["wait"]  # All satellites can wait
    
    if sat_idx == 1
        # First satellite can only pass right or transmit
        push!(actions, "transmit")
        push!(actions, "pass-right")
    elseif sat_idx == m.num_satellites
        # Last satellite can only pass left or transmit
        push!(actions, "transmit")
        push!(actions, "pass-left")
    else
        # Middle satellites can pass left, pass right, or transmit
        push!(actions, "transmit")
        push!(actions, "pass-left")
        push!(actions, "pass-right")
    end
    
    return actions
end

# General agent actions for controller generation
function agent_actions(m::SatelliteNetworkPOMDP)
    # Return the most general set of actions any satellite might take
    return ["wait", "transmit", "pass-left", "pass-right"]
end

# Get individual agent observations
function agent_observations(m::SatelliteNetworkPOMDP)
    return ["has-data", "no-data"]
end

# Example controller creation function
function create_initial_controllers(m::SatelliteNetworkPOMDP)
    # Create a controller for each satellite
    controllers = AgentController[]
    
    for sat_idx in 1:m.num_satellites
        # Get available actions for this satellite
        sat_actions = agent_actions(m, sat_idx)
        obs_list = agent_observations(m)
        
        # Create nodes - one for each observation
        has_data_node = FSCNode(
            # When has data: transmit for high-prob satellites, pass for low-prob satellites
            findfirst(a -> a == (m.ground_tx_probs[sat_idx] > 0.5 ? "transmit" : 
                      (sat_idx < m.num_satellites ? "pass-right" : "pass-left")), sat_actions),
            Dict("has-data" => 1, "no-data" => 2)  # Stay in same node if has data
        )
        
        no_data_node = FSCNode(
            findfirst(a -> a == "wait", sat_actions),  # Wait if no data
            Dict("has-data" => 1, "no-data" => 2)  # Switch to has-data node if gets data
        )
        
        push!(controllers, AgentController([has_data_node, no_data_node]))
    end
    
    return JointController(controllers)
end

# Evaluation function (placeholder for your implementation)
function evaluate_controller(joint_controller::JointController, prob::SatelliteNetworkPOMDP; 
                           max_iter=1000, tolerance=1e-6)
    # Get all possible states
    states = POMDPs.states(prob)
    
    # Get observations for satellites
    observations = agent_observations(prob)
    
    # Map states to indices for easier array access
    state_map = Dict(s => i for (i, s) in enumerate(states))
    
    # Get the number of states and number of nodes for each agent
    num_states = length(states)
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
                state = states[s]
                
                # If terminal state, value is 0
                if state == "data-transmitted"
                    V_new[node_indices..., s] = 0.0
                    continue
                end
                
                # Extract the satellite number from the state (which satellite has data)
                data_sat = parse(Int, last(split(state, "-")))
                
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
                    next_state = states[next_s]
                    
                    # Get transition distribution
                    trans_distribution = POMDPs.transition(prob, state, joint_action)
                    
                    # Calculate transition probability based on distribution type
                    trans_prob = 0.0
                    
                    if typeof(trans_distribution) <: POMDPTools.Deterministic
                        # Deterministic transition
                        trans_prob = (trans_distribution.val == next_state) ? 1.0 : 0.0
                    elseif typeof(trans_distribution) <: POMDPTools.SparseCat
                        # SparseCat distribution
                        for (idx, s_val) in enumerate(trans_distribution.vals)
                            if s_val == next_state
                                trans_prob = trans_distribution.probs[idx]
                                break
                            end
                        end
                    else
                        # Default case - try to use pdf if available
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
                    
                    # Generate the observation each satellite would see
                    next_joint_obs = ["no-data" for _ in 1:num_satellites]
                    if next_state != "data-transmitted"
                        next_data_sat = parse(Int, last(split(next_state, "-")))
                        next_joint_obs[next_data_sat] = "has-data"
                    end
                    next_joint_obs_tuple = Tuple(next_joint_obs)
                    
                    # Get observation probability
                    obs_distribution = POMDPs.observation(prob, joint_action, next_state)
                    obs_prob = 0.0
                    
                    if typeof(obs_distribution) <: POMDPTools.Deterministic
                        obs_prob = (obs_distribution.val == next_joint_obs_tuple) ? 1.0 : 0.0
                    elseif typeof(obs_distribution) <: POMDPTools.SparseCat
                        for (idx, o_val) in enumerate(obs_distribution.vals)
                            if o_val == next_joint_obs_tuple
                                obs_prob = obs_distribution.probs[idx]
                                break
                            end
                        end
                    else
                        try
                            obs_prob = pdf(obs_distribution, next_joint_obs_tuple)
                        catch
                            obs_prob = 0.0
                        end
                    end
                    
                    # Skip if observation probability is 0
                    if obs_prob ≈ 0.0
                        continue
                    end
                    
                    # Determine next nodes for each satellite
                    next_nodes = []
                    for i in 1:num_satellites
                        current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                        next_node = current_node.transitions[next_joint_obs[i]]
                        push!(next_nodes, next_node)
                    end
                    
                    # Future value
                    future_value = V[next_nodes..., next_s]
                    
                    # Update expected future reward
                    future_reward += trans_prob * obs_prob * future_value
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
    initial_state_idx = findfirst(s -> s == "data-at-1", states)
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