using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat

"""
    MultiAgentRescuePOMDP

A scalable decentralized POMDP where agents must coordinate to rescue survivors
in a disaster scenario. Each agent can either search for survivors or extract
survivors from a location. Successful extraction requires coordination.
"""
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

struct MultiAgentRescuePOMDP <: POMDP{String, Tuple{Vararg{String}}, Tuple{Vararg{String}}}
    num_agents::Int                  # Number of agents in the problem
    num_locations::Int               # Number of locations in the environment
    discount_factor::Float64         # Discount factor for future rewards
    search_cost::Float64             # Cost for searching
    failed_extraction_cost::Float64  # Cost for failed extraction
    successful_extraction_reward::Float64  # Reward for successful extraction
    p_correct_obs::Float64           # Probability of correct observation
end

# Default constructor
function MultiAgentRescuePOMDP(;
    num_agents = 2,
    num_locations = 2,
    discount_factor = 0.95,
    search_cost = -1.0,
    failed_extraction_cost = -2.0,
    successful_extraction_reward = 10.0,
    p_correct_obs = 0.8
)
    return MultiAgentRescuePOMDP(
        num_agents,
        num_locations,
        discount_factor,
        search_cost,
        failed_extraction_cost,
        successful_extraction_reward,
        p_correct_obs
    )
end

# Define state space - survivor locations
function POMDPs.states(m::MultiAgentRescuePOMDP)
    locations = []
    for i in 1:m.num_locations
        push!(locations, "survivor-loc-$i")
    end
    return locations
end

# Define agent actions
function POMDPs.actions(m::MultiAgentRescuePOMDP)
    # Individual agent actions: search, extract-loc-1, extract-loc-2, etc.
    agent_actions = ["search"]
    for i in 1:m.num_locations
        push!(agent_actions, "extract-loc-$i")
    end
    
    # Generate all combinations of actions for all agents
    joint_actions = Vector{NTuple{m.num_agents, String}}()
    
    current = fill(agent_actions[1], m.num_agents)
    
    function generate_actions(depth)
        if depth > m.num_agents
            push!(joint_actions, Tuple(current))
            return
        end
        
        for action in agent_actions
            current[depth] = action
            generate_actions(depth + 1)
        end
    end
    
    generate_actions(1)
    return joint_actions
end

# Define observation space
function POMDPs.observations(m::MultiAgentRescuePOMDP)
    # Individual agent observations: no-signal, signal-loc-1, signal-loc-2, etc.
    agent_obs = ["no-signal"]
    for i in 1:m.num_locations
        push!(agent_obs, "signal-loc-$i")
    end
    
    # Generate all combinations of observations for all agents
    joint_obs = Vector{NTuple{m.num_agents, String}}()
    
    current = fill(agent_obs[1], m.num_agents)
    
    function generate_observations(depth)
        if depth > m.num_agents
            push!(joint_obs, Tuple(current))
            return
        end
        
        for obs in agent_obs
            current[depth] = obs
            generate_observations(depth + 1)
        end
    end
    
    generate_observations(1)
    return joint_obs
end

# Define initial state distribution (uniform)
function POMDPs.initialstate(m::MultiAgentRescuePOMDP)
    return Uniform(states(m))
end

# Define discount factor
function POMDPs.discount(m::MultiAgentRescuePOMDP)
    return m.discount_factor
end

# Define transition function
function POMDPs.transition(m::MultiAgentRescuePOMDP, s::String, a::Tuple{Vararg{String}})
    # Extract the location number from the state
    survivor_loc = parse(Int, last(split(s, "-")))
    
    # Count agents performing extraction at each location
    extraction_counts = zeros(Int, m.num_locations)
    for action in a
        if startswith(action, "extract-loc-")
            loc = parse(Int, last(split(action, "-")))
            extraction_counts[loc] += 1
        end
    end
    
    # If at least 2 agents attempt extraction at the survivor location,
    # the survivor is rescued and moves to a new random location
    if extraction_counts[survivor_loc] >= 2
        return Uniform(states(m))
    else
        # Otherwise, survivor stays in place
        return Deterministic(s)
    end
end

# Define observation function
function POMDPs.observation(m::MultiAgentRescuePOMDP, a::Tuple{Vararg{String}}, sp::String)
    # Extract the location number from the next state
    survivor_loc = parse(Int, last(split(sp, "-")))
    
    # Get all possible joint observations
    all_observations = observations(m)
    
    # Calculate probabilities for each joint observation
    probabilities = Float64[]
    
    for joint_obs in all_observations
        prob = 1.0
        
        for (agent_idx, obs) in enumerate(joint_obs)
            action = a[agent_idx]
            
            if action == "search"
                # When searching, agents get informative observations
                if obs == "no-signal"
                    # Probability of no signal when survivor is elsewhere
                    is_correct = true
                    for i in 1:m.num_locations
                        if i != survivor_loc && obs == "signal-loc-$i"
                            is_correct = false
                            break
                        end
                    end
                    
                    if is_correct
                        prob *= 1 - m.p_correct_obs
                    else
                        prob *= (1 - m.p_correct_obs) / (m.num_locations - 1)
                    end
                elseif obs == "signal-loc-$survivor_loc"
                    # Correct observation
                    prob *= m.p_correct_obs
                else
                    # Incorrect signal
                    prob *= (1 - m.p_correct_obs) / (m.num_locations)
                end
            else
                # When extracting, observations are uninformative
                if obs == "no-signal"
                    prob *= 1 / (m.num_locations + 1)
                else
                    prob *= 1 / (m.num_locations + 1)
                end
            end
        end
        
        push!(probabilities, prob)
    end
    
    # Normalize probabilities to ensure they sum to 1
    probabilities = probabilities ./ sum(probabilities)
    
    return SparseCat(all_observations, probabilities)
end

# Define reward function
function POMDPs.reward(m::MultiAgentRescuePOMDP, s::String, a::Tuple{Vararg{String}})
    # Extract the location number from the state
    survivor_loc = parse(Int, last(split(s, "-")))
    
    # Count search actions
    num_search = 0
    
    # Count extraction actions at each location
    extraction_counts = zeros(Int, m.num_locations)
    
    for action in a
        if action == "search"
            num_search += 1
        elseif startswith(action, "extract-loc-")
            loc = parse(Int, last(split(action, "-")))
            extraction_counts[loc] += 1
        end
    end
    
    # Calculate reward
    reward = 0.0
    
    # Cost for searching
    reward += num_search * m.search_cost
    
    # Check if extraction was successful at survivor location
    if extraction_counts[survivor_loc] >= 2
        reward += m.successful_extraction_reward
    else
        # Penalty for failed extraction attempts
        for loc in 1:m.num_locations
            if extraction_counts[loc] > 0 && (loc != survivor_loc || extraction_counts[loc] < 2)
                reward += extraction_counts[loc] * m.failed_extraction_cost
            end
        end
    end
    
    return reward
end

# Additional helper functions for working with individual agents

# Get individual agent actions
function agent_actions(m::MultiAgentRescuePOMDP)
    actions = ["search"]
    for i in 1:m.num_locations
        push!(actions, "extract-loc-$i")
    end
    return actions
end

# Get individual agent observations
function agent_observations(m::MultiAgentRescuePOMDP)
    observations = ["no-signal"]
    for i in 1:m.num_locations
        push!(observations, "signal-loc-$i")
    end
    return observations
end

# Example controller creation function
function create_initial_controllers(m::MultiAgentRescuePOMDP, controller_size::Int=1)
    # Create a controller for each agent
    controllers = AgentController[]
    
    act_list = agent_actions(m)
    obs_list = agent_observations(m)
    
    for agent in 1:m.num_agents
        nodes = FSCNode[]
        
        # Add nodes to the controller
        for i in 1:controller_size
            # Default action: search
            action_idx = 1
            
            # Default transitions: stay in the same node
            transitions = Dict{String, Int}()
            for obs in obs_list
                transitions[obs] = i
            end
            
            push!(nodes, FSCNode(action_idx, transitions))
        end
        
        push!(controllers, AgentController(nodes))
    end
    
    return JointController(controllers)
end

# Example evaluation function
function evaluate_controller(joint_controller::JointController, prob::MultiAgentRescuePOMDP; 
                           max_iter=1000, tolerance=1e-6)
    # This is a placeholder for your actual evaluation function
    # It should return the expected value of the controller and belief state info
    
    # Assume value is the negative of the complexity as a simple placeholder
    value = -sum(length(controller.nodes) for controller in joint_controller.controllers)
    
    # In a real implementation, you'd compute the actual value through value iteration
    # or policy evaluation
    
    return value, Dict() # Return value and empty belief info
end