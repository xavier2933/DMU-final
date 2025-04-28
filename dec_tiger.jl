# dec_tiger.jl - modified version
using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat

"""
    DecTigerPOMDP

Implementation of the Dec-Tiger problem as described in the DPOMDP file.
"""
struct DecTigerPOMDP <: POMDP{String, Tuple{String,String}, Tuple{String,String}}
    discount_factor::Float64
    listen_cost::Float64
    tiger_penalty::Float64
    escape_reward::Float64
    open_same_doors_penalty::Float64
    one_listens_one_opens_cost::Float64
    p_correct_obs::Float64
end

# Default constructor with values from the DPOMDP file
function DecTigerPOMDP(;
    discount_factor = 1.0,
    listen_cost = -2.0,
    tiger_penalty = -50.0,
    escape_reward = 20.0,
    open_same_doors_penalty = -100.0,
    one_listens_one_opens_cost = -101.0,
    one_escapes_reward = 9.0,
    p_correct_obs = 0.85
)
    return DecTigerPOMDP(
        discount_factor,
        listen_cost,
        tiger_penalty,
        escape_reward,
        open_same_doors_penalty,
        one_listens_one_opens_cost,
        p_correct_obs
    )
end

# Define state space 
function POMDPs.states(m::DecTigerPOMDP)
    return ["tiger-left", "tiger-right"]
end

# Define action space for each agent
function POMDPs.actions(m::DecTigerPOMDP)
    # Tuple of actions for both agents: (agent1_action, agent2_action)
    agent_actions = ["listen", "open-left", "open-right"]
    joint_actions = Tuple{String,String}[]
    
    for a1 in agent_actions
        for a2 in agent_actions
            push!(joint_actions, (a1, a2))
        end
    end
    
    return joint_actions
end

# Define observation space for each agent
function POMDPs.observations(m::DecTigerPOMDP)
    # Tuple of observations for both agents: (agent1_obs, agent2_obs)
    agent_obs = ["hear-left", "hear-right"]
    joint_obs = Tuple{String,String}[]
    
    for o1 in agent_obs
        for o2 in agent_obs
            push!(joint_obs, (o1, o2))
        end
    end
    
    return joint_obs
end

# Define initial state distribution (uniform)
function POMDPs.initialstate(m::DecTigerPOMDP)
    return Uniform(states(m))
end

# Define discount factor
function POMDPs.discount(m::DecTigerPOMDP)
    return m.discount_factor
end

# Define transition function
function POMDPs.transition(m::DecTigerPOMDP, s::String, a::Tuple{String,String})
    a1, a2 = a
    
    # If both agents listen, tiger stays where it is
    if a1 == "listen" && a2 == "listen"
        return Deterministic(s) 
    else
        # If any door is opened, tiger location is reset uniformly
        return Uniform(states(m))
    end
end

# Define observation function
function POMDPs.observation(m::DecTigerPOMDP, a::Tuple{String,String}, sp::String)
    a1, a2 = a
    p_correct = m.p_correct_obs
    
    # Calculate joint observation probabilities
    if a1 == "listen" && a2 == "listen"
        if sp == "tiger-left"
            # Probabilities from the DPOMDP file for tiger-left
            probs = [
                0.7225,  # hear-left, hear-left
                0.1275,  # hear-left, hear-right
                0.1275,  # hear-right, hear-left
                0.0225   # hear-right, hear-right
            ]
            return SparseCat(observations(m), probs)
        else  # sp == "tiger-right"
            # Probabilities from the DPOMDP file for tiger-right
            probs = [
                0.0225,  # hear-left, hear-left
                0.1275,  # hear-left, hear-right
                0.1275,  # hear-right, hear-left
                0.7225   # hear-right, hear-right
            ]
            return SparseCat(observations(m), probs)
        end
    else
        # If any door is opened, observation is uniform
        return Uniform(observations(m))
    end
end

# Define reward function
# Define reward function
# Define reward function - keeping the same function signature
function POMDPs.reward(m::DecTigerPOMDP, s::String, a::Tuple{String,String})
    a1, a2 = a
    
    if a1 == "listen" && a2 == "listen"
        return -2.0  # Both agents listen
    end
    
    # At least one agent opens a door
    if s == "tiger-left"
        # Tiger is on the left
        if a1 == "open-right" && a2 == "open-right"
            return 20.0  # Both open correct door
        elseif a1 == "open-left" && a2 == "open-left"
            return -50.0  # Both open tiger door
        else
            return -15.0  # Mixed door opening
        end
    else  # s == "tiger-right"
        # Tiger is on the right
        if a1 == "open-left" && a2 == "open-left"
            return 20.0  # Both open correct door
        elseif a1 == "open-right" && a2 == "open-right"
            return -50.0  # Both open tiger door
        else
            return -15.0  # Mixed door opening
        end
    end
end

# Create the POMDP model
dec_tiger = DecTigerPOMDP()

# For policy iteration, you'll need to implement a solver
# This is a placeholder for where you would use a policy iteration algorithm
# Exact policy iteration for Dec-POMDPs is challenging due to the complexity

# Example of how to use a solver (you'll need to define or use an appropriate solver)
# using DecPOMDPSolver  # This is a hypothetical package
# solver = PolicyIterationSolver()
# policy = solve(solver, dec_tiger)

# For simulation or evaluation:
# sim = Simulator()
# result = simulate(sim, dec_tiger, policy)