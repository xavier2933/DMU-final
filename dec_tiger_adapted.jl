# dec_tiger_adapted.jl
include("GENERIC_dec_tiger.jl")

"""
    DecTigerPOMDP

Implementation of the Dec-Tiger problem adapted to work with the generic DecPOMDP interface.
"""
struct DecTigerPOMDP <: DecPOMDP
    discount_factor::Float64
    listen_cost::Float64
    tiger_penalty::Float64
    escape_reward::Float64
    open_same_doors_penalty::Float64
    one_listens_one_opens_cost::Float64
    p_correct_obs::Float64
    
    # Keep constructor the same
    function DecTigerPOMDP(;
        discount_factor = 1.0,
        listen_cost = -2.0,
        tiger_penalty = -50.0,
        escape_reward = 20.0,
        open_same_doors_penalty = -100.0,
        one_listens_one_opens_cost = -101.0,
        p_correct_obs = 0.85
    )
        return new(
            discount_factor,
            listen_cost,
            tiger_penalty,
            escape_reward,
            open_same_doors_penalty,
            one_listens_one_opens_cost,
            p_correct_obs
        )
    end
end

# Implement the required interface functions

function states(prob::DecTigerPOMDP)
    return ["tiger-left", "tiger-right"]
end

function num_agents(prob::DecTigerPOMDP)
    return 2
end

function actions(prob::DecTigerPOMDP)
    # Return separate action lists for each agent
    agent_actions = ["listen", "open-left", "open-right"]
    return [agent_actions, agent_actions]
end

function observations(prob::DecTigerPOMDP)
    # Return separate observation lists for each agent
    agent_obs = ["hear-left", "hear-right"]
    return [agent_obs, agent_obs]
end

function transition_probability(prob::DecTigerPOMDP, state, next_state, joint_action)
    # Convert action indices to actions
    action_names = ["listen", "open-left", "open-right"]
    a1 = action_names[joint_action[1]]
    a2 = action_names[joint_action[2]]
    
    # If both agents listen, tiger stays where it is
    if a1 == "listen" && a2 == "listen"
        return state == next_state ? 1.0 : 0.0
    else
        # If any door is opened, tiger location is reset uniformly
        return 0.5  # Equal probability for either state
    end
end

function observation_probability(prob::DecTigerPOMDP, state, joint_action, joint_obs)
    # Convert action indices to actions
    action_names = ["listen", "open-left", "open-right"]
    a1 = action_names[joint_action[1]]
    a2 = action_names[joint_action[2]]
    
    # Calculate individual observation probabilities
    probs = []
    
    for i in 1:2
        action = action_names[joint_action[i]]
        obs = joint_obs[i]
        
        if action == "listen"
            # Correct observation with probability p_correct_obs
            if (state == "tiger-left" && obs == "hear-left") || 
               (state == "tiger-right" && obs == "hear-right")
                push!(probs, prob.p_correct_obs)
            else
                push!(probs, 1.0 - prob.p_correct_obs)
            end
        else
            # After opening a door, observation is random
            push!(probs, 0.5)
        end
    end
    
    # Joint observation probability (independent observations)
    return prod(probs)
end

function reward(prob::DecTigerPOMDP, state, joint_action)
    # Convert action indices to actions
    action_names = ["listen", "open-left", "open-right"]
    a1 = action_names[joint_action[1]]
    a2 = action_names[joint_action[2]]
    
    if a1 == "listen" && a2 == "listen"
        return prob.listen_cost  # Both agents listen
    end
    
    # At least one agent opens a door
    if state == "tiger-left"
        # Tiger is on the left
        if a1 == "open-right" && a2 == "open-right"
            return prob.escape_reward  # Both open correct door
        elseif a1 == "open-left" && a2 == "open-left"
            return prob.tiger_penalty  # Both open tiger door
        else
            return prob.one_listens_one_opens_cost  # Mixed door opening
        end
    else  # state == "tiger-right"
        # Tiger is on the right
        if a1 == "open-left" && a2 == "open-left"
            return prob.escape_reward  # Both open correct door
        elseif a1 == "open-right" && a2 == "open-right"
            return prob.tiger_penalty  # Both open tiger door
        else
            return prob.one_listens_one_opens_cost  # Mixed door opening
        end
    end
end

function initial_belief(prob::DecTigerPOMDP)
    # Equal probability for either state
    return [0.5, 0.5]
end

function discount_factor(prob::DecTigerPOMDP)
    return prob.discount_factor
end

# Helper function to create the heuristic controller for Dec-Tiger
function create_heuristic_controller(prob::DecTigerPOMDP)
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2
        # Node 1: Listen
        node1 = FSCNode(
            1,  # Listen action
            Dict("hear-left" => 2, "hear-right" => 3)  # Transition based on observation
        )
        
        # Node 2: Listen again to confirm (heard tiger on left)
        node2 = FSCNode(
            1,  # Listen action
            Dict("hear-left" => 4, "hear-right" => 1)  # Confirm or go back to listening
        )
        
        # Node 3: Listen again to confirm (heard tiger on right)
        node3 = FSCNode(
            1,  # Listen action
            Dict("hear-left" => 1, "hear-right" => 5)  # Confirm or go back to listening
        )
        
        # Node 4: Open right door (confident tiger is on left)
        node4 = FSCNode(
            3,  # Open right
            Dict("hear-left" => 1, "hear-right" => 1)  # Go back to listening
        )
        
        # Node 5: Open left door (confident tiger is on right)
        node5 = FSCNode(
            2,  # Open left
            Dict("hear-left" => 1, "hear-right" => 1)  # Go back to listening
        )
        
        ctrl = AgentController([node1, node2, node3, node4, node5])
        push!(agent_controllers, ctrl)
    end
    
    return JointController(agent_controllers)
end