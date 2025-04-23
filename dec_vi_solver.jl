using POMDPs
using POMDPTools
using LinearAlgebra
using Random

"""
    NodeController

Represents a single node in a finite state controller for an agent.
"""
struct NodeController
    id::Int                   # Unique identifier for the node
    action::String            # Action to take at this node
    transitions::Dict{String, Vector{Tuple{Float64, Int}}}  # Maps observation to (probability, next_node) tuples
end

"""
    FiniteStateController

A finite state controller for a single agent in a Dec-POMDP.
"""
struct FiniteStateController
    nodes::Vector{NodeController}  # The nodes in the controller
    initial_node::Int             # The starting node
end

"""
    ExhaustiveBackup

Performs an exhaustive backup of a finite state controller to improve its value.
"""
function ExhaustiveBackup(fsc::FiniteStateController, dec_pomdp::DecTigerPOMDP, agent_idx::Int, other_agent_fsc::FiniteStateController)
    # Get problem components
    S = states(dec_pomdp)
    A_i = ["listen", "open-left", "open-right"]  # Actions for this agent
    O_i = ["hear-left", "hear-right"]            # Observations for this agent
    
    new_nodes = Vector{NodeController}()
    
    # For each possible action for this agent
    for a_i in A_i
        # For each possible next node mapping for each observation
        observation_mappings = Dict{String, Vector{Tuple{Float64, Int}}}()
        
        for o_i in O_i
            # For simple deterministic controllers, just map to a node with probability 1
            # In more complex implementations, we would consider all possible distributions
            observation_mappings[o_i] = [(1.0, rand(1:length(fsc.nodes)))]
        end
        
        # Create a new node and add it to the controller
        new_node = NodeController(length(new_nodes) + 1, a_i, observation_mappings)
        push!(new_nodes, new_node)
    end
    
    # Create a new controller with the new nodes
    # For simplicity, we maintain the same initial node or set to 1 if it's out of bounds
    initial_node = (fsc.initial_node <= length(new_nodes)) ? fsc.initial_node : 1
    return FiniteStateController(new_nodes, initial_node)
end

"""
    ComputeControllerValue

Computes the value function for a joint controller.
"""
function ComputeControllerValue(dec_pomdp::DecTigerPOMDP, 
                               controllers::Vector{FiniteStateController})
    # Get problem components
    S = states(dec_pomdp)
    discount_factor = discount(dec_pomdp)
    
    # Create value function matrix
    # Dimensions: |S| x |Q_1| x |Q_2| (for 2 agents)
    # For simplicity, let's use a Dict for sparse representation
    V = Dict{Tuple{String, Int, Int}, Float64}()
    
    # Initialize with zeros
    for s in S
        for q1 in 1:length(controllers[1].nodes)
            for q2 in 1:length(controllers[2].nodes)
                V[(s, q1, q2)] = 0.0
            end
        end
    end
    
    # Value iteration to compute controller value
    max_iterations = 100
    epsilon = 1e-6
    
    for iter in 1:max_iterations
        delta = 0.0
        
        # Update value for each state-controller node combination
        for s in S
            for q1 in 1:length(controllers[1].nodes)
                for q2 in 1:length(controllers[2].nodes)
                    # Get actions from current controller nodes
                    a1 = controllers[1].nodes[q1].action
                    a2 = controllers[2].nodes[q2].action
                    joint_action = (a1, a2)
                    
                    # Get immediate reward
                    r = reward(dec_pomdp, s, joint_action)
                    
                    # Compute expected future reward
                    future_value = 0.0
                    
                    # For each possible next state
                    for sp in S
                        # Get transition probability
                        trans_dist = transition(dec_pomdp, s, joint_action)
                        trans_prob = pdf(trans_dist, sp)
                        
                        if trans_prob > 0
                            # For each possible observation combination
                            obs_dist = observation(dec_pomdp, joint_action, sp)
                            
                            for o1 in ["hear-left", "hear-right"]
                                for o2 in ["hear-left", "hear-right"]
                                    joint_obs = (o1, o2)
                                    
                                    # Probability of this observation
                                    obs_prob = pdf(obs_dist, joint_obs)
                                    
                                    if obs_prob > 0
                                        # Get next controller nodes
                                        next_q1_dist = controllers[1].nodes[q1].transitions[o1]
                                        next_q2_dist = controllers[2].nodes[q2].transitions[o2]
                                        
                                        # For each possible next controller node combination
                                        for (p1, next_q1) in next_q1_dist
                                            for (p2, next_q2) in next_q2_dist
                                                future_value += trans_prob * obs_prob * p1 * p2 * 
                                                               V[(sp, next_q1, next_q2)]
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    
                    # New value is immediate reward plus discounted future reward
                    new_value = r + discount_factor * future_value
                    
                    # Update max delta
                    delta = max(delta, abs(new_value - V[(s, q1, q2)]))
                    
                    # Update value function
                    V[(s, q1, q2)] = new_value
                end
            end
        end
        
        # Check for convergence
        if delta < epsilon
            break
        end
    end
    
    return V
end

"""
    PruneNode

Prunes dominated nodes from a controller.
"""
function PruneNode(agent_idx::Int, other_agent_fscs::Vector{FiniteStateController}, 
                  fsc::FiniteStateController, dec_pomdp::DecTigerPOMDP)
    # Get problem components
    S = states(dec_pomdp)
    
    # For simplicity, we'll just remove a random node if there are more than 1
    # In a real implementation, we would use linear programming to find dominated nodes
    if length(fsc.nodes) <= 1
        return fsc, false  # Nothing pruned
    end
    
    # Remove a random node that is not the initial node
    prunable_nodes = filter(i -> i != fsc.initial_node, 1:length(fsc.nodes))
    
    if isempty(prunable_nodes)
        return fsc, false  # Nothing pruned
    end
    
    node_to_remove = rand(prunable_nodes)
    
    # Create new set of nodes without the pruned node
    new_nodes = [n for n in fsc.nodes if n.id != node_to_remove]
    
    # Update node IDs to be sequential
    for (i, node) in enumerate(new_nodes)
        # Create a new node with updated ID
        new_transitions = Dict{String, Vector{Tuple{Float64, Int}}}()
        for (obs, trans) in node.transitions
            # Update transitions to handle removed node
            new_trans = [(p, q > node_to_remove ? q-1 : q) for (p, q) in trans if q != node_to_remove]
            
            # If all transitions were to the removed node, redirect to a random remaining node
            if isempty(new_trans)
                new_trans = [(1.0, rand(1:length(new_nodes)))]
            end
            
            new_transitions[obs] = new_trans
        end
        
        # Replace the node
        new_nodes[i] = NodeController(i, node.action, new_transitions)
    end
    
    # Update initial node if needed
    new_initial = fsc.initial_node > node_to_remove ? fsc.initial_node - 1 : fsc.initial_node
    
    return FiniteStateController(new_nodes, new_initial), true  # Node was pruned
end

"""
    UpdateController

Updates the controller after pruning to maintain a consistent structure.
"""
function UpdateController(fsc::FiniteStateController)
    # For our simple implementation, no further update is needed
    # In more complex implementations, this might involve recomputing node values
    return fsc
end

"""
    PolicyIteration

Implements the policy iteration algorithm for Dec-POMDPs.
"""
function PolicyIteration(dec_pomdp::DecTigerPOMDP, initial_controllers::Vector{FiniteStateController}; 
                        max_iterations=10, epsilon=0.1)
    # Number of agents
    n_agents = 2
    
    # Initialize
    tau = 0
    controllers_tau = initial_controllers
    
    # Get maximum possible reward for convergence check
    R_max = 20.0  # For Dec-Tiger, the max reward is +20
    discount_factor = discount(dec_pomdp)
    
    # Main policy iteration loop
    while true
        println("Iteration $tau")
        
        # Backup and evaluate
        new_controllers = Vector{FiniteStateController}()
        
        for i in 1:n_agents
            # Get other agent controllers
            other_agent_idx = i == 1 ? 2 : 1
            other_agent_fsc = controllers_tau[other_agent_idx]
            
            # Perform exhaustive backup
            new_fsc = ExhaustiveBackup(controllers_tau[i], dec_pomdp, i, other_agent_fsc)
            push!(new_controllers, new_fsc)
        end
        
        # Update controllers
        controllers_tau = new_controllers
        
        # Compute value of joint controller
        V = ComputeControllerValue(dec_pomdp, controllers_tau)
        
        # Prune dominated nodes until none can be removed
        nodes_pruned = true
        while nodes_pruned
            nodes_pruned = false
            
            for i in 1:n_agents
                # Get other agent controllers
                other_agent_controllers = [controllers_tau[j] for j in 1:n_agents if j != i]
                
                # Prune dominated nodes
                new_fsc, was_pruned = PruneNode(i, other_agent_controllers, controllers_tau[i], dec_pomdp)
                
                if was_pruned
                    nodes_pruned = true
                    controllers_tau[i] = new_fsc
                    
                    # Update controller
                    controllers_tau[i] = UpdateController(controllers_tau[i])
                    
                    # Recompute value
                    V = ComputeControllerValue(dec_pomdp, controllers_tau)
                end
            end
        end
        
        # Check for convergence
        tau += 1
        if tau >= max_iterations || discount_factor^(tau+1) * R_max <= epsilon * (1 - discount_factor)
            break
        end
    end
    
    return controllers_tau
end

"""
    CreateInitialControllers

Creates initial finite state controllers for each agent.
"""
function CreateInitialControllers(dec_pomdp::DecTigerPOMDP, n_nodes::Int=2)
    controllers = Vector{FiniteStateController}()
    
    for agent_idx in 1:2
        nodes = Vector{NodeController}()
        
        for i in 1:n_nodes
            # Assign random action
            actions = ["listen", "open-left", "open-right"]
            action = actions[rand(1:length(actions))]
            
            # Create random transitions
            transitions = Dict{String, Vector{Tuple{Float64, Int}}}()
            for obs in ["hear-left", "hear-right"]
                transitions[obs] = [(1.0, rand(1:n_nodes))]
            end
            
            push!(nodes, NodeController(i, action, transitions))
        end
        
        push!(controllers, FiniteStateController(nodes, 1))
    end
    
    return controllers
end

"""
    EvaluateControllers

Evaluates a joint controller by simulating the Dec-POMDP.
"""
function EvaluateControllers(dec_pomdp::DecTigerPOMDP, controllers::Vector{FiniteStateController}, 
                            n_episodes::Int=100, max_steps::Int=20)
    total_reward = 0.0
    
    for episode in 1:n_episodes
        # Initialize state
        state_dist = initialstate(dec_pomdp)
        state = rand(state_dist)
        
        # Initialize controller nodes
        controller_nodes = [controllers[i].initial_node for i in 1:2]
        
        episode_reward = 0.0
        discount_factor = discount(dec_pomdp)
        
        for step in 1:max_steps
            # Get actions from controller nodes
            a1 = controllers[1].nodes[controller_nodes[1]].action
            a2 = controllers[2].nodes[controller_nodes[2]].action
            joint_action = (a1, a2)
            
            # Get reward
            r = reward(dec_pomdp, state, joint_action)
            episode_reward += (discount_factor ^ (step - 1)) * r
            
            # Get next state
            next_state_dist = transition(dec_pomdp, state, joint_action)
            next_state = rand(next_state_dist)
            
            # Get observations
            obs_dist = observation(dec_pomdp, joint_action, next_state)
            joint_obs = rand(obs_dist)
            
            # Update controller nodes
            for i in 1:2
                obs = joint_obs[i]
                # Follow the transitions in the controller
                next_node_dist = controllers[i].nodes[controller_nodes[i]].transitions[obs]
                # Sample from distribution
                rand_val = rand()
                cumulative_prob = 0.0
                for (prob, next_node) in next_node_dist
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob
                        controller_nodes[i] = next_node
                        break
                    end
                end
            end
            
            # Update state
            state = next_state
        end
        
        total_reward += episode_reward
    end
    
    return total_reward / n_episodes
end

# Usage example:
function solve_dec_tiger()
    # Create Dec-Tiger POMDP
    dec_tiger = DecTigerPOMDP()
    
    # Create initial controllers
    initial_controllers = CreateInitialControllers(dec_tiger, 2)
    
    # Run policy iteration
    final_controllers = PolicyIteration(dec_tiger, initial_controllers, max_iterations=5)
    
    # Evaluate the controllers
    avg_reward = EvaluateControllers(dec_tiger, final_controllers)
    
    println("Average reward: $avg_reward")
    
    return final_controllers
end