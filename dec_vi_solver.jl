# dec-tiger-policy-iteration.jl
using Pkg
Pkg.activate("dec_pomdp_env")  # Create a new environment
include("dec_tiger.jl")  # Your Dec-Tiger implementation file

using POMDPs
using POMDPTools
using LinearAlgebra
using JuMP
using GLPK

# First, let's define the controller structure
struct FiniteStateController
    # Number of nodes in the controller
    num_nodes::Int
    
    # Action for each node
    action_map::Dict{Int, String}
    
    # Next node for each (current_node, observation) pair
    # transition_map[current_node][observation] = next_node
    transition_map::Dict{Int, Dict{String, Int}}
end

# Initialize a simple controller with one node for each action
function create_initial_controller(actions::Vector{String}, observations::Vector{String})
    num_nodes = length(actions)
    action_map = Dict{Int, String}()
    transition_map = Dict{Int, Dict{String, Int}}()
    
    # Create a simple controller where each node executes a different action
    # and transitions back to itself regardless of the observation
    for (i, action) in enumerate(actions)
        action_map[i] = action
        transition_map[i] = Dict{String, Int}()
        
        for obs in observations
            transition_map[i][obs] = i  # Stay in the same node
        end
    end
    
    return FiniteStateController(num_nodes, action_map, transition_map)
end

# Function to perform an exhaustive backup for a single agent's controller
function exhaustive_backup(controller::FiniteStateController, actions::Vector{String}, observations::Vector{String})
    old_num_nodes = controller.num_nodes
    new_action_map = copy(controller.action_map)
    new_transition_map = deepcopy(controller.transition_map)
    
    # For each possible action
    node_counter = old_num_nodes + 1
    
    for action in actions
        # For each possible combination of (observation, next_node)
        for obs in observations
            for next_node in 1:old_num_nodes
                # Add a new node to the controller with this action and transition
                new_action_map[node_counter] = action
                new_transition_map[node_counter] = Dict{String, Int}()
                
                # Set transitions for all observations
                for o in observations
                    if o == obs
                        new_transition_map[node_counter][o] = next_node
                    else
                        # Default transition for other observations (can be improved)
                        new_transition_map[node_counter][o] = 1  # Default to first node
                    end
                end
                
                node_counter += 1
            end
        end
    end
    
    return FiniteStateController(node_counter - 1, new_action_map, new_transition_map)
end

# Function to evaluate a joint controller for a Dec-POMDP
function evaluate_joint_controller(pomdp::DecTigerPOMDP, 
                                  controllers::Vector{FiniteStateController})
    # Get all states
    S = states(pomdp)
    
    # Number of nodes in each controller
    num_nodes = [c.num_nodes for c in controllers]
    
    # Create a value table: V[s, n1, n2] = value of being in state s with agents in nodes n1, n2
    V = Dict{Tuple{String, Int, Int}, Float64}()
    
    # Initialize values to 0
    for s in S
        for n1 in 1:num_nodes[1]
            for n2 in 1:num_nodes[2]
                V[(s, n1, n2)] = 0.0
            end
        end
    end
    
    # Iterate until convergence
    gamma = discount(pomdp)
    epsilon = 1e-6
    max_iter = 1000
    
    for iter in 1:max_iter
        delta = 0.0
        
        for s in S
            for n1 in 1:num_nodes[1]
                for n2 in 1:num_nodes[2]
                    old_value = V[(s, n1, n2)]
                    
                    # Get actions from controller nodes
                    a1 = controllers[1].action_map[n1]
                    a2 = controllers[2].action_map[n2]
                    joint_action = (a1, a2)
                    
                    # Get immediate reward
                    r = reward(pomdp, s, joint_action)
                    
                    # Calculate expected future value
                    future_value = 0.0
                    
                    # For each possible next state
                    for sp in S
                        # Get transition probability
                        trans_prob = pdf(transition(pomdp, s, joint_action), sp)
                        
                        # For each possible joint observation
                        obs_dist = observation(pomdp, joint_action, sp)
                        for joint_obs in support(obs_dist)
                            o1, o2 = joint_obs
                            obs_prob = pdf(obs_dist, joint_obs)
                            
                            # Get next nodes according to controllers
                            next_n1 = controllers[1].transition_map[n1][o1]
                            next_n2 = controllers[2].transition_map[n2][o2]
                            
                            # Add to future value
                            future_value += trans_prob * obs_prob * V[(sp, next_n1, next_n2)]
                        end
                    end
                    
                    # Update value
                    V[(s, n1, n2)] = r + gamma * future_value
                    
                    # Track maximum change
                    delta = max(delta, abs(V[(s, n1, n2)] - old_value))
                end
            end
        end
        
        # Check for convergence
        if delta < epsilon
            println("Value iteration converged after $iter iterations")
            break
        end
        
        if iter == max_iter
            println("Warning: Value iteration reached maximum iterations without converging")
        end
    end
    
    return V
end

# Function to prune dominated nodes from a controller
function prune_controller(pomdp::DecTigerPOMDP, 
                         agent_idx::Int, 
                         controllers::Vector{FiniteStateController}, 
                         V::Dict{Tuple{String, Int, Int}, Float64})
    controller = controllers[agent_idx]
    S = states(pomdp)
    
    # No need to prune if there's only one node
    if controller.num_nodes <= 1
        return controller, false
    end
    
    # For each node, check if it's dominated by a convex combination of other nodes
    for node_to_check in 1:controller.num_nodes
        # Create a linear program to check if node is dominated
        model = Model(GLPK.Optimizer)
        set_silent(model)
        
        # Variables: weights for each other node
        @variable(model, w[1:controller.num_nodes] >= 0)
        
        # Constraint: weights sum to 1
        @constraint(model, sum(w) == 1)
        
        # Constraint: weight of node being checked is 0
        @constraint(model, w[node_to_check] == 0)
        
        # For each state and other agent's node, the value of this node should be
        # less than or equal to the weighted sum of other nodes
        for s in S
            if agent_idx == 1
                for other_node in 1:controllers[2].num_nodes
                    # Get value of current node
                    v_current = V[(s, node_to_check, other_node)]
                    
                    # Add constraint: weighted sum of other nodes must have >= value
                    @constraint(model, 
                               sum(w[n] * V[(s, n, other_node)] for n in 1:controller.num_nodes) >= v_current)
                end
            else # agent_idx == 2
                for other_node in 1:controllers[1].num_nodes
                    # Get value of current node
                    v_current = V[(s, other_node, node_to_check)]
                    
                    # Add constraint: weighted sum of other nodes must have >= value
                    @constraint(model, 
                               sum(w[n] * V[(s, other_node, n)] for n in 1:controller.num_nodes) >= v_current)
                end
            end
        end
        
        # Objective: maximize any slack (not necessary, we just need feasibility)
        @objective(model, Max, 1)
        
        # Solve the model
        optimize!(model)
        
        # If a solution exists, the node is dominated and can be pruned
        if termination_status(model) == MOI.OPTIMAL
            # Node is dominated, get the dominating distribution
            w_val = JuMP.value.(w)
            
            # Create new controller without the dominated node
            new_action_map = Dict{Int, String}()
            new_transition_map = Dict{Int, Dict{String, Int}}()
            
            # Copy non-dominated nodes
            new_idx = 1
            old_to_new = Dict{Int, Int}()
            
            for old_idx in 1:controller.num_nodes
                if old_idx != node_to_check
                    new_action_map[new_idx] = controller.action_map[old_idx]
                    new_transition_map[new_idx] = Dict{String, Int}()
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
                end
            end
            
            # Update transitions (note: this is simplified, actual implementation would need
            # to handle stochastic transitions based on the convex combination)
            # For simplicity, we'll use the highest weight node as the replacement
            replacement_node = argmax([i != node_to_check ? w_val[i] : -Inf for i in 1:controller.num_nodes])
            
            for old_idx in 1:controller.num_nodes
                if old_idx != node_to_check
                    new_idx = old_to_new[old_idx]
                    
                    for obs in keys(controller.transition_map[old_idx])
                        next_node = controller.transition_map[old_idx][obs]
                        
                        if next_node == node_to_check
                            # If transitioning to pruned node, redirect to replacement
                            new_transition_map[new_idx][obs] = old_to_new[replacement_node]
                        else
                            # Otherwise, update to new index
                            new_transition_map[new_idx][obs] = old_to_new[next_node]
                        end
                    end
                end
            end
            
            return FiniteStateController(controller.num_nodes - 1, new_action_map, new_transition_map), true
        end
    end
    
    # If we get here, no node could be pruned
    return controller, false
end

# Main policy iteration algorithm
function policy_iteration(pomdp::DecTigerPOMDP, epsilon::Float64=0.01)
    # Get problem information
    S = states(pomdp)
    joint_actions = actions(pomdp)
    joint_observations = observations(pomdp)
    
    # Extract individual actions and observations
    agent_actions = ["listen", "open-left", "open-right"]
    agent_observations = ["hear-left", "hear-right"]
    
    # Initialize controllers for both agents
    controllers = [
        create_initial_controller(agent_actions, agent_observations),
        create_initial_controller(agent_actions, agent_observations)
    ]
    
    # Maximum possible reward (used for termination condition)
    R_max = max(
        abs(pomdp.listen_cost),
        abs(pomdp.tiger_penalty),
        abs(pomdp.escape_reward),
        abs(pomdp.open_same_doors_penalty),
        abs(pomdp.one_listens_one_opens_cost)
    )
    
    gamma = discount(pomdp)
    iteration = 0
    
    while true
        iteration += 1
        println("Policy iteration step: $iteration")
        
        # Perform exhaustive backup for each agent
        for i in 1:2
            controllers[i] = exhaustive_backup(controllers[i], agent_actions, agent_observations)
            println("  Agent $i controller size after backup: $(controllers[i].num_nodes) nodes")
        end
        
        # Evaluate joint controller
        V = evaluate_joint_controller(pomdp, controllers)
        
        # Prune dominated nodes until no more can be removed
        pruned_something = true
        while pruned_something
            pruned_something = false
            
            for i in 1:2
                println("  Pruning agent $i controller (current size: $(controllers[i].num_nodes))")
                new_controller, was_pruned = prune_controller(pomdp, i, controllers, V)
                
                if was_pruned
                    controllers[i] = new_controller
                    # Re-evaluate after pruning
                    V = evaluate_joint_controller(pomdp, controllers)
                    pruned_something = true
                    println("    Pruned to $(controllers[i].num_nodes) nodes")
                end
            end
        end
        
        # Check termination condition: γ^(t+1) * |R_max| / (1-γ) <= ε
        bound = (gamma^(iteration+1) * R_max) / (1 - gamma)
        println("  Current error bound: $bound (target: $epsilon)")
        
        if bound <= epsilon
            println("Policy iteration converged after $iteration iterations")
            break
        end
    end
    
    return controllers, V
end

# Create the Dec-Tiger POMDP
dec_tiger = DecTigerPOMDP()

# Run policy iteration
controllers, V = policy_iteration(dec_tiger, 0.01)

# Display controller information
for (i, controller) in enumerate(controllers)
    println("Agent $i controller:")
    println("  Number of nodes: $(controller.num_nodes)")
    
    for node in 1:controller.num_nodes
        println("  Node $node:")
        println("    Action: $(controller.action_map[node])")
        println("    Transitions:")
        
        for (obs, next_node) in controller.transition_map[node]
            println("      On observation '$obs' -> Node $next_node")
        end
    end
end

# Display value function for initial belief
s0_dist = initialstate(dec_tiger)
initial_value = 0.0

for s in states(dec_tiger)
    s_prob = pdf(s0_dist, s)
    for n1 in 1:controllers[1].num_nodes
        for n2 in 1:controllers[2].num_nodes
            # We could weight by initial controller node distribution
            # Here assuming uniform initial distribution over controller nodes
            node_prob = 1.0 / (controllers[1].num_nodes * controllers[2].num_nodes)
            initial_value += s_prob * node_prob * V[(s, n1, n2)]
        end
    end
end

println("Expected value of joint policy: $initial_value")