using Pkg
using Statistics
using POMDPTools
Pkg.activate("dec_pomdp_env")
include("../Problems/satellite_dec_pomdp.jl")


function dec_pomdp_pi(controller::JointController, prob)
    # Initialize
    it = 0
    epsilon = 0.01  # Desired precision
    R_max = find_maximum_absolute_reward(prob)
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
            # println("Backing up agent $i...")
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
        # println("After backup, controller value: $(V_curr)")
        
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

function find_maximum_absolute_reward(prob::POMDP)
    max_abs_reward = 0.0
    
    # Iterate through all states
    for s in states(prob)
        # Iterate through all joint actions
        for a in actions(prob)
            # Calculate reward for this state-action pair
            r = reward(prob, s, a)
            
            # Update maximum if this absolute reward is higher
            if abs(r) > max_abs_reward
                max_abs_reward = abs(r)
            end
        end
    end
    
    return max_abs_reward
end

# Perform exhaustive backup on a controller
function improved_exhaustive_backup(controller::AgentController, joint_controller::JointController, agent_idx::Int, prob::POMDP, num_agents::Int=2)
    original_controller = deepcopy(controller)
    
    # Extract individual agent actions from the joint actions
    joint_actions = POMDPs.actions(prob)
    
    # Get unique actions for this specific agent
    # For a tuple with num_agents elements, extract the agent_idx element
    agent_actions = unique([a[agent_idx] for a in joint_actions])
    
    # Get individual agent observations
    joint_observations = POMDPs.observations(prob)
    agent_observations = unique([o[agent_idx] for o in joint_observations])
    
    current_nodes = length(controller.nodes)
    candidate_nodes = Vector{FSCNode}()
    
    # Generate all possible new nodes
    for action_idx in 1:length(agent_actions)
        num_observations = length(agent_observations)
        num_combinations = current_nodes^num_observations
        
        for combo_idx in 0:(num_combinations - 1)
            transitions = Dict{String, Int}()
            
            for (obs_idx, obs) in enumerate(agent_observations)
                divisor = current_nodes^(num_observations - obs_idx)
                node_idx = (combo_idx ÷ divisor) % current_nodes + 1
                transitions[obs] = node_idx
            end
            
            # Add this candidate node to our list
            push!(candidate_nodes, FSCNode(action_idx, transitions))
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
    if !improved && length(controller.nodes) < 10 # Limit size to prevent explosion
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
                # We found an improvement, so we can break early
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
