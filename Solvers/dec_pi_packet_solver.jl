using POMDPs
using POMDPTools: Deterministic, Uniform, SparseCat
using Statistics
using Random


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


function dec_pomdp_pi(controller::JointController, prob::POMDP)
    it = 0
    epsilon = 0.01 
    R_max = 100.0
    gamma = prob.discount_factor
    
    ctrlr_t = deepcopy(controller)
    n = length(ctrlr_t.controllers)
    
    V_prev = evaluate_controller(ctrlr_t, prob)
    println("Initial controller value: $(V_prev)")
    
    V_curr = V_prev
    improvement = Inf
    
    # 30 seemed reasonable for this problem with pi
    while it < 30 && improvement > epsilon
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
        
        V_curr = evaluate_controller(ctrlr_t, prob)
        println("After backup, controller value: $(V_curr)")
        
        improvement = abs(V_curr - V_prev)
        println("Improvement: $(improvement)")
        
        V_prev = V_curr
        
        it += 1
        println("Completed iteration $(it)")
        
        # Stopping criteria from book
        if (gamma^it * R_max) < epsilon
            println("Theoretical bound reached, algorithm converged.")
            break
        end
    end
    
    if it >= 30
        println("Stopped due to maximum iterations reached.")
    elseif improvement <= epsilon
        println("Stopped due to convergence (improvement below threshold).")
    end
    
    return it, ctrlr_t
end

# Improved exhaustive backup function - help from Claude
function improved_exhaustive_backup(controller::AgentController, joint_controller::JointController, agent_idx::Int, prob::POMDP)
    original_controller = deepcopy(controller)
    
    # Pre-compute these values once
    agent_act = agent_actions(prob, agent_idx)
    obs_list = agent_observations(prob)
    current_nodes = length(controller.nodes)
    
    # Pre-allocate with a reasonable capacity to avoid resizing
    candidate_nodes = Vector{FSCNode}(undef, length(agent_act) * 5)
    candidate_count = 0
    
    # Generate candidate nodes more efficiently
    for action_idx in 1:length(agent_act)
        # Simple pattern: go to node 1 for all observations
        transitions1 = Dict{String, Int}()
        for obs in obs_list
            transitions1[obs] = 1
        end
        candidate_count += 1
        candidate_nodes[candidate_count] = FSCNode(action_idx, transitions1)
        
        # Pattern: separate nodes for data vs no data observations
        if current_nodes >= 2
            transitions2 = Dict{String, Int}()
            for obs in obs_list
                # Go to node 1 if observation suggests we have data, else node 2
                transitions2[obs] = has_packets(obs, agent_idx) ? 1 : 2
            end
            candidate_count += 1
            candidate_nodes[candidate_count] = FSCNode(action_idx, transitions2)
        end
        
        # Add random transition patterns (limit to 3)
        for _ in 1:3
            transitions_rand = Dict{String, Int}()
            for obs in obs_list
                transitions_rand[obs] = rand(1:max(2, current_nodes))
            end
            candidate_count += 1
            candidate_nodes[candidate_count] = FSCNode(action_idx, transitions_rand)
        end
    end
    
    # Resize to actual number of candidates
    resize!(candidate_nodes, candidate_count)
    
    # Evaluate the current controller value just once
    current_value = evaluate_controller(joint_controller, prob)
    println("Current value before backup: $current_value")
    
    # Track best results
    best_controller = deepcopy(controller)
    best_value = current_value
    improved = false
    
    # Reuse these temporary objects to reduce allocations
    temp_controller = deepcopy(controller)
    temp_joint_controller = deepcopy(joint_controller)
    
    # For each existing node in the controller
    for node_idx in 1:length(controller.nodes)
        # For better performance, avoid recreation of the entire controller each time
        original_node = controller.nodes[node_idx]
        
        # Try replacing with each candidate node
        for candidate in candidate_nodes
            # Replace just the specific node instead of deep copying everything
            temp_controller.nodes[node_idx] = candidate
            
            # Update the joint controller reference
            temp_joint_controller.controllers[agent_idx] = temp_controller
            
            # Evaluate this controller
            temp_value = evaluate_controller(temp_joint_controller, prob)
            
            # If it's better, keep it
            if temp_value > best_value
                best_value = temp_value
                # Only deepcopy when we find a better solution
                best_controller = deepcopy(temp_controller)
                improved = true
                println("Found improvement by replacing node $node_idx, new value: $temp_value")
            end
        end
        
        # Restore the original node for the next iteration
        temp_controller.nodes[node_idx] = original_node
    end
    
    # Try adding a new node (only if no improvement was found and controller isn't too large)
    if !improved && length(controller.nodes) < 8
        # Prepare the temporary controller with space for a new node
        temp_controller = deepcopy(controller)
        push!(temp_controller.nodes, FSCNode(1, Dict{String, Int}())) # Placeholder
        
        for (i, candidate) in enumerate(candidate_nodes)
            # Replace the last node with the candidate
            temp_controller.nodes[end] = candidate
            
            # Update the joint controller reference
            temp_joint_controller.controllers[agent_idx] = temp_controller
            
            # Evaluate this controller
            temp_value = evaluate_controller(temp_joint_controller, prob)
            
            # If it's better, keep it
            if temp_value > best_value
                best_value = temp_value
                best_controller = deepcopy(temp_controller)
                improved = true
                println("Found improvement by adding a new node, new value: $temp_value")
                # Break early when we find an improvement
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

function evaluate_controller(joint_controller::JointController, prob::POMDP; max_iter=5000, tolerance=1e-6)
    all_states = POMDPs.states(prob)
    
    # Map states to indices for easier array access
    state_map = Dict(s => i for (i, s) in enumerate(all_states))
    
    num_states = length(all_states)
    num_satellites = length(joint_controller.controllers)
    nodes_per_sat = [length(c.nodes) for c in joint_controller.controllers]
    
    # Create value function matrix
    # Dimensions: one for each satellite's node, plus one for state
    dims = [nodes_per_sat..., num_states]
    V = zeros(dims...)
    
    gamma = prob.discount_factor
    
    terminal_states = [s == "all-transmitted" for s in all_states]
    
    action_mappings = [
        [joint_controller.controllers[i].nodes[n].action for n in 1:nodes_per_sat[i]]
        for i in 1:num_satellites
    ]
    
    for iter in 1:max_iter
        V_new = copy(V)
        max_delta = 0.0
        
        for node_indices in Iterators.product([1:n for n in nodes_per_sat]...)
            for s in 1:num_states

                if terminal_states[s]
                    V_new[node_indices..., s] = 0.0
                    continue
                end
                
                state = all_states[s]
                
                joint_action_indices = [action_mappings[i][node_indices[i]] 
                                      for i in 1:num_satellites]
                
                # Convert action indices to strings
                joint_action = Tuple(agent_actions(prob, i)[joint_action_indices[i]] 
                                   for i in 1:num_satellites)
                
                immediate_reward = POMDPs.reward(prob, state, joint_action)
                
                future_reward = 0.0
                
                for next_s in 1:num_states
                    next_state = all_states[next_s]
                    
                    trans_distribution = POMDPs.transition(prob, state, joint_action)
                    
                    trans_prob = 0.0
                    

                    for (idx, s_val) in enumerate(trans_distribution.vals)
                        if s_val == next_state
                            trans_prob = trans_distribution.probs[idx]
                            break
                        end
                    end

                    if trans_prob ≈ 0.0
                        continue
                    end
                    
                    obs_distribution = POMDPs.observation(prob, joint_action, next_state)
                    
                    if typeof(obs_distribution) <: Deterministic
                        next_obs = obs_distribution.val
                        
                        next_nodes = Vector{Int}(undef, num_satellites)
                        for i in 1:num_satellites
                            current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                            next_node_idx = current_node.transitions[next_obs[i]]
                            next_nodes[i] = next_node_idx
                        end
                        
                        future_value = V[next_nodes..., next_s]
                        
                        future_reward += trans_prob * future_value
                    else
                        next_nodes = Vector{Int}(undef, num_satellites)
                        
                        for (obs_idx, joint_obs) in enumerate(obs_distribution.vals)
                            obs_prob = obs_distribution.probs[obs_idx]
                            
                            if obs_prob ≈ 0.0
                                continue
                            end
                            
                            for i in 1:num_satellites
                                current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                                
                                if haskey(current_node.transitions, joint_obs[i])
                                    next_nodes[i] = current_node.transitions[joint_obs[i]]
                                else
                                    next_nodes[i] = node_indices[i]
                                end
                            end
                            
                            future_value = V[next_nodes..., next_s]
                            future_reward += trans_prob * obs_prob * future_value
                        end
                    end
                end
                
                V_new[node_indices..., s] = immediate_reward + gamma * future_reward
                
                delta = abs(V_new[node_indices..., s] - V[node_indices..., s])
                max_delta = max(max_delta, delta)
            end
        end
        
        V = copy(V_new)
        
        if max_delta < tolerance
            break
        end
        
        if iter == max_iter
            @warn "Value iteration did not converge after $(max_iter) iterations"
        end
    end
    
    initial_state = initialstate(prob).val
    initial_state_idx = state_map[initial_state]
    
    initial_nodes = [1 for _ in 1:num_satellites]
    
    value = V[initial_nodes..., initial_state_idx]
    
    return value
end