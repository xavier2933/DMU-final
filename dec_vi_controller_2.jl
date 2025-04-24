using Pkg
Pkg.activate("dec_pomdp_env")  # Create a new environment
include("dec_tiger.jl")  # Your Dec-Tiger implementation file


# Define the FSC node structure
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


function create_heuristic_controller(prob::DecTigerPOMDP)
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2
        # Create a controller based on a sensible strategy for Dec-Tiger
        
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

function compute_reward(state, joint_action, actions)
    # Convert indices back to action names for readability
    action1 = actions[joint_action[1]]
    action2 = actions[joint_action[2]]
    
    if action1 == "listen" && action2 == "listen"
        return -2.0  # Both agents listen
    end
    
    # At least one agent opens a door
    if state == "tiger-left"
        # Tiger is on the left
        if action1 == "open-right" && action2 == "open-right"
            return 20.0  # Both open correct door
        elseif action1 == "open-left" && action2 == "open-left"
            return -50.0  # Both open tiger door
        else
            return -15.0  # Mixed door opening
        end
    else  # state == "tiger-right"
        # Tiger is on the right
        if action1 == "open-left" && action2 == "open-left"
            return 20.0  # Both open correct door
        elseif action1 == "open-right" && action2 == "open-right"
            return -50.0  # Both open tiger door
        else
            return -15.0  # Mixed door opening
        end
    end
end

function compute_observation_probability(state, joint_action, joint_obs, actions, observations)
    # Convert indices to action and observation names
    action1 = actions[joint_action[1]]
    action2 = actions[joint_action[2]]
    obs1 = joint_obs[1]
    obs2 = joint_obs[2]
    
    # Observation probabilities for each agent
    prob1 = compute_single_observation_probability(state, action1, obs1)
    prob2 = compute_single_observation_probability(state, action2, obs2)
    
    # Joint observation probability (independent observations)
    return prob1 * prob2
end

# Helper function for individual observation probability
function compute_single_observation_probability(state, action, obs)
    if action == "listen"
        # Correct observation with 85% probability
        if (state == "tiger-left" && obs == "hear-left") || 
           (state == "tiger-right" && obs == "hear-right")
            return 0.85
        else
            return 0.15
        end
    else
        # After opening a door, observation probabilities are uniform
        return 0.5
    end
end

function evaluate_controller(joint_controller::JointController, problem::DecTigerPOMDP)

    states = ["tiger-left", "tiger-right"]
    actions = ["listen", "open-left", "open-right"]
    observations = ["hear-left", "hear-right"]
    
    # Map states to indices for easier array access
    state_map = Dict("tiger-left" => 1, "tiger-right" => 2)
    action_map = Dict("listen" => 1, "open-left" => 2, "open-right" => 3)
    obs_map = Dict("hear-left" => 1, "hear-right" => 2)
    
    # Get the number of states and number of nodes for each agent
    num_states = length(states)
    num_agents = length(joint_controller.controllers)
    nodes_per_agent = [length(c.nodes) for c in joint_controller.controllers]
    
    # Create value function matrix
    # Dimensions: one for each agent's node, plus one for state
    # For example, with 2 agents each with 4 nodes, and 2 states:
    # V is a 4×4×2 array where V[i,j,s] is the value when
    # agent 1 is in node i, agent 2 is in node j, and the state is s
    
    dims = [nodes_per_agent..., num_states]
    V = zeros(dims...)
    
    # Discount factor
    gamma = 0.9  # Adjust to match your problem definition
    
    # Value iteration to compute the value function
    max_iterations = 1000
    epsilon = 0.001
    
    for iter in 1:max_iterations
        # Make a copy of V for updating
        V_new = copy(V)
        
        # For each possible joint node configuration and state
        for node_indices in Iterators.product([1:n for n in nodes_per_agent]...)
            for s in 1:num_states
                # Get the current state name
                state = states[s]
                
                # Joint action probability (deterministic in this case)
                # For each agent, get the action they take in their current node
                joint_action = [joint_controller.controllers[i].nodes[node_indices[i]].action 
                               for i in 1:num_agents]
                
                # Compute immediate reward for this state and joint action
                immediate_reward = compute_reward(state, joint_action, actions)
                
                # Expected future reward
                future_reward = 0.0
                
                # For each possible next state and observation
                for next_s in 1:num_states
                    next_state = states[next_s]
                    
                    # Transition probability
                    # In Dec-Tiger, if any agent opens a door, state is reset randomly
                    # If both agents listen, state stays the same
                    if joint_action[1] > 1 || joint_action[2] > 1  # If any agent opens a door
                        trans_prob = 0.5  # Reset to either state with equal probability
                    else
                        # Both agents listen, state stays the same
                        trans_prob = next_s == s ? 1.0 : 0.0
                    end
                    
                    # For all possible combinations of observations
                    for obs_indices in Iterators.product([1:length(observations) for _ in 1:num_agents]...)
                        joint_obs = [observations[o] for o in obs_indices]
                        
                        # Observation probability depends on state and action
                        obs_prob = compute_observation_probability(next_state, joint_action, joint_obs, actions, observations)
                        
                        # Next nodes for each agent based on their observation
                        next_nodes = []
                        for i in 1:num_agents
                            # Get the next node index from the transition function
                            current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                            next_node = current_node.transitions[joint_obs[i]]
                            push!(next_nodes, next_node)
                        end
                        
                        # Look up future value from the PREVIOUS iteration's V
                        future_value = V[next_nodes..., next_s]
                        
                        # Add weighted contribution to expected future reward
                        future_reward += trans_prob * obs_prob * future_value
                    end
                end
                
                # Total value for this configuration
                V_new[node_indices..., s] = immediate_reward + gamma * future_reward
            end
        end
        
        # Check convergence
        if maximum(abs.(V_new - V)) < epsilon
            # println("Value iteration converged after $(iter) iterations")
            break
        end
        
        # Update value function for next iteration
        V = copy(V_new)
        
        if iter == max_iterations
            println("Warning: Value iteration did not converge after $(max_iterations) iterations")
        end
    end
    
    # Compute the value for the initial belief state and initial controller nodes
    initial_belief = [0.5, 0.5]  # Equal probability for either state
    initial_nodes = [1 for _ in 1:num_agents]  # Start at first node for each agent
    
    value = sum(initial_belief[s] * V[initial_nodes..., s] for s in 1:num_states)
    
    return value, V
end

# Prune dominated nodes from a controller
function prune_controller(agent_idx::Int, controller::AgentController, 
                         other_controllers::Vector{AgentController}, 
                         problem::DecTigerPOMDP)
    
    # Get current nodes in controller
    nodes = controller.nodes
    num_nodes = length(nodes)
    
    # If we have only one node, we can't prune
    if num_nodes <= 1
        return controller, false
    end
    
    # Get information about the problem
    states = ["tiger-left", "tiger-right"]
    num_states = length(states)
    
    # Get information about other controllers
    other_controller_sizes = [length(c.nodes) for c in other_controllers]
    
    # Create a matrix to store values for all combinations
    # V[i, j, s] = value of node i when other agents are in joint config j and state is s
    num_other_configs = prod(other_controller_sizes)
    V = zeros(num_nodes, num_other_configs, num_states)
    
    # Populate the value matrix
    # This would normally come from evaluating value function V(qi, q-i, s)
    # For demonstration, we'll compute it on-the-fly
    
    # Create all possible configurations of other agents' nodes
    other_config_indices = collect(Iterators.product([1:size for size in other_controller_sizes]...))
    
    # For each node in our controller
    for node_idx in 1:num_nodes
        # For each configuration of other agents' nodes
        for (config_idx, other_config) in enumerate(other_config_indices)
            # For each state
            for state_idx in 1:num_states
                state = states[state_idx]
                
                # Get actions for this agent and other agents
                action = nodes[node_idx].action
                other_actions = [other_controllers[i].nodes[other_config[i]].action 
                                for i in 1:length(other_controllers)]
                
                # Convert to joint action
                joint_action = vcat(action, other_actions)
                
                # Compute immediate reward
                immediate_reward = compute_reward(state, joint_action, ["listen", "open-left", "open-right"])
                
                # Compute expected future reward (this would be more complex in practice)
                # For demonstration, we'll use a simplified approach
                future_reward = 0.0
                
                # Calculate total value
                V[node_idx, config_idx, state_idx] = immediate_reward + future_reward
            end
        end
    end
    
    # Now check for dominance between nodes
    pruned_nodes = []
    pruned_any = false
    
    # Find the best node (highest average value)
    best_node_idx = 1
    best_node_value = -Inf
    
    for node_idx in 1:num_nodes
        node_avg_value = mean(V[node_idx, :, :])
        if node_avg_value > best_node_value
            best_node_value = node_avg_value
            best_node_idx = node_idx
        end
    end
    
    # For each node (except the best node)
    for node_idx in 1:num_nodes
        # Don't prune the best node
        if node_idx == best_node_idx
            continue
        end
        
        # Try to find if this node is dominated
        is_dominated = false
        dominating_node = 0
        
        for other_node in 1:num_nodes
            if other_node == node_idx
                continue
            end
            
            # Check if other_node dominates node_idx
            dominates = true
            
            for config_idx in 1:num_other_configs
                for state_idx in 1:num_states
                    if V[other_node, config_idx, state_idx] < V[node_idx, config_idx, state_idx]
                        dominates = false
                        break
                    end
                end
                if !dominates
                    break
                end
            end
            
            if dominates
                is_dominated = true
                dominating_node = other_node
                break
            end
        end
        
        if is_dominated
            push!(pruned_nodes, (node_idx, dominating_node))
            pruned_any = true
        end
    end
    
    # If no nodes were dominated, return the original controller
    if !pruned_any
        return controller, false
    end
    
    # Create a new controller without the dominated nodes
    new_nodes = []
    
    # Create a mapping from old node indices to new ones
    node_map = Dict{Int, Int}()
    new_idx = 1
    
    for old_idx in 1:num_nodes
        # Check if this node was pruned
        pruned = false
        for (pruned_idx, _) in pruned_nodes
            if old_idx == pruned_idx
                pruned = true
                break
            end
        end
        
        if !pruned
            # Add this node to the new controller
            push!(new_nodes, deepcopy(nodes[old_idx]))
            node_map[old_idx] = new_idx
            new_idx += 1
        end
    end
    
    # If we've somehow pruned all nodes, keep the best node
    if length(new_nodes) == 0
        println("Warning: Attempted to prune all nodes. Keeping the best node.")
        push!(new_nodes, deepcopy(nodes[best_node_idx]))
        node_map[best_node_idx] = 1
    end
    
    # Fix for the KeyError - also map pruned nodes to their dominating node's new index
    for (pruned_idx, dominating_idx) in pruned_nodes
        # Make sure the dominating node has a mapping
        if haskey(node_map, dominating_idx)
            node_map[pruned_idx] = node_map[dominating_idx]
        else
            # If the dominating node was also pruned, follow the chain
            # This is a simplification - ideally we'd resolve the full chain of dominance
            for (p_idx, d_idx) in pruned_nodes
                if p_idx == dominating_idx && haskey(node_map, d_idx)
                    node_map[pruned_idx] = node_map[d_idx]
                    break
                end
            end
            
            # If we still don't have a mapping, default to node 1
            if !haskey(node_map, pruned_idx)
                node_map[pruned_idx] = 1
            end
        end
    end
    
    # Update transitions in the remaining nodes
    for node in new_nodes
        for obs in keys(node.transitions)
            old_next = node.transitions[obs]
            
            # Use the node mapping to update the transition
            if haskey(node_map, old_next)
                node.transitions[obs] = node_map[old_next]
            else
                # Fallback to node 1 if mapping doesn't exist
                node.transitions[obs] = 1
            end
        end
    end
    
    return AgentController(new_nodes), true
end

function dec_pomdp_pi(controller::JointController, prob)
    # Initialize
    it = 0
    epsilon = 0.01
    R_max = find_maximum_absolute_reward(prob)
    gamma = prob.discount_factor
    
    ctrlr_t = deepcopy(controller)
    n = length(ctrlr_t.controllers)
    
    # Initial evaluation
    V_prev, _ = evaluate_controller(ctrlr_t, prob)
    println("Initial controller value: $(V_prev)")
    improvement = 1000000
    V_curr = -100000
    # Main policy iteration loop
    while it < 30 && improvement > 0.01 && V_curr < 0
        improved = false
        
        # [Backup and evaluate]
        for i in 1:n
            println("Backing up agent $i...")
            new_controller = improved_exhaustive_backup(
                ctrlr_t.controllers[i],
                ctrlr_t,
                i,
                prob
            )
            
            # Update the controller
            ctrlr_t.controllers[i] = new_controller
        end
        
        # Evaluate the joint controller
        V_curr, _ = evaluate_controller(ctrlr_t, prob)
        println("After backup, controller value: $(V_curr)")
        
        # Check for improvement
        # # [Prune dominated nodes until none can be removed]
        # pruned_any = true
        # while pruned_any
        #     pruned_any = false
            
        #     for i in 1:n
        #         # Create a list of other controllers
        #         other_controllers = [ctrlr_t.controllers[j] for j in 1:n if j != i]
                
        #         # Try to prune agent i's controller
        #         new_controller, was_pruned = prune_controller(i, ctrlr_t.controllers[i], other_controllers, prob)
                
        #         if was_pruned
        #             ctrlr_t.controllers[i] = new_controller
        #             pruned_any = true
        #             println("Pruned agent $(i)'s controller to $(length(new_controller.nodes)) nodes")
        #         end
        #     end
            
        #     if pruned_any
        #         # Re-evaluate after pruning
        #         V_curr, _ = evaluate_controller(ctrlr_t, prob)
        #         println("After pruning, controller value: $(V_curr)")
        #     end
        # end

        if V_curr > V_prev
            V_prev = V_curr
            improved = true
            println("Iteration $it improved value to: $V_curr")
        end
        
        # Calculate improvement
        improvement = abs(V_curr - V_prev)
        
        it += 1
        println("Completed iteration $(it)")
    end
    
    return it, ctrlr_t
end

# Helper function to find the maximum absolute reward
function find_maximum_absolute_reward(prob::DecTigerPOMDP)
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
function improved_exhaustive_backup(controller::AgentController, joint_controller::JointController, agent_idx::Int, prob::DecTigerPOMDP)
    # First, keep track of the original controller
    original_controller = deepcopy(controller)
    
    # Get the other agent's controller
    other_agent_idx = agent_idx == 1 ? 2 : 1
    other_controller = joint_controller.controllers[other_agent_idx]
    
    # Create the initial set of candidate nodes (same as your original approach)
    agent_actions = ["listen", "open-left", "open-right"]
    agent_observations = ["hear-left", "hear-right"]
    
    current_nodes = length(controller.nodes)
    candidate_nodes = Vector{FSCNode}()
    
    # Generate all possible new nodes (similar to your original approach)
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
    # This is the key improvement step
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

function test_controller(controller::JointController, problem::DecTigerPOMDP; 
                        num_episodes=1000, max_steps=20)
    total_reward = 0.0
    
    for episode in 1:num_episodes
        # Reset the environment
        current_state = rand(["tiger-left", "tiger-right"])
        
        # Initialize controller nodes (start at first node for each agent)
        current_nodes = [1, 1]
        
        episode_reward = 0.0
        
        # Run one episode
        for step in 1:max_steps
            # Get actions from the controller nodes
            actions = []
            for agent_idx in 1:2
                agent_controller = controller.controllers[agent_idx]
                node = agent_controller.nodes[current_nodes[agent_idx]]
                
                if node.action == 1
                    push!(actions, "listen")
                elseif node.action == 2
                    push!(actions, "open-left")
                else
                    push!(actions, "open-right")
                end
            end
            
            # Execute actions and get reward
            step_reward = 0.0
            
            # Simplified Dec-Tiger reward logic
            if actions[1] == "listen" && actions[2] == "listen"
                step_reward = -2.0  # Both agents listen
            elseif actions[1] == "listen" || actions[2] == "listen"
                step_reward = -1.0  # One agent listens
            else
                # Both agents open doors
                if current_state == "tiger-left"
                    if actions[1] == "open-right" && actions[2] == "open-right"
                        step_reward = 20.0  # Both open correct door
                    elseif actions[1] == "open-left" && actions[2] == "open-left"
                        step_reward = -50.0  # Both open tiger door
                    else
                        step_reward = -15.0  # Mixed door opening
                    end
                else  # tiger-right
                    if actions[1] == "open-left" && actions[2] == "open-left"
                        step_reward = 20.0  # Both open correct door
                    elseif actions[1] == "open-right" && actions[2] == "open-right"
                        step_reward = -50.0  # Both open tiger door
                    else
                        step_reward = -15.0  # Mixed door opening
                    end
                end
                
                # Reset the state after door opening
                current_state = rand(["tiger-left", "tiger-right"])
            end
            
            episode_reward += step_reward
            
            # Generate observations
            observations = []
            for agent_idx in 1:2
                if actions[agent_idx] == "listen"
                    # Correct observation 85% of the time
                    if rand() < 0.85
                        push!(observations, current_state == "tiger-left" ? "hear-left" : "hear-right")
                    else
                        push!(observations, current_state == "tiger-left" ? "hear-right" : "hear-left")
                    end
                else
                    # If door was opened, observation doesn't matter (uniform random)
                    push!(observations, rand(["hear-left", "hear-right"]))
                end
            end
            
            # Update controller nodes based on observations
            for agent_idx in 1:2
                obs = observations[agent_idx]
                agent_controller = controller.controllers[agent_idx]
                next_node = agent_controller.nodes[current_nodes[agent_idx]].transitions[obs]
                current_nodes[agent_idx] = next_node
            end
        end
        
        total_reward += episode_reward
    end
    
    average_reward = total_reward / num_episodes
    
    println("Controller with [$(length(controller.controllers[1].nodes)), $(length(controller.controllers[2].nodes))] nodes")
    println("Average reward over $(num_episodes) episodes: $(average_reward)")
    
    return average_reward
end

# Simple function to compare initial and improved controllers
function compare_controllers()
    dec_tiger = DecTigerPOMDP()
    
    # Create initial controller
    initial_controller = create_initial_controller(dec_tiger)
    
    # Test initial controller
    println("\nTesting initial controller:")
    initial_reward = test_controller(initial_controller, dec_tiger)
    
    # Run policy iteration
    println("\nRunning policy iteration...")
    _, improved_controller = dec_pomdp_pi(initial_controller, dec_tiger)
    
    # Test improved controller
    println("\nTesting improved controller:")
    improved_reward = test_controller(improved_controller, dec_tiger)
    
    # Print comparison
    println("\nResults comparison:")
    println("Initial controller reward: $(initial_reward)")
    println("Improved controller reward: $(improved_reward)")
    println("Improvement: $(improved_reward - initial_reward)")
    
    return improved_controller
end

# Run the comparison
# compare_controllers()

# Run the test
# test_result = test_policy_iteration()

# @show dec_tiger = DecTigerPOMDP()
# @show states(dec_tiger)

println("running")

ctrl = create_heuristic_controller(dec_tiger)
dec_pomdp_pi(ctrl, dec_tiger)


# prob = DecTigerPOMDP()
# best_controller, best_value = test_multiple_controllers(prob)
# println("Best controller found with value: $best_value")