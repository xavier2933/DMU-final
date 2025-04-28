using Pkg
using Statistics

# Generic Dec-POMDP Interface
abstract type DecPOMDP end

# Define the FSC node structure (already generic)
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

# Problem-specific functions that need to be defined for each problem
function states(prob::DecPOMDP) end
function actions(prob::DecPOMDP) end
function observations(prob::DecPOMDP) end
function num_agents(prob::DecPOMDP) end
function transition_probability(prob::DecPOMDP, state, next_state, joint_action) end
function observation_probability(prob::DecPOMDP, next_state, joint_action, joint_obs) end
function reward(prob::DecPOMDP, state, joint_action) end
function initial_belief(prob::DecPOMDP) end
function discount_factor(prob::DecPOMDP) end

# Generic controller evaluation function
function evaluate_controller(joint_controller::JointController, problem::DecPOMDP)
    state_list = states(problem)
    
    # Get the number of states and number of nodes for each agent
    num_states = length(state_list)
    agents = num_agents(problem)
    nodes_per_agent = [length(c.nodes) for c in joint_controller.controllers]
    
    # Create value function matrix
    dims = [nodes_per_agent..., num_states]
    V = zeros(dims...)
    
    # Get discount factor
    gamma = discount_factor(problem)
    
    # Value iteration to compute the value function
    max_iterations = 1000
    epsilon = 0.001
    
    for iter in 1:max_iterations
        # Make a copy of V for updating
        V_new = copy(V)
        
        # For each possible joint node configuration and state
        for node_indices in Iterators.product([1:n for n in nodes_per_agent]...)
            for s in 1:num_states
                # Get the current state
                state = state_list[s]
                
                # Joint action from current nodes
                joint_action = [joint_controller.controllers[i].nodes[node_indices[i]].action 
                               for i in 1:agents]
                
                # Compute immediate reward
                immediate_reward = reward(problem, state, joint_action)
                
                # Expected future reward
                future_reward = 0.0
                
                # For each possible next state
                for next_s in 1:num_states
                    next_state = state_list[next_s]
                    
                    # Transition probability
                    trans_prob = transition_probability(problem, state, next_state, joint_action)
                    
                    if trans_prob > 0  # Only consider non-zero transitions for efficiency
                        # Get observations for each agent
                        agent_observations = observations(problem)
                        
                        # For all possible combinations of observations
                        for obs_combos in Iterators.product([agent_observations[i] for i in 1:agents]...)
                            # Create joint observation
                            joint_obs = collect(obs_combos)
                            
                            # Observation probability
                            obs_prob = observation_probability(problem, next_state, joint_action, joint_obs)
                            
                            if obs_prob > 0  # Only consider non-zero observations
                                # Next nodes for each agent based on their observation
                                next_nodes = []
                                for i in 1:agents
                                    # Get the next node index from the transition function
                                    current_node = joint_controller.controllers[i].nodes[node_indices[i]]
                                    next_node = current_node.transitions[joint_obs[i]]
                                    push!(next_nodes, next_node)
                                end
                                
                                # Look up future value
                                future_value = V[next_nodes..., next_s]
                                
                                # Add weighted contribution
                                future_reward += trans_prob * obs_prob * future_value
                            end
                        end
                    end
                end
                
                # Total value for this configuration
                V_new[node_indices..., s] = immediate_reward + gamma * future_reward
            end
        end
        
        # Check convergence
        if maximum(abs.(V_new - V)) < epsilon
            break
        end
        
        # Update value function
        V = copy(V_new)
        
        if iter == max_iterations
            @warn "Value iteration did not converge after $(max_iterations) iterations"
        end
    end
    
    # Compute the value for the initial belief state and initial controller nodes
    belief = initial_belief(problem)
    initial_nodes = [1 for _ in 1:agents]  # Start at first node for each agent
    
    value = sum(belief[s] * V[initial_nodes..., s] for s in 1:num_states)
    
    return value, V
end


# Generic policy iteration function
function dec_pomdp_pi(controller::JointController, prob::DecPOMDP; max_iterations=30, epsilon=0.01)
    # Initialize
    it = 0
    R_max = find_maximum_absolute_reward(prob)
    gamma = discount_factor(prob)
    
    ctrlr_t = deepcopy(controller)
    n = num_agents(prob)
    
    # Initial evaluation
    V_prev, _ = evaluate_controller(ctrlr_t, prob)
    println("Initial controller value: $(V_prev)")
    
    # Set initial values
    V_curr = V_prev
    improvement = Inf
    
    # Main policy iteration loop
    while it < max_iterations && improvement > epsilon
        # Backup and evaluate for each agent
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
        
        # Calculate improvement
        improvement = abs(V_curr - V_prev)
        println("Improvement: $(improvement)")
        
        # Update previous value
        V_prev = V_curr
        
        it += 1
        println("Completed iteration $(it)")
        
        # Additional stopping condition
        if (gamma^it * R_max) < epsilon
            println("Theoretical bound reached, algorithm converged.")
            break
        end
    end
    
    # Report reason for stopping
    if it >= max_iterations
        println("Stopped due to maximum iterations reached.")
    elseif improvement <= epsilon
        println("Stopped due to convergence (improvement below threshold).")
    end
    
    return it, ctrlr_t
end

# Helper function to find the maximum absolute reward (generic)
function find_maximum_absolute_reward(prob::DecPOMDP)
    max_abs_reward = 0.0
    
    # Iterate through all states
    for s in states(prob)
        # Iterate through all possible joint actions
        for a_indices in Iterators.product([1:length(actions(prob)[i]) for i in 1:num_agents(prob)]...)
            joint_action = [a_indices[i] for i in 1:num_agents(prob)]
            
            # Calculate reward for this state-action pair
            r = reward(prob, s, joint_action)
            
            # Update maximum if this absolute reward is higher
            if abs(r) > max_abs_reward
                max_abs_reward = abs(r)
            end
        end
    end
    
    return max_abs_reward
end

# Generic improved exhaustive backup function
function improved_exhaustive_backup(controller::AgentController, joint_controller::JointController, agent_idx::Int, prob::DecPOMDP)
    # Keep track of the original controller
    original_controller = deepcopy(controller)
    
    # Get action and observation spaces for this agent
    agent_actions = actions(prob)[agent_idx]
    agent_observations = observations(prob)[agent_idx]
    
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
            
            # Add this candidate node
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
                println("Found improvement by replacing node $node_idx, new value: $temp_value")
            end
        end
    end
    
    # Try adding a new node (only if it helps and controller isn't too large)
    if !improved && length(controller.nodes) < 10
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

# Generic controller verification function
function verify_controller(joint_controller::JointController, prob::DecPOMDP; num_episodes=10000, max_steps=20)
    total_reward = 0.0
    rewards_per_episode = []
    
    # Get problem details
    state_list = states(prob)
    num_agents = length(joint_controller.controllers)
    action_lists = actions(prob)
    obs_lists = observations(prob)
    
    # For each episode
    for episode in 1:num_episodes
        # Initialize state randomly
        state = rand(state_list)
        # Start at initial nodes
        nodes = [1 for _ in 1:num_agents]  # Assuming first node is initial
        episode_reward = 0.0
        
        # Run for max_steps
        for step in 1:max_steps
            # Get actions from current nodes
            joint_action = [joint_controller.controllers[i].nodes[nodes[i]].action 
                          for i in 1:num_agents]
            
            # Get reward
            step_reward = reward(prob, state, joint_action)
            episode_reward += step_reward
            
            # Determine next state (transition)
            next_state_probs = [transition_probability(prob, state, next_s, joint_action) 
                               for next_s in state_list]
            # Normalize probabilities if needed
            if sum(next_state_probs) > 0
                next_state_probs = next_state_probs ./ sum(next_state_probs)
                state = sample_from_distribution(state_list, next_state_probs)
            else
                state = rand(state_list)  # Fallback to random state
            end
            
            # Generate observations for each agent
            observations = []
            for i in 1:num_agents
                obs_probs = [observation_probability(prob, state, joint_action, 
                             [j == i ? o : "dummy" for j in 1:num_agents]) 
                             for o in obs_lists[i]]
                
                if sum(obs_probs) > 0
                    obs_probs = obs_probs ./ sum(obs_probs)
                    obs = sample_from_distribution(obs_lists[i], obs_probs)
                else
                    obs = rand(obs_lists[i])  # Fallback to random observation
                end
                
                push!(observations, obs)
            end
            
            # Transition to next nodes
            for i in 1:num_agents
                nodes[i] = joint_controller.controllers[i].nodes[nodes[i]].transitions[observations[i]]
            end
        end
        
        total_reward += episode_reward
        push!(rewards_per_episode, episode_reward)
    end
    
    # Calculate statistics
    avg_reward = total_reward / num_episodes
    std_dev = std(rewards_per_episode)
    
    println("=== Controller Verification Results ===")
    println("Average reward per episode: $avg_reward")
    println("Standard deviation: $std_dev")
    
    # Analyze controller structure
    println("\n=== Controller Structure Analysis ===")
    for i in 1:num_agents
        println("Agent $i controller has $(length(joint_controller.controllers[i].nodes)) nodes")
        # Count action distribution
        action_counts = zeros(Int, length(action_lists[i]))
        for node in joint_controller.controllers[i].nodes
            action_counts[node.action] += 1
        end
        
        for a in 1:length(action_lists[i])
            action_name = action_lists[i][a]
            percentage = action_counts[a] / length(joint_controller.controllers[i].nodes) * 100
            println("  $action_name: $(round(percentage, digits=1))%")
        end
    end
    
    return avg_reward
end

# Helper function for sampling from a distribution
function sample_from_distribution(items, probs)
    r = rand()
    cumulative = 0.0
    for i in 1:length(items)
        cumulative += probs[i]
        if r <= cumulative
            return items[i]
        end
    end
    return items[end]  # Fallback
end

# Create a generic function to build initial FSC controllers
function create_initial_controller(prob::DecPOMDP; nodes_per_agent=5)
    agent_controllers = Vector{AgentController}()
    
    for i in 1:num_agents(prob)
        agent_actions = actions(prob)[i]
        agent_observations = observations(prob)[i]
        
        # Create nodes for this agent
        nodes = Vector{FSCNode}()
        
        for n in 1:nodes_per_agent
            # Create a random action and transitions
            action = rand(1:length(agent_actions))
            transitions = Dict{String, Int}()
            
            for obs in agent_observations
                # Transition to a random node
                transitions[obs] = rand(1:nodes_per_agent)
            end
            
            push!(nodes, FSCNode(action, transitions))
        end
        
        push!(agent_controllers, AgentController(nodes))
    end
    
    return JointController(agent_controllers)
end


