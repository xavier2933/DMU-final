using Pkg
using Statistics
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


function create_heuristic_controller(prob::POMDP)
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

function evaluate_controller(joint_controller::JointController, problem::POMDP)

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
                action1 = actions[joint_action[1]]
                action2 = actions[joint_action[2]]
                joint_action_strings = (action1, action2)
                
                # Now call the reward function with the correct types
                immediate_reward = POMDPs.reward(problem, state, joint_action_strings)
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
            println("Backing up agent $i...")
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
        println("After backup, controller value: $(V_curr)")
        
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

# Helper function to find the maximum absolute reward
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


println("running")

ctrl = create_heuristic_controller(dec_tiger)
_, controller = dec_pomdp_pi(ctrl, dec_tiger)

function verify_controller(joint_controller::JointController, prob::POMDP, num_episodes=10000, max_steps=20)
    total_reward = 0.0
    rewards_per_episode = []
    
    # Track statistics
    door_opened_correctly = 0
    door_opened_incorrectly = 0
    episodes_with_pos_reward = 0
    
    # For each episode
    for episode in 1:num_episodes
        # Initialize state randomly
        state = rand(["tiger-left", "tiger-right"])
        # Start at initial nodes
        nodes = [1, 1]  # Assuming first node is initial node
        episode_reward = 0.0
        
        # Run for max_steps or until door is opened
        for step in 1:max_steps
            # Get actions from current nodes
            actions = [joint_controller.controllers[i].nodes[nodes[i]].action 
                       for i in 1:2]
            
            # Convert action indices to names for readability
            action_names = ["listen", "open-left", "open-right"]
            action1 = action_names[actions[1]]
            action2 = action_names[actions[2]]
            
            joint_action_strings = (action1, action2)

            # Now call the reward function with the correct types
            step_reward = POMDPs.reward(prob, state, joint_action_strings)
            episode_reward += step_reward
            
            # Check if doors were opened
            if action1 != "listen" || action2 != "listen"
                if step_reward > 0
                    door_opened_correctly += 1
                elseif step_reward < -10
                    door_opened_incorrectly += 1
                end
                
                # Reset state after door opening
                state = rand(["tiger-left", "tiger-right"])
            end
            
            # Generate observations
            observations = []
            for i in 1:2
                if actions[i] == 1  # Listen
                    # Correct observation with 85% probability
                    if state == "tiger-left"
                        obs = rand() < 0.85 ? "hear-left" : "hear-right"
                    else
                        obs = rand() < 0.85 ? "hear-right" : "hear-left"
                    end
                else
                    # After opening a door, observation is random
                    obs = rand(["hear-left", "hear-right"])
                end
                push!(observations, obs)
            end
            
            # Transition to next nodes
            for i in 1:2
                nodes[i] = joint_controller.controllers[i].nodes[nodes[i]].transitions[observations[i]]
            end
        end
        
        total_reward += episode_reward
        push!(rewards_per_episode, episode_reward)
        
        if episode_reward > 0
            episodes_with_pos_reward += 1
        end
    end
    
    # Calculate statistics
    avg_reward = total_reward / num_episodes
    std_dev = std(rewards_per_episode)
    success_rate = episodes_with_pos_reward / num_episodes
    
    println("=== Controller Verification Results ===")
    println("Average reward per episode: $avg_reward")
    println("Standard deviation: $std_dev")
    println("Episodes with positive reward: $(success_rate * 100)%")
    println("Correct door openings: $door_opened_correctly")
    println("Incorrect door openings: $door_opened_incorrectly")
    println("Correct opening ratio: $(door_opened_correctly / (door_opened_correctly + door_opened_incorrectly))")
    
    # Analyze controller structure
    println("\n=== Controller Structure Analysis ===")
    for i in 1:2
        println("Agent $i controller has $(length(joint_controller.controllers[i].nodes)) nodes")
        # Count action distribution
        action_counts = zeros(Int, 3)
        for node in joint_controller.controllers[i].nodes
            action_counts[node.action] += 1
        end
        
        for a in 1:3
            action_name = ["listen", "open-left", "open-right"][a]
            percentage = action_counts[a] / length(joint_controller.controllers[i].nodes) * 100
            println("  $action_name: $(round(percentage, digits=1))%")
        end
    end
    
    return avg_reward, success_rate
end

m = DecTigerPOMDP()
verify_controller(controller, m)
# prob = DecTigerPOMDP()
# best_controller, best_value = test_multiple_controllers(prob)
# println("Best controller found with value: $best_value")