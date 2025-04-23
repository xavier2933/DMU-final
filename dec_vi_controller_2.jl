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


function create_initial_controller(problem::DecTigerPOMDP)
    # Create a simple initial controller for each agent
    controllers = []
    
    # Get actions and observations from your problem
    agent_actions = ["listen", "open-left", "open-right"]
    agent_observations = ["hear-left", "hear-right"]
    
    # We need two controllers (one for each agent)
    for i in 1:2
        # Create a single node that always listens (the safest initial action)
        # Using index 1 for "listen" based on your action ordering
        node = FSCNode(1, Dict{String, Int}())
        
        # For each observation, transition back to this node
        for obs in agent_observations
            node.transitions[obs] = 1
        end
        
        push!(controllers, AgentController([node]))
    end
    
    return JointController(controllers)
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
            println("Value iteration converged after $(iter) iterations")
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
    epsilon = 0.01
    R_max = find_maximum_absolute_reward(prob)
    gamma = prob.discount_factor
    
    ctrlr_t = deepcopy(controller)
    n = length(ctrlr_t.controllers)
    
    # Initial evaluation
    V_prev, _ = evaluate_controller(ctrlr_t, prob)
    println("Initial controller value: $(V_prev)")
    
    # Main policy iteration loop
    while it < 10
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
        if V_curr > V_prev
            V_prev = V_curr
            improved = true
            println("Iteration $it improved value to: $V_curr")
        else
            println("No improvement in iteration $it")
            
            # Optional: Add randomization to escape local optima
            if it > 0 && !improved
                println("Trying random perturbation...")
                for i in 1:n
                    if length(ctrlr_t.controllers[i].nodes) > 0
                        # Randomly modify one node
                        rand_node_idx = rand(1:length(ctrlr_t.controllers[i].nodes))
                        old_node = ctrlr_t.controllers[i].nodes[rand_node_idx]
                        
                        # Create new node with random action
                        rand_action = rand(1:3)
                        new_transitions = Dict{String, Int}()
                        
                        # Copy transitions with random changes
                        for (obs, next) in old_node.transitions
                            if rand() < 0.3  # 30% chance to change
                                new_transitions[obs] = rand(1:length(ctrlr_t.controllers[i].nodes))
                            else
                                new_transitions[obs] = next
                            end
                        end
                        
                        # Create new node
                        new_node = FSCNode(rand_action, new_transitions)
                        
                        # Replace in controller
                        new_nodes = Vector{FSCNode}()
                        for j in 1:length(ctrlr_t.controllers[i].nodes)
                            if j == rand_node_idx
                                push!(new_nodes, new_node)
                            else
                                push!(new_nodes, ctrlr_t.controllers[i].nodes[j])
                            end
                        end
                        
                        ctrlr_t.controllers[i] = AgentController(new_nodes)
                    end
                end
                
                # Re-evaluate after randomization
                V_rand, _ = evaluate_controller(ctrlr_t, prob)
                println("After randomization, value: $V_rand")
                
                if V_rand > V_prev
                    V_prev = V_rand
                    improved = true
                    println("Randomization improved value to: $V_rand")
                else
                    println("No improvement from randomization, stopping.")
                    break
                end
            else
                break  # No improvement and no randomization applied, so stop
            end
        end
        
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

# ctrl = create_initial_controller(dec_tiger)
# dec_pomdp_pi(ctrl, dec_tiger)

function test_backup()
    # Create a simple Dec-Tiger POMDP
    prob = DecTigerPOMDP()
    
    # Create controllers for two agents
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2  # Two agents
        # Create a single node - always performing "listen" action
        node = FSCNode(
            1,  # Action 1 is "listen"
            Dict("hear-left" => 1, "hear-right" => 1)  # Always go back to node 1
        )
        
        # Create an agent controller with the single node
        ctrl = AgentController([node])
        push!(agent_controllers, ctrl)
    end
    
    # Create the joint controller
    joint_ctrl = JointController(agent_controllers)
    
    # Evaluate the initial controller
    initial_value, _ = evaluate_controller(joint_ctrl, prob)
    println("Initial controller value: $(initial_value)")
    println("Initial controller configuration:")
    print_controller(joint_ctrl)
    
    # Test backup on the first agent's controller
    println("\nTesting backup on agent 1...")
    new_ctrl1 = exhaustive_backup(joint_ctrl.controllers[1])
    
    # Create a new joint controller with the backed-up controller
    new_joint_ctrl = JointController([new_ctrl1, joint_ctrl.controllers[2]])
    
    # Evaluate the new controller
    new_value, _ = evaluate_controller(new_joint_ctrl, prob)
    println("\nAfter backing up agent 1:")
    println("New controller value: $(new_value)")
    println("New controller configuration:")
    print_controller(new_joint_ctrl)
    
    # Did the value improve?
    if new_value > initial_value
        println("\nSuccess! The backup improved the value.")
    else
        println("\nWarning: Backup did not improve the value.")
    end
    
    # Test backup on both agents
    println("\nTesting backup on both agents...")
    new_ctrl1 = exhaustive_backup(joint_ctrl.controllers[1])
    new_ctrl2 = exhaustive_backup(joint_ctrl.controllers[2])
    
    new_joint_ctrl = JointController([new_ctrl1, new_ctrl2])
    
    # Evaluate the new controller
    new_value, _ = evaluate_controller(new_joint_ctrl, prob)
    println("\nAfter backing up both agents:")
    println("New controller value: $(new_value)")
    println("New controller configuration:")
    print_controller(new_joint_ctrl)
    
    # Did the value improve?
    if new_value > initial_value
        println("\nSuccess! The backup improved the value.")
    else
        println("\nWarning: Backup did not improve the value.")
    end
end

# Helper function to print controller configuration
function print_controller(joint_controller)
    for i in 1:length(joint_controller.controllers)
        println("Agent $(i):")
        for j in 1:length(joint_controller.controllers[i].nodes)
            node = joint_controller.controllers[i].nodes[j]
            println("  Node $(j):")
            println("    Action: $(node.action)")
            println("    Transitions:")
            for (obs, next_node) in node.transitions
                println("      On $(obs) -> Node $(next_node)")
            end
        end
    end
end

# test_backup()

function test_multiple_controllers(prob::DecTigerPOMDP)
    # Create several different initial controllers to test
    controllers = []
    values = []
    
    # 1. Listen-only controller (both agents always listen)
    joint_ctrl1 = create_listen_only_controller()
    push!(controllers, joint_ctrl1)
    
    # 2. Tiger-avoidance controller (listen, then open the door opposite to what you hear)
    joint_ctrl2 = create_tiger_avoidance_controller()
    push!(controllers, joint_ctrl2)
    
    # 3. Random controller
    joint_ctrl3 = create_random_controller(2, 3)  # 2 nodes, 3 actions
    push!(controllers, joint_ctrl3)
    
    # 4. Coordinated door opening controller
    joint_ctrl4 = create_coordinated_controller()
    push!(controllers, joint_ctrl4)
    
    # Evaluate each controller
    for (i, ctrl) in enumerate(controllers)
        value, _ = evaluate_controller(ctrl, prob)
        push!(values, value)
        println("Controller $i value: $value")
        println("Configuration:")
        print_controller(ctrl)
        println("\n")
    end
    
    # Return the best controller
    best_idx = argmax(values)
    return controllers[best_idx], values[best_idx]
end

# Helper functions to create different controllers
function create_listen_only_controller()
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2
        # Create a single node that always listens
        node = FSCNode(
            1,  # Action 1 is "listen"
            Dict("hear-left" => 1, "hear-right" => 1)  # Always go back to node 1
        )
        
        ctrl = AgentController([node])
        push!(agent_controllers, ctrl)
    end
    
    return JointController(agent_controllers)
end

function create_tiger_avoidance_controller()
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2
        # Create a controller with 2 nodes
        # Node 1: Listen and transition based on what it hears
        node1 = FSCNode(
            1,  # Listen
            Dict("hear-left" => 2, "hear-right" => 3)  # Go to node 2 if hear-left, node 3 if hear-right
        )
        
        # Node 2: Open right door (assuming tiger is on left)
        node2 = FSCNode(
            3,  # Open right
            Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
        )
        
        # Node 3: Open left door (assuming tiger is on right)
        node3 = FSCNode(
            2,  # Open left
            Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
        )
        
        ctrl = AgentController([node1, node2, node3])
        push!(agent_controllers, ctrl)
    end
    
    return JointController(agent_controllers)
end

function create_random_controller(num_nodes, num_actions)
    agent_controllers = Vector{AgentController}()
    
    for i in 1:2
        nodes = []
        for j in 1:num_nodes
            action = rand(1:num_actions)
            transitions = Dict(
                "hear-left" => rand(1:num_nodes),
                "hear-right" => rand(1:num_nodes)
            )
            push!(nodes, FSCNode(action, transitions))
        end
        
        ctrl = AgentController(nodes)
        push!(agent_controllers, ctrl)
    end
    
    return JointController(agent_controllers)
end

function create_coordinated_controller()
    # More sophisticated controller where agents coordinate
    # Agent 1 listens and then signals with its action
    # Agent 2 responds to agent 1's signal
    
    # Agent 1 controller
    agent1_nodes = []
    
    # Node 1: Listen
    node1_1 = FSCNode(
        1,  # Listen
        Dict("hear-left" => 2, "hear-right" => 3)  # Go to node 2 if hear-left, node 3 if hear-right
    )
    
    # Node 2: Open left as a signal (tiger on left)
    node1_2 = FSCNode(
        2,  # Open left (signal)
        Dict("hear-left" => 4, "hear-right" => 4)  # Go to node 4 after signaling
    )
    
    # Node 3: Open right as a signal (tiger on right)
    node1_3 = FSCNode(
        3,  # Open right (signal)
        Dict("hear-left" => 5, "hear-right" => 5)  # Go to node 5 after signaling
    )
    
    # Node 4: Open right (after signaling tiger on left)
    node1_4 = FSCNode(
        3,  # Open right
        Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
    )
    
    # Node 5: Open left (after signaling tiger on right)
    node1_5 = FSCNode(
        2,  # Open left
        Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
    )
    
    push!(agent1_nodes, node1_1, node1_2, node1_3, node1_4, node1_5)
    agent1_ctrl = AgentController(agent1_nodes)
    
    # Agent 2 controller - more responsive to agent 1's signals
    agent2_nodes = []
    
    # Node 1: Listen
    node2_1 = FSCNode(
        1,  # Listen
        Dict("hear-left" => 1, "hear-right" => 1)  # Keep listening until agent 1 signals
    )
    
    # Node 2: Open right (responding to agent 1's left signal)
    node2_2 = FSCNode(
        3,  # Open right
        Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
    )
    
    # Node 3: Open left (responding to agent 1's right signal)
    node2_3 = FSCNode(
        2,  # Open left
        Dict("hear-left" => 1, "hear-right" => 1)  # Back to listening
    )
    
    push!(agent2_nodes, node2_1, node2_2, node2_3)
    agent2_ctrl = AgentController(agent2_nodes)
    
    return JointController([agent1_ctrl, agent2_ctrl])
end

prob = DecTigerPOMDP()
best_controller, best_value = test_multiple_controllers(prob)
println("Best controller found with value: $best_value")