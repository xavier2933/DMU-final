include("../Solvers/dec_pi_solver.jl")
include("../Problems/satellite_dec_pomdp.jl")


function verify_satellite_controller(joint_controller::JointController, prob::SatelliteNetworkPOMDP, num_episodes=1000, max_steps=50)
    total_reward = 0.0
    rewards_per_episode = []
    steps_to_transmit = []
    transmission_success_count = 0
    
    # Track statistics for each satellite
    satellite_transmission_attempts = zeros(Int, prob.num_satellites)
    satellite_transmission_successes = zeros(Int, prob.num_satellites)
    satellite_pass_attempts = zeros(Int, prob.num_satellites)
    satellite_pass_successes = zeros(Int, prob.num_satellites)
    
    # For each episode
    for episode in 1:num_episodes
        # Initialize state - data at satellite 1
        state = "data-at-1"
        
        # Start with first node for each satellite
        nodes = ones(Int, prob.num_satellites)
        episode_reward = 0.0
        step_count = 0
        
        # Run episode until max steps or transmission
        while step_count < max_steps && state != "data-transmitted"
            step_count += 1
            
            # Get data-holding satellite
            data_sat = parse(Int, last(split(state, "-")))
            
            # Get actions from current nodes
            action_indices = [joint_controller.controllers[i].nodes[nodes[i]].action 
                             for i in 1:prob.num_satellites]
            
            # Convert action indices to strings
            actions = []
            for i in 1:prob.num_satellites
                sat_actions = agent_actions(prob, i)
                push!(actions, sat_actions[action_indices[i]])
            end
            joint_action = Tuple(actions)
            
            # Get reward
            step_reward = POMDPs.reward(prob, state, joint_action)
            episode_reward += step_reward
            
            # Track action statistics
            data_sat_action = actions[data_sat]
            if data_sat_action == "transmit"
                satellite_transmission_attempts[data_sat] += 1
                
                # Check if transmission successful (based on probability)
                if rand() < prob.ground_tx_probs[data_sat]
                    satellite_transmission_successes[data_sat] += 1
                    state = "data-transmitted"
                end
            elseif data_sat_action == "pass-left" && data_sat > 1
                satellite_pass_attempts[data_sat] += 1
                
                # Check if pass successful
                if rand() < prob.pass_success_prob
                    satellite_pass_successes[data_sat] += 1
                    state = "data-at-$(data_sat-1)"
                end
            elseif data_sat_action == "pass-right" && data_sat < prob.num_satellites
                satellite_pass_attempts[data_sat] += 1
                
                # Check if pass successful
                if rand() < prob.pass_success_prob
                    satellite_pass_successes[data_sat] += 1
                    state = "data-at-$(data_sat+1)"
                end
            end
            
            # Generate observations for each satellite
            observations = ["no-data" for _ in 1:prob.num_satellites]
            if state != "data-transmitted"
                next_data_sat = parse(Int, last(split(state, "-")))
                observations[next_data_sat] = "has-data"
            end
            
            # Update node for each satellite
            for i in 1:prob.num_satellites
                nodes[i] = joint_controller.controllers[i].nodes[nodes[i]].transitions[observations[i]]
            end
        end
        
        # Record episode statistics
        push!(rewards_per_episode, episode_reward)
        if state == "data-transmitted"
            transmission_success_count += 1
            push!(steps_to_transmit, step_count)
        end
        
        total_reward += episode_reward
    end
    
    # Calculate overall statistics
    avg_reward = total_reward / num_episodes
    std_dev = std(rewards_per_episode)
    success_rate = transmission_success_count / num_episodes
    avg_steps = isempty(steps_to_transmit) ? NaN : mean(steps_to_transmit)
    
    println("=== Satellite Network Controller Verification Results ===")
    println("Average reward per episode: $avg_reward")
    println("Standard deviation: $std_dev")
    println("Transmission success rate: $(success_rate * 100)%")
    println("Average steps to successful transmission: $avg_steps")
    
    println("\n=== Satellite Transmission Statistics ===")
    for i in 1:prob.num_satellites
        tx_success_rate = satellite_transmission_attempts[i] > 0 ? 
            satellite_transmission_successes[i] / satellite_transmission_attempts[i] : 0.0
        pass_success_rate = satellite_pass_attempts[i] > 0 ? 
            satellite_pass_successes[i] / satellite_pass_attempts[i] : 0.0
        
        println("Satellite $i (Ground TX prob: $(prob.ground_tx_probs[i])):")
        println("  Transmission attempts: $(satellite_transmission_attempts[i])")
        println("  Transmission successes: $(satellite_transmission_successes[i]) ($(tx_success_rate*100)%)")
        println("  Pass attempts: $(satellite_pass_attempts[i])")
        println("  Pass successes: $(satellite_pass_successes[i]) ($(pass_success_rate*100)%)")
    end
    
    println("\n=== Controller Structure Analysis ===")
    for i in 1:prob.num_satellites
        println("Satellite $i controller has $(length(joint_controller.controllers[i].nodes)) nodes")
        
        # Analyze node actions
        for (j, node) in enumerate(joint_controller.controllers[i].nodes)
            sat_actions = agent_actions(prob, i)
            action_name = sat_actions[node.action]
            println("  Node $j: Action = $action_name")
            println("    Transitions: $(node.transitions)")
        end
    end
    
    return avg_reward, success_rate
end

# Create the problem
sat_network = SatelliteNetworkPOMDP(
    num_satellites = 3,
    ground_tx_probs = [0.3, 0.5, 0.6],
    discount_factor = 0.9,
    pass_success_prob = 0.95,
    successful_tx_reward = 20.0,
    unsuccessful_tx_penalty = -5.0,
    pass_cost = -1.0
)

# Create initial controller
initial_ctrl = create_initial_controllers(sat_network)

# Run policy iteration
iterations, final_controller = dec_pomdp_pi(initial_ctrl, sat_network)

# Verify the controller
avg_reward, success_rate = verify_satellite_controller(final_controller, sat_network)

println("Policy iteration completed in $iterations iterations.")
println("Final controller average reward: $avg_reward")
println("Successful transmission rate: $(success_rate * 100)%")

