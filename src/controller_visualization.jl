using CSV
using DataFrames
using Plots
using Statistics
using Random

"""
    run_and_record_controller(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP; 
                             num_episodes=10, max_steps=50, output_path="controller_data.csv")

Runs the controller for multiple episodes and records the state of the system at each step.
Saves the data to a CSV file for later visualization.

Returns a DataFrame containing the recorded data.
"""
function run_and_record_controller(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP; 
                                  num_episodes=10, max_steps=50, output_path="controller_data.csv")
    
    # Initialize empty arrays for each column
    episodes = Int[]
    steps = Int[]
    states = String[]
    joint_actions = String[]
    rewards = Float64[]
    
    # For each satellite, create arrays for packet counts, actions, and nodes
    sat_packets = [Int[] for _ in 1:prob.num_satellites]
    sat_actions = [String[] for _ in 1:prob.num_satellites]
    sat_nodes = [Int[] for _ in 1:prob.num_satellites]
    
    # Arrays for transmitted packets and total packets
    transmitted_packets = Int[]
    total_packets_arr = Int[]
    
    # For each episode
    for episode in 1:num_episodes
        # Initialize state: all packets at first satellite
        initial_distribution = zeros(Int, prob.num_satellites)
        initial_distribution[1] = prob.total_packets
        state = join(initial_distribution, "-")
        
        # Start with first node for each satellite
        nodes = ones(Int, prob.num_satellites)
        step_count = 0
        episode_packets_transmitted = 0
        total_packets_val = prob.total_packets
        
        # Run episode until max steps or all data transmitted
        while step_count < max_steps && state != "all-transmitted"
            step_count += 1
            
            # Get current packet distribution
            packet_counts = parse_state(state)
            
            # Extend packet_counts if it's shorter than num_satellites
            if length(packet_counts) < prob.num_satellites
                append!(packet_counts, zeros(Int, prob.num_satellites - length(packet_counts)))
            end
            
            # Get actions from current nodes
            action_indices = [joint_controller.controllers[i].nodes[nodes[i]].action 
                             for i in 1:prob.num_satellites]
            
            # Convert action indices to strings
            actions = String[]
            for i in 1:prob.num_satellites
                sat_actions_list = agent_actions(prob, i)
                
                # Handle out-of-bounds indices
                if action_indices[i] <= length(sat_actions_list)
                    push!(actions, sat_actions_list[action_indices[i]])
                else
                    push!(actions, "wait")  # Default to wait if invalid index
                end
            end
            joint_action = Tuple(actions)
            
            # Get reward
            step_reward = POMDPs.reward(prob, state, joint_action)
            
            # Record current state
            push!(episodes, episode)
            push!(steps, step_count)
            push!(states, state)
            push!(joint_actions, string(joint_action))
            push!(rewards, step_reward)
            
            # Record satellite-specific information
            for i in 1:prob.num_satellites
                if i <= length(packet_counts)
                    push!(sat_packets[i], packet_counts[i])
                else
                    push!(sat_packets[i], 0)
                end
                
                if i <= length(actions)
                    push!(sat_actions[i], actions[i])
                else
                    push!(sat_actions[i], "wait")
                end
                
                push!(sat_nodes[i], nodes[i])
            end
            
            # Record transmitted and total packets
            push!(transmitted_packets, episode_packets_transmitted)
            push!(total_packets_arr, total_packets_val)
            
            # Process actions and generate next state
            new_packet_counts = copy(packet_counts)
            
            # Process transmissions
            for sat_idx in 1:prob.num_satellites
                if sat_idx <= length(actions)
                    action = actions[sat_idx]
                    
                    if startswith(action, "transmit-")
                        num_packets = parse(Int, split(action, "-")[2])
                        actual_packets = min(num_packets, packet_counts[sat_idx])
                        
                        if actual_packets > 0
                            # Simulate each packet transmission
                            for _ in 1:actual_packets
                                if rand() < prob.ground_tx_probs[sat_idx]
                                    # Successful transmission
                                    new_packet_counts[sat_idx] -= 1
                                    episode_packets_transmitted += 1
                                end
                            end
                        end
                    elseif startswith(action, "pass-")
                        parts = split(action, "-")
                        direction = parts[2]
                        num_packets = parse(Int, parts[3])
                        actual_packets = min(num_packets, packet_counts[sat_idx])
                        
                        if actual_packets > 0
                            # Determine target satellite
                            target_idx = -1
                            if direction == "left" && sat_idx > 1
                                target_idx = sat_idx - 1
                            elseif direction == "right" && sat_idx < prob.num_satellites
                                target_idx = sat_idx + 1
                            else
                                continue  # Invalid direction
                            end
                            
                            # Simulate each packet pass
                            for _ in 1:actual_packets
                                if rand() < prob.pass_success_prob
                                    # Successful pass
                                    new_packet_counts[sat_idx] -= 1
                                    new_packet_counts[target_idx] += 1
                                end
                            end
                        end
                    end
                end
            end
            
            # Check if all packets have been transmitted
            if sum(new_packet_counts) == 0
                state = "all-transmitted"
            else
                state = join(new_packet_counts, "-")
            end
            
            # Generate observations for each satellite
            observations = []
            
            if state == "all-transmitted"
                # All satellites observe empty network
                empty_obs = join(zeros(Int, prob.num_satellites), "-")
                observations = [empty_obs for _ in 1:prob.num_satellites]
            else
                # Each satellite observes with some noise
                for i in 1:prob.num_satellites
                    if rand() < prob.observation_accuracy
                        # Accurate observation
                        push!(observations, state)
                    else
                        # Noisy observation
                        noisy_counts = copy(new_packet_counts)
                        
                        # Add noise to counts
                        for j in 1:length(noisy_counts)
                            noise = rand(-1:1)
                            noisy_counts[j] = max(0, noisy_counts[j] + noise)
                        end
                        
                        push!(observations, join(noisy_counts, "-"))
                    end
                end
            end
            
            # Update node for each satellite
            for i in 1:prob.num_satellites
                current_node = joint_controller.controllers[i].nodes[nodes[i]]
                
                # Handle potentially missing observations in transitions
                if haskey(current_node.transitions, observations[i])
                    nodes[i] = current_node.transitions[observations[i]]
                else
                    # Default to staying in current node if observation not recognized
                    nodes[i] = nodes[i]
                end
            end
        end
        
        # Record terminal state if episode ended before max steps
        if step_count < max_steps && state == "all-transmitted"
            # All packets transmitted, create a final record
            push!(episodes, episode)
            push!(steps, step_count + 1)
            push!(states, "all-transmitted")
            
            final_actions = ["wait" for _ in 1:prob.num_satellites]
            push!(joint_actions, string(Tuple(final_actions)))
            
            push!(rewards, 0.0)
            
            # Record satellite-specific information for terminal state
            for i in 1:prob.num_satellites
                push!(sat_packets[i], 0)  # No packets in terminal state
                push!(sat_actions[i], "wait")  # Default action in terminal state
                push!(sat_nodes[i], nodes[i])  # Keep the current node
            end
            
            # Record transmitted and total packets
            push!(transmitted_packets, total_packets_val)  # All packets transmitted
            push!(total_packets_arr, total_packets_val)
        end
    end
    
    # Create the DataFrame
    data = DataFrame(
        episode = episodes,
        step = steps,
        state = states,
        action = joint_actions,
        reward = rewards
    )
    
    # Add satellite columns
    for i in 1:prob.num_satellites
        data[!, Symbol("sat$(i)_packets")] = sat_packets[i]
        data[!, Symbol("sat$(i)_action")] = sat_actions[i]
        data[!, Symbol("sat$(i)_node")] = sat_nodes[i]
    end
    
    # Add transmitted and total packets columns
    data[!, :transmitted_packets] = transmitted_packets
    data[!, :total_packets] = total_packets_arr
    
    # Save to CSV
    CSV.write(output_path, data)
    println("Controller data saved to $output_path")
    
    return data
end

"""
    visualize_controller_data(csv_path::String; episode=1, output_path=nothing)

Visualizes data flow through the satellite network for a specific episode.
Creates multiple plots showing packets per satellite over time, actions taken,
and overall network performance.
"""
function visualize_controller_data(csv_path::String; episode=1, output_path=nothing)
    # Load the data
    data = CSV.read(csv_path, DataFrame)
    
    # Filter for the specific episode
    episode_data = filter(row -> row.episode == episode, data)
    
    if isempty(episode_data)
        error("Episode $episode not found in the data")
    end
    
    # Get the number of satellites from the column names
    satellite_cols = filter(c -> startswith(String(c), "sat") && endswith(String(c), "_packets"), names(episode_data))
    num_satellites = length(satellite_cols)
    
    # Extract packet counts for each satellite
    packet_data = Matrix{Float64}(undef, nrow(episode_data), num_satellites)
    for (i, col) in enumerate(satellite_cols)
        packet_data[:, i] = episode_data[:, col]
    end
    
    # Calculate transmitted packets at each step
    transmitted = episode_data.transmitted_packets
    
    # Create plots
    
    # 1. Packet distribution over time
    p1 = plot(
        title="Packet Distribution Over Time (Episode $episode)",
        xlabel="Time Step",
        ylabel="Number of Packets",
        legend=:right,
        size=(800, 500)
    )
    
    # Plot packets per satellite
    for i in 1:num_satellites
        plot!(p1, episode_data.step, packet_data[:, i], 
              label="Satellite $i", marker=:circle, markersize=4, linewidth=2)
    end
    
    # Plot transmitted packets
    plot!(p1, episode_data.step, transmitted, 
          label="Transmitted", marker=:star, markersize=6, linewidth=2, linestyle=:dash)
    
    # 2. Actions taken by each satellite
    p2 = plot(
        title="Satellite Actions Over Time (Episode $episode)",
        xlabel="Time Step",
        ylabel="Action",
        legend=:right,
        size=(800, 500)
    )
    
    # Create a categorical mapping for actions to y-values
    # This function now uses String as input and handles any string type
    function action_to_y(action_str)
        action = String(action_str)  # Convert to standard String
        if action == "wait"
            return 0
        elseif startswith(action, "transmit")
            return 1
        elseif startswith(action, "pass-left")
            return -1
        elseif startswith(action, "pass-right")
            return 2
        else
            return 0
        end
    end
    
    # Plot actions for each satellite
    for i in 1:num_satellites
        action_col = Symbol("sat$(i)_action")
        y_values = Float64[]
        
        # Process each action individually with error handling
        for action in episode_data[:, action_col]
            try
                push!(y_values, action_to_y(action))
            catch e
                println("Error processing action: $action, error: $e")
                push!(y_values, 0.0)  # Default to wait on error
            end
        end
        
        # Add jitter to separate satellites visually
        jitter = (i - 1) * 0.2
        
        scatter!(p2, episode_data.step, y_values .+ jitter, 
                label="Satellite $i", markersize=6, alpha=0.8)
    end
    
    # Add horizontal lines and annotations for action types
    hline!(p2, [0], linestyle=:dash, color=:gray, label=nothing)
    hline!(p2, [1], linestyle=:dash, color=:gray, label=nothing)
    hline!(p2, [2], linestyle=:dash, color=:gray, label=nothing)
    hline!(p2, [-1], linestyle=:dash, color=:gray, label=nothing)
    
    annotate!(p2, [(maximum(episode_data.step) + 1, -1, "Pass Left", :left),
                  (maximum(episode_data.step) + 1, 0, "Wait", :left),
                  (maximum(episode_data.step) + 1, 1, "Transmit", :left),
                  (maximum(episode_data.step) + 1, 2, "Pass Right", :left)])
    
    # 3. Network Flow Diagram: Sankey-like visualization
    p3 = plot(
        title="Network Flow Summary (Episode $episode)",
        xlabel="Satellite",
        ylabel="Packets",
        legend=:outertopright,
        size=(800, 500)
    )
    
    # Calculate total packets held by each satellite throughout episode
    total_held = vec(sum(packet_data, dims=1))
    
    # Calculate packets that were transmitted to ground by each satellite
    # We need to analyze action and packet changes
    transmitted_by_sat = zeros(Int, num_satellites)
    
    # Iterate through steps to detect successful transmissions
    for i in 1:(nrow(episode_data)-1)
        for sat in 1:num_satellites
            action_col = Symbol("sat$(sat)_action")
            if !(action_col in propertynames(episode_data))
                continue
            end
            
            action = episode_data[i, action_col]
            
            # Check if action starts with "transmit" (handle different string types)
            if startswith(String(action), "transmit")
                # Detect successful transmission by comparing packet counts
                packets_col = Symbol("sat$(sat)_packets")
                if !(packets_col in propertynames(episode_data))
                    continue
                end
                
                packets_before = episode_data[i, packets_col]
                packets_after = episode_data[i+1, packets_col]
                
                # If packets decreased and it's due to transmission (not passing)
                if packets_before > packets_after
                    transmitted_by_sat[sat] += packets_before - packets_after
                end
            end
        end
    end
    
    # Bar chart of packets held vs. transmitted
    bar!(p3, 1:num_satellites, total_held, label="Total Held", alpha=0.7)
    bar!(p3, 1:num_satellites, transmitted_by_sat, label="Transmitted", alpha=0.7)
    
    # 4. FSC Node transitions
    p4 = plot(
        title="Controller Node Transitions (Episode $episode)",
        xlabel="Time Step",
        ylabel="FSC Node",
        legend=:right,
        size=(800, 500)
    )
    
    # Plot node transitions for each satellite
    for i in 1:num_satellites
        node_col = Symbol("sat$(i)_node")
        if node_col in propertynames(episode_data)
            plot!(p4, episode_data.step, episode_data[:, node_col], 
                label="Satellite $i", marker=:circle, markersize=4, linewidth=2)
        end
    end
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
    
    # Save the plot if requested
    if output_path !== nothing
        savefig(combined_plot, output_path)
        println("Visualization saved to $output_path")
    end
    
    return combined_plot
end

"""
    batch_analyze_controllers(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP;
                            num_episodes=20, max_steps=50, output_dir="controller_analysis")

Runs a batch analysis of controller performance, saving both the raw data
and summary visualizations.
"""
function batch_analyze_controllers(joint_controller::JointController, prob::MultiPacketSatelliteNetworkPOMDP;
                                  num_episodes=20, max_steps=50, output_dir="controller_analysis")
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Run and record controller data
    csv_path = joinpath(output_dir, "controller_data.csv")
    data = run_and_record_controller(joint_controller, prob; 
                                    num_episodes=num_episodes, 
                                    max_steps=max_steps, 
                                    output_path=csv_path)
    
    # Generate summary statistics
    summary_df = DataFrame(
        episode = Int[],
        steps_to_complete = Int[],
        packets_transmitted = Int[],
        total_reward = Float64[],
        completed = Bool[]
    )
    
    for episode in 1:num_episodes
        episode_data = filter(row -> row.episode == episode, data)
        
        # Check if the episode completed (all packets transmitted)
        final_state = episode_data[end, :state]
        completed = final_state == "all-transmitted"
        
        # Count steps to completion
        steps = completed ? nrow(episode_data) : max_steps
        
        # Sum rewards
        total_reward = sum(episode_data.reward)
        
        # Count transmitted packets
        packets_transmitted = episode_data[end, :transmitted_packets]
        
        push!(summary_df, [episode, steps, packets_transmitted, total_reward, completed])
    end
    
    # Save summary statistics
    summary_path = joinpath(output_dir, "summary_statistics.csv")
    CSV.write(summary_path, summary_df)
    
    # Generate visualizations for a few sample episodes
    sample_episodes = min(5, num_episodes)
    for ep in 1:sample_episodes
        output_path = joinpath(output_dir, "episode_$(ep)_visualization.png")
        visualize_controller_data(csv_path; episode=ep, output_path=output_path)
    end
    
    # Create a summary plot
    p = plot(
        title="Controller Performance Summary",
        xlabel="Episode",
        ylabel="Value",
        legend=:right,
        size=(800, 500)
    )
    
    plot!(p, summary_df.episode, summary_df.steps_to_complete, 
          label="Steps to Complete", marker=:circle, color=:blue)
    
    plot!(p, summary_df.episode, summary_df.packets_transmitted, 
          label="Packets Transmitted", marker=:diamond, color=:green)
    
    plot!(p, summary_df.episode, summary_df.total_reward, 
          label="Total Reward", marker=:star, color=:red)
    
    completion_mask = summary_df.completed
    scatter!(p, summary_df.episode[completion_mask], fill(maximum(summary_df.steps_to_complete) * 1.1, sum(completion_mask)),
            marker=:square, markersize=8, label="Completed", color=:purple)
    
    summary_plot_path = joinpath(output_dir, "performance_summary.png")
    savefig(p, summary_plot_path)
    
    println("Batch analysis completed. Results saved to $output_dir")
    
    return summary_df
end

"""
    compare_controllers(controllers::Vector{JointController}, prob::MultiPacketSatelliteNetworkPOMDP,
                      names::Vector{String}; num_episodes=20, max_steps=50, output_dir="controller_comparison")

Compares multiple controllers by running them on the same problem and visualizing their performance.
"""
function compare_controllers(controllers::Vector{JointController}, prob::MultiPacketSatelliteNetworkPOMDP,
                           names::Vector{String}; num_episodes=20, max_steps=50, output_dir="controller_comparison")
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    summary_data = []
    
    # Run analysis for each controller
    for (i, controller) in enumerate(controllers)
        controller_dir = joinpath(output_dir, "controller_$(i)_$(names[i])")
        mkpath(controller_dir)
        
        summary = batch_analyze_controllers(controller, prob;
                                          num_episodes=num_episodes,
                                          max_steps=max_steps,
                                          output_dir=controller_dir)
        
        # Add controller name to summary
        summary[!, :controller] .= names[i]
        push!(summary_data, summary)
    end
    
    # Combine all summaries
    combined_summary = vcat(summary_data...)
    
    # Save combined summary
    combined_path = joinpath(output_dir, "combined_summary.csv")
    CSV.write(combined_path, combined_summary)
    
    # Create comparative visualizations
    
    # 1. Comparison of average steps to completion
    p1 = plot(
        title="Average Steps to Completion",
        ylabel="Steps",
        size=(600, 400),
        legend=:topleft,
        rotation=45
    )
    
    for name in names
        controller_data = filter(row -> row.controller == name, combined_summary)
        avg_steps = mean(controller_data.steps_to_complete)
        bar!(p1, [name], [avg_steps], label=nothing, alpha=0.7)
    end
    
    # 2. Comparison of completion rates
    p2 = plot(
        title="Completion Rate (%)",
        ylabel="Percentage",
        size=(600, 400),
        legend=:topleft,
        rotation=45
    )
    
    for name in names
        controller_data = filter(row -> row.controller == name, combined_summary)
        completion_rate = 100 * mean(controller_data.completed)
        bar!(p2, [name], [completion_rate], label=nothing, alpha=0.7)
    end
    
    # 3. Comparison of average reward
    p3 = plot(
        title="Average Total Reward",
        ylabel="Reward",
        size=(600, 400),
        legend=:topleft,
        rotation=45
    )
    
    for name in names
        controller_data = filter(row -> row.controller == name, combined_summary)
        avg_reward = mean(controller_data.total_reward)
        bar!(p3, [name], [avg_reward], label=nothing, alpha=0.7)
    end
    
    # 4. Comparison of average packets transmitted
    p4 = plot(
        title="Average Packets Transmitted",
        ylabel="Packets",
        size=(600, 400),
        legend=:topleft,
        rotation=45
    )
    
    for name in names
        controller_data = filter(row -> row.controller == name, combined_summary)
        avg_packets = mean(controller_data.packets_transmitted)
        bar!(p4, [name], [avg_packets], label=nothing, alpha=0.7)
    end
    
    # Combine the comparison plots
    comparison_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
    
    comparison_plot_path = joinpath(output_dir, "controller_comparison.png")
    savefig(comparison_plot, comparison_plot_path)
    
    println("Controller comparison completed. Results saved to $output_dir")
    
    return combined_summary
end