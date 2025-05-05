using Pkg
Pkg.activate("dec_pomdp_env")

using POMDPs
using POMDPTools
using Statistics
using Random
using CSV
using DataFrames
using Plots

# Include your problem and solver definitions
include("../Problems/test.jl")  # Your MultiPacketSatelliteNetworkPOMDP definition
include("../Solvers/dec_pi_packet_solver.jl")  # Your solver
include("controller_visualization.jl")  # Include the FINAL visualization module

# Create a satellite network problem (adjust parameters as needed)
sat_network = MultiPacketSatelliteNetworkPOMDP(
    num_satellites = 3,  # Reduced to match your example
    total_packets = 3,   # Reduced to match your example
    max_capacity = 2,
    ground_tx_probs = [0.1, 0.7, .9],  # First satellite has low transmission probability, last has high
    discount_factor = 0.9,
    pass_success_prob = 0.95,
    observation_accuracy = 1.0,
    successful_tx_reward = 10.0,
    unsuccessful_tx_penalty = -2.0,
    pass_cost = -1.0,
    congestion_penalty = -5.0
)

# Create initial controller
println("Creating initial controller...")
initial_ctrl = create_initial_controllers(sat_network)

# Run policy iteration to create an optimized controller
println("Running policy iteration...")
iterations, final_controller = dec_pomdp_pi(initial_ctrl, sat_network)

# Verify the results
avg_reward, completion_rate = verify_satellite_controller(final_controller, sat_network)
println("Policy iteration completed in $iterations iterations.")
println("Final controller average reward: $avg_reward")
println("Completion rate: $(completion_rate * 100)%")

csvfile = "2_sats_3_packets.csv"
# Step 1: Save controller execution data to CSV
println("Recording controller execution data...")
data = run_and_record_controller(final_controller, sat_network, 
                               num_episodes=3,  # Run for 3 episodes
                               max_steps=15,    # Maximum of 15 steps per episode
                               output_path=csvfile)  # Save to CSV

# Step 2: Create visualizations from the CSV data
println("Creating visualizations...")
# Visualize episode 1
p1 = visualize_controller_data(csvfile, 
                             episode=1, 
                             output_path="2_sats_3_packets.png")

println("Done! Check the output files:")
println("- controller_data.csv: Raw execution data")
println("- episode_1_visualization.png: Visualization of episode 1")