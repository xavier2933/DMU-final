# Create POMDP, solve using value iteration, and run the policy
include("src/SatelliteSim.jl")
using .SatelliteSim  # note the dot! this loads the *local* module


base_case_flag = false
first_case_flag = true

if base_case_flag
    # Base Case
    num_satellites::Int = 10
    num_ground_stations::Int = 1
    initial_information_vector::Tuple = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
elseif first_case_flag
    # First Case
    num_satellites::Int = 5
    num_ground_stations::Int = 1
    initial_information_vector::Tuple = (1, 0, 0, 0, 0, 0)
else
    # Second Case
    num_satellites::Int = 3
    num_ground_stations::Int = 2
    initial_information_vector::Tuple = (1, 0, 0, 0, 0)
end

satellite_network_POMDP = create_simulation(
    num_satellites = num_satellites,
    num_ground_stations = num_ground_stations,
    initial_information_vector = initial_information_vector
)
run_policy(satellite_network_POMDP)
