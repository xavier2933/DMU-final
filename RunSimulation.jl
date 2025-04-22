# Create POMDP, solve using value iteration, and run the policy
include("src/SatelliteSim.jl")
using .SatelliteSim  # note the dot! this loads the *local* module

satellite_network_POMDP = create_simulation()
run_policy(satellite_network_POMDP)
