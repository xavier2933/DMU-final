# run_dec_tiger.jl
# run_minimal.jl
# First, install any missing packages
using Pkg
Pkg.activate("dec_pomdp_env")  # Create a new environment
using POMDPs
using POMDPTools
using LinearAlgebra
using Random
include("dec_tiger.jl")  # Your Dec-Tiger implementation file
include("dec_vi_solver.jl")  # Policy iteration algorithm file

"""
stuff:
https://arxiv.org/pdf/1401.3460
https://www.scaler.com/topics/artificial-intelligence-tutorial/decentralized-pomdp/
"""

# # Create a Dec-Tiger instance
# println("Creating Dec-Tiger POMDP...")
# dec_tiger = DecTigerPOMDP()

# # Create initial controllers (small ones to start)
# println("Creating initial controllers...")
# initial_controllers = CreateInitialControllers(dec_tiger, 2)

# # Run a simplified version with fewer iterations
# println("Running policy iteration (2 iterations)...")
# final_controllers = PolicyIteration(dec_tiger, initial_controllers, max_iterations=100, epsilon=0.0001)

# # Print the results
# println("\nFinal controllers:")
# println("Agent 1 Controller:")
# for node in final_controllers[1].nodes
#     println("Node $(node.id): Action = $(node.action)")
#     println("  Transitions:")
#     for (obs, trans) in node.transitions
#         println("    On observation '$obs': $(trans)")
#     end
# end

# println("\nAgent 2 Controller:")
# for node in final_controllers[2].nodes
#     println("Node $(node.id): Action = $(node.action)")
#     println("  Transitions:")
#     for (obs, trans) in node.transitions
#         println("    On observation '$obs': $(trans)")
#     end
# end

# println("\nDone!")

# avg_reward = EvaluateControllers(dec_tiger, final_controllers, 100, 20)
# println("Average reward over 100 episodes: $avg_reward")

solve_dec_tiger()