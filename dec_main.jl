# run_dec_tiger.jl
# run_minimal.jl
# First, install any missing packages
using Pkg
Pkg.activate("dec_pomdp_env")  # Create a new environment
using POMDPs
using POMDPTools
using LinearAlgebra
using Random
# include("dec_tiger.jl")  # Your Dec-Tiger implementation file
# include("dec_vi_solver.jl")  # Policy iteration algorithm file

"""
stuff:
https://arxiv.org/pdf/1401.3460
https://www.scaler.com/topics/artificial-intelligence-tutorial/decentralized-pomdp/
"""

include("GENERIC_dec_tiger.jl")
include("dec_tiger_adapted.jl")

function main()
    # Create Dec-Tiger problem
    dec_tiger = DecTigerPOMDP()
    
    # Create initial controller
    ctrl = create_heuristic_controller(dec_tiger)
    
    # Run policy iteration
    iterations, improved_ctrl = dec_pomdp_pi(ctrl, dec_tiger, max_iterations=10, epsilon=0.01)
    
    # Verify the controller
    avg_reward, success_rate = verify_controller(improved_ctrl, dec_tiger, num_episodes=1000, max_steps=20)
    
    println("\n=== Final Results ===")
    println("Iterations: $iterations")
    println("Average reward: $avg_reward")
    println("Success rate: $(success_rate * 100)%")
end

main()