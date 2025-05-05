# SatelliteSim - A Satellite Network Simulation

`SatelliteSim` is a simulation model for managing a satellite network using Partially Observable Markov Decision Processes (POMDPs). The goal is to model how satellites can transmit data either to other satellites or to a ground station, based on a policy that optimizes the transmission process. The module provides functionality to define and simulate a satellite network, including actions such as left, right, ground, and none for each satellite.

## Usage

This is a Julia package. Ensure that you have Julia installed on your system, and install all required dependencies. The code automatically activates a development environment, which should ensure compatability. 

In order to run the simulation, run the main script at `src/main.jl`. There, you can modify the parameters of the simulation, shown in the below code block:
```
sat_network = MultiPacketSatelliteNetworkPOMDP(
    num_satellites = 4,
    total_packets = 4,
    max_capacity = 2,
    ground_tx_probs = [0.2, 0.8, 0.9, 0.99],  # Increased the first satellite's tx probability
    discount_factor = 0.95,             # Slightly higher discount factor to value future rewards more
    pass_success_prob = 0.95,
    observation_accuracy = 1.0,
    successful_tx_reward = 80.0,        # Doubled the reward for successful transmission
    unsuccessful_tx_penalty = -1.0,     # Reduced penalty for unsuccessful transmission
    pass_cost = -0.1,                   # Reduced pass cost to encourage packet movement
    congestion_penalty = -20.0           # Increased congestion penalty to discourage holding packets
)
```

## Contents

The main scripts of interest are `Problems/multi_packet_satellite_pomdp.jl`, which contains the definition of the problem, `Solvers/dec_pi_packet_solver`, which contains the code for policy iteration, and `src/main.jl`, which provides a driver script for the package. The others problems/solvers were developed on the way to get the main one to work.


### Problems

Definitions for the three different types of problems that were used to verify the funcitonality of our solver. These were the dec-Tiger problem, and a multi-agent rescue problem. The respective solvers for these are in the Solvers folder.

### Solvers

This is where the solvers live. The one for our project is `dec_pi_packet_solver.jl`. The others are for the other problems, or earlier iterations of this solver.

### dec_pomdp_env
Julia environment for this project. Probably not best practice to commit this, but I think it makes it easier to run?

### prev

Initial work for MDP.

### src

Where the driver and visualization code live.
