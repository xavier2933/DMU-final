# SatelliteSim - A Satellite Network Simulation

`SatelliteSim` is a simulation model for managing a satellite network using Partially Observable Markov Decision Processes (POMDPs). The goal is to model how satellites can transmit data either to other satellites or to a ground station, based on a policy that optimizes the transmission process. The module provides functionality to define and simulate a satellite network, including actions such as left, right, ground, and none for each satellite.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [Creating the Simulation](#creating-the-simulation)
   - [Running the Policy](#running-the-policy)
3. [State Space](#state-space)
4. [Actions](#actions)
5. [Transitions](#transitions)
6. [Rewards](#rewards)
7. [Module Overview](#module-overview)
8. [Dependencies](#dependencies)
9. [License](#license)
10. [Contributing](#contributing)

## Installation

To use `SatelliteSim`, you need to have Julia installed. You can install the required dependencies by using Julia's package manager.

1. Copy the `SatelliteSim.jl` module from this repository:
2. Install the required packages by running the following command in Julia's REPL:

```julia
using Pkg
Pkg.add("POMDPs")
Pkg.add("QuickPOMDPs")
Pkg.add("POMDPTools")
Pkg.add("DiscreteValueIteration")
Pkg.add("IterTools")
Pkg.add("Distributions")
```

## Usage
Assume the structure of the project is as follows:

```
DMU-Final
├── src
│   └── SatelliteSim.jl
└── RunSimulation.jl
``` 

Then you can run the simulation by copying the `SatelliteSim.jl` file into the `src` directory and running the `RunSimulation.jl` script.


### Creating the Simulation

The first step in using the module is creating a satellite network simulation. You can use the `create_simulation` function to generate a simulation environment:

```julia
satellite_network = create_simulation()
```

### Running the Policy
Then, you can run the policy on the simulation environment:

```julia
run_policy(satellite_network_POMDP)
```
This will execute the policy and return the results of the simulation.


## State Space

The state space in `SatelliteSim` represents all possible configurations of information within the satellite network. Each state is a tuple of length 11 (one for each satellite and one for the ground station), where the values represent the amount of information held by each node.

- **Satellite State**: A tuple with values `[s1, s2, ..., s10]` representing the amount of information held by each satellite.
- **Ground Station State**: The last element in the tuple, representing the amount of information at the ground station.

### Example of State Representation

```julia
# Example state where the first satellite has all the information
state = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
```

## Actions Space

The actions available for each satellite are:
- `:left`: Move information to the satellite on the left (if possible).
- `:right`: Move information to the satellite on the right (if possible).
- `:ground`: Send information to the ground station.
- `:none`: No action, remain in the current state.

Each satellite can independently choose one of these actions at each time step.

### Example of Actions

```julia
# Example action for each satellite
actions = (:left, :right, :ground, :none, :none, :none, :none, :none, :none, :none)
```

## Transitions

The transition model defines how the system evolves based on the current state and chosen actions. It uses probabilities to model the likelihood of transitioning from one state to another based on the satellite actions.

Currently, there is a single transition model that is used for all satellites. The transition model is defined in the `SatelliteSim.jl` file. Satellite to satellite transitions are deterministic with a probability of 0.8 and ground station transitions are deterministic with a probability of 0.2. The transition model is defined in the `transition` function.

## Rewards

The reward function assigns a reward based on the state transitions. The reward is defined as the difference between the amount of information at the ground station (`sp[GROUND_ID]`) at the next state and the amount of information currently at the ground station (`s[GROUND_ID]`).

- **Reward**: `reward(s, a, sp) = sp[GROUND_ID] - s[GROUND_ID]`

This rewards the system for transferring more information to the ground station.

### Example of Reward Function

```julia
reward = (s, a, sp) -> sp[GROUND_ID] - s[GROUND_ID]
```

## Discount Factor
The discount factor is set to 0.9, which is used in the value iteration algorithm to determine the present value of future rewards.

## Initial State
The initial state is set to have all information at the first satellite and none at the ground station:

```julia
initial_state = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
```

## Terminal State
The terminal state is defined as the state where all information has been successfully transmitted to the ground station. In this case, the ground station holds all the information, and all other satellites have none.

## Policy Solver
The policy solver uses value iteration on the underlying MDP to find the optimal policy for the satellite network. The policy is then used to determine the actions for each satellite based on the current state.

## Simulation Rollout
The simulation rollout function executes the policy on the satellite network and returns the results of the simulation.
The rollout function is defined in the `run_policy` function, which simulates the satellite network based on the current policy and returns the results.

## Notes
This did not work for 5 units of information in about 15 minutes of run time. The transitions are deterministic, so it should be able to find the optimal policy. The issue may be with the reward function or the transition model. I think the reward function needs to consider transition probabilities.


