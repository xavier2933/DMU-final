module SatelliteSim
    using POMDPs
    using QuickPOMDPs: QuickPOMDP
    using POMDPTools: SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, UnderlyingMDP
    using DiscreteValueIteration: ValueIterationSolver
    using IterTools: product
    using Distributions: Categorical, pdf, rand
    export create_simulation, run_policy, pomcp_solve, simulate_pomcp_policy
    using BasicPOMCP


    
    function create_simulation( ; 
                                num_satellites::Int,
                                num_ground_stations::Int,
                                initial_information_vector::Tuple
                            )

        @assert length(initial_information_vector) == (num_satellites + num_ground_stations) "Length of initial_information_vector must match num_satellites + num_ground_stations"

        SAT_IDS = 1:num_satellites
        GROUND_ID = num_satellites + 1

        total_info = sum(initial_information_vector)

        # Define the full state space: all combinations of info across satellites and ground stations summing to total_info
        states = [Tuple(s) for s in product(fill(0:total_info, num_satellites + num_ground_stations)...) if sum(s) == total_info]

        # Define initial state
        initialstate = Deterministic(initial_information_vector)

        satellite_network = QuickPOMDP(

            # States are all combinations of 0:total_info over 11 positions
            states = states,

            # Actions are all combinations of :left, :right, :ground, :none over 10 positions
            actions = collect(Iterators.product(fill([:left, :right, :ground, :none], 10)...)),
    
            # Transition to ground with probability 0.2, left/right with probability 0.8, and do nothing with probability 1.0
            transition = function(s, a)
                new_s = collect(s)
                prob = 1.0
                transition_probs = Float64[]  # To store probabilities
            
                for idx in SAT_IDS
                    if new_s[idx] > 0
                        act = a[idx]
                        if act == :left && idx > 1
                            new_s[idx] -= 1
                            new_s[idx - 1] += 1
                            push!(transition_probs, 0.8)
                        elseif act == :right && idx < 10
                            new_s[idx] -= 1
                            new_s[idx + 1] += 1
                            push!(transition_probs, 0.8)
                        elseif act == :ground
                            new_s[idx] -= 1
                            new_s[GROUND_ID] += 1
                            push!(transition_probs, 0.2)
                        elseif act == :none
                            push!(transition_probs, 1.0)  # No change in state, probability = 1
                        end
                    end
                end
            
                # Normalize the probabilities so that they sum to 1
                if length(transition_probs) > 0
                    total_prob = sum(transition_probs)
                    transition_probs .= transition_probs ./ total_prob
                else
                    # If no transitions occurred, set a default transition
                    transition_probs .= [1.0]
                    new_s .= s  # No change in state
                end
            
                return SparseCat([Tuple(new_s)], transition_probs)
            end,            
            
            # Observation is just the state for now
            observation = (s, a, sp) -> sp,
            obstype = NTuple{11, Int},

            # Reward is the change in information at the ground station
            reward = (s, a, sp) -> sp[GROUND_ID] - s[GROUND_ID],
    
            # Penalize holding information at satellite
            discount = 0.9,

            # Initial state is all information in the first satellite
            initialstate = initialstate,

            # Terminal state is when all information is at the ground station
            isterminal = s -> s[GROUND_ID] == total_info
        )
    
        return satellite_network
    end
    

    function run_policy(pomdp)
        solver = ValueIterationSolver()
        mdp = UnderlyingMDP(pomdp)  # Convert the POMDP into an MDP for planning

        state_list = ordered_states(mdp)
        action_list = ordered_actions(mdp)
        
        println("State space size: ", length(state_list))
        println("Action space size: ", length(action_list))
        
        policy = solve(solver, mdp)
    
        s = rand(initialstate(pomdp))  # Draw initial state from distribution
        println("Initial state: ", s)
    
        # Run the policy for a fixed number of steps
        for t in 1:5

            if isterminal(pomdp, s)
                println("Reached terminal state: ", s)
                break
            end

            a = action(policy, s)
            println("t=$t | state=$s | action=$a")
    
            # Get next states and transition probabilities
            dist = transition(pomdp, s, a)
            
            # Get next states and transition probabilities
            possible_next_states = support(dist)

            if isempty(possible_next_states)
                println("No valid transitions for action: ", a)
                break
            end

            probs = [pdf(dist, sp) for sp in possible_next_states]
            
            if !isempty(possible_next_states)
                next_index = rand(Categorical(probs))
                s = possible_next_states[next_index]
            else
                println("No transitions possible from this state.")
                break
            end
        end
    end

    function pomcp_solve(m) # this function makes capturing m in the rollout policy more efficient
        c_val = 50.0
        println("C=$c_val")
        mdp = UnderlyingMDP(m)
        
        # solve mdp to get decent policy for rollout
        mdp_solver = ValueIterationSolver(max_iterations=1000)
        mdp_policy = solve(mdp_solver, mdp)

        solver = POMCPSolver(tree_queries=10,
                            c=c_val,
                            max_time = 0.5, # this should be enough time to get a score in the 30s
                            default_action=:ground,
                            estimate_value=FORollout(mdp_policy))
        pomcp_p = solve(solver, m)
        return pomcp_p
    end

    function simulate_pomcp_policy(pomcp_p, pomdp; num_steps=10, verbose=true)
        # Initialize state
        s = rand(initialstate(pomdp))
        
        total_reward = 0.0
        
        if verbose
            println("Initial state: ", s)
        end
        
        # Run simulation for specified number of steps
        for t in 1:num_steps
            if isterminal(pomdp, s)
                if verbose
                    println("Reached terminal state: ", s)
                end
                break
            end
            
            # Select action using the POMCP policy
            a = action(pomcp_p, s)
            
            if verbose
                println("t=$t | state=$s | action=$a")
            end
            
            # Try to get next state directly from your model
            # Since we're having issues with the transition function
            sp = nothing
            try
                # Try to get the next state without using transition function
                # If you have a custom function, use that instead
                sp = next_state(pomdp, s, a)
            catch e
                if verbose
                    println("Error getting next state: ", e)
                    println("Ending simulation.")
                end
                break
            end
            
            # If we couldn't get a next state, end the simulation
            if sp === nothing
                if verbose
                    println("Could not determine next state, ending simulation.")
                end
                break
            end
            
            # Update state
            s = sp
        end
        
        if verbose
            println("Simulation ended.")
        end
        
        return total_reward
    end
    
    # Helper function to get next state without using transition
    function next_state(pomdp, s, a)
        # This is a placeholder - replace with your specific state transition logic
        # Example implementation:
        if a == :ground
            # Logic for ground action
            # For example, update the first element to indicate completed ground action
            new_s = collect(s)
            new_s[1] = 1  # Set first element to 1
            return Tuple(new_s)
        elseif a isa Integer || a isa Symbol
            # Logic for satellite actions
            # For example, update the corresponding satellite element
            new_s = collect(s)
            if a isa Integer && 1 <= a <= length(new_s)
                new_s[a] = 1  # Update the satellite state
            end
            return Tuple(new_s)
        else
            # Handle other action types
            return s  # Return same state if action not recognized
        end
    end

    
end  # module SatelliteSim