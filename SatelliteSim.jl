module SatelliteSim
    using POMDPs
    using QuickPOMDPs: QuickPOMDP
    using POMDPTools: SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, UnderlyingMDP
    using DiscreteValueIteration: ValueIterationSolver
    using IterTools: product
    using Distributions: Categorical, pdf, rand
    export create_simulation, run_policy

    const SAT_IDS = 1:10
    const GROUND_ID = 11

    function create_simulation()
        # Amount of information in the satellite_network
        total_info = 1 # NOTE: I tried more info it never terminated

        # Define the full state space: all combinations of 0:total_info over 11 positions
        states = [Tuple(s) for s in product(fill(0:total_info, 11)...) if sum(s) == total_info]

        # Define the initial state: all information is in the first satellite
        initialstate = Deterministic(Tuple([total_info; zeros(Int, 10)]))
    
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
            # transition = function(s, a)
            #     new_s = collect(s)
            #     prob = 1.0
            #     transition_probs = Float64[]  # To store probabilities
                
            #     for idx in SAT_IDS
            #         if new_s[idx] > 0
            #             act = a[idx]
            #             if act == :left && idx > 1
            #                 new_s[idx] -= 1
            #                 new_s[idx - 1] += 1
            #                 push!(transition_probs, 0.8)
            #             elseif act == :right && idx < 10
            #                 new_s[idx] -= 1
            #                 new_s[idx + 1] += 1
            #                 push!(transition_probs, 0.8)
            #             elseif act == :ground
            #                 new_s[idx] -= 1
            #                 new_s[GROUND_ID] += 1
            #                 push!(transition_probs, 0.2)
            #             elseif act == :none
            #                 push!(transition_probs, 1.0)  # No change in state, probability = 1
            #             end
            #         end
            #     end
                
            #     # Normalize the probabilities so that they sum to 1
            #     total_prob = sum(transition_probs)
            #     transition_probs .= transition_probs ./ total_prob
                
            #     return SparseCat([Tuple(new_s)], transition_probs)
            # end,
            
            # transition = function(s, a)
            #     new_s = collect(s)
            #     prob = 1.0
            
            #     for idx in SAT_IDS
            #         if new_s[idx] > 0
            #             act = a[idx]
            #             if act == :left && idx > 1
            #                 new_s[idx] -= 1
            #                 new_s[idx - 1] += 1
            #                 prob = 0.8
            #             elseif act == :right && idx < 10
            #                 new_s[idx] -= 1
            #                 new_s[idx + 1] += 1
            #                 prob = 0.8
            #             elseif act == :ground
            #                 new_s[idx] -= 1
            #                 new_s[GROUND_ID] += 1
            #                 prob = 0.2
            #             elseif act == :none
            #                 prob = 1.0
            #             end
            #         end
            #     end
            
            #     return SparseCat([Tuple(new_s)], [prob])
            # end,
    
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
    
end  # module SatelliteSim
