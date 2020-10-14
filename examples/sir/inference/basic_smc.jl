## Basic SMC algorithm without rejuvenation

function smc_basic(T::Int, observations::ChoiceMap, n_particles::Int)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Step through filter
    println("Running basic particle filter...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        pf_update!(state, (t, noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end
