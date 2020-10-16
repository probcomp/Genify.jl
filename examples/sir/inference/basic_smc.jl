## Basic SMC algorithm without rejuvenation

function basic_smc(T::Int, observations::ChoiceMap, n_particles::Int;
                   obs_noise::Float64=5.0)
    # Initialize filter
    state = pf_initialize(bayesian_sir, (0, obs_noise), choicemap(), n_particles)
    # Step through filter
    println("Running basic SMC...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        pf_update!(state, (t, obs_noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end
