## SMC algorithm with Gaussian drift rejuvenation of parameters

function smc_drift(T::Int, observations::ChoiceMap, n_particles::Int)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Step through filter
    println("Running particle filter with drift rejuvenation...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        if effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        pf_update!(state, (t, noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end
