## SMC algorithm with Gaussian drift rejuvenation of parameters

function drift_smc(T::Int, observations::ChoiceMap, n_particles::Int;
                   obs_noise::Float64=5.0)
    # Initialize filter
    state = pf_initialize(bayesian_sir, (0, obs_noise), choicemap(), n_particles)
    # Step through filter
    println("Running SMC with drift rejuvenation...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        if effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        pf_update!(state, (t, obs_noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end
