## Data-driven SMC algorithm
# Data-driven Poisson regeneration proposal
# Gaussian drift rejuvenation of parameters

@gen function poisson_rejuv(tr::Trace, count::Int)
    {:step => t => :agents => :agent_step! => :transmit! => :n} ~ Poisson(count)
end

function data_driven_smc(T::Int, 
                         observations::ChoiceMap, 
                         n_particles::Int; 
                         uptick::Int = 3)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Compute number of infected per time step.
    num_infected_per_time = [to_array(get_selected(observations, Gen.select(:step => t => :obs => :infected)), Float64) |> length for t in 1 : T]
    println("Number of infected per time:\n$(num_infected_per_time)")
    # Step through filter
    println("Running data-driven particle filter with targeted Poisson regeneration...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        # Apply drift rejuvenation.
        if effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        # Check num infected - perform rejuvenation step in Poisson in transmit!
        num_infected_per_time[t] > uptick && pf_rejuvenate!(state, mh, (poisson_rejuv, num_infected_per_time[t]))
        pf_update!(state, (t, noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end
