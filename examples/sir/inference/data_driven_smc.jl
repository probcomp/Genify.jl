## Data-driven SMC algorithm
# Data-driven Poisson regeneration proposal
# Gaussian drift rejuvenation of parameters

@gen function poisson_prop(trace, t::Int, λ::Int)
    {:step => t => :agents => :agent_step! => :transmit! => :n} ~ poisson(λ)
end

function data_driven_smc(T::Int, 
                         observations::ChoiceMap, 
                         n_particles::Int; 
                         drift::Bool = false,
                         uptick::Float64 = 7.0)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Compute number of infected per time step.
    max_num_infected_per_time = [to_array(get_selected(observations, Gen.select(:step => t => :obs => :infected)), Float64) |> maximum for t in 1 : T]
    Δ_max_num_infected_per_time = max_num_infected_per_time - append!(Float64[0.0], max_num_infected_per_time[1 : end - 1])
    println("Δ max of infected per time:\n$(Δ_max_num_infected_per_time)")
    # Step through filter
    println("Running data-driven particle filter with targeted Poisson proposal...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        # Apply drift rejuvenation.
        if drift && effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        # Check num infected - perform proposal step for Poisson in transmit!
        # TODO: Proposal needs to propose to more than just Poisson? For some reason I thought this was required.
        if Δ_max_num_infected_per_time[t] > uptick
            λ = Int(floor(Δ_max_num_infected_per_time[t]))
            pf_update!(state, (t, noise), (UnknownChange(), NoChange()), 
                       obs, poisson_prop, (t, λ))
        else
            pf_update!(state, (t, noise), (UnknownChange(), NoChange()), 
                       obs)
        end
    end
    # Return particles and log ML estimate
    return state
end
