## Data-driven SMC algorithm

@gen function migrate_proposal(prev_trace, t, obs)
    model = get_retval(prev_trace)
    # Propose to migration locations for each observed location
    for i in 1:nagents(model)
        loc_addr = :step => t => :obs => :location => i
        if !has_value(obs, loc_addr) continue end
        probs = [c == obs[loc_addr] ? .99 : .01/(model.C-1) for c in 1:model.C]
        {:step => t => :agents => :step! => :agent_step! => 1 =>
         i => :migrate! => :m} ~ Gen.categorical(probs)
    end
end

function data_driven_smc(T::Int, observations::ChoiceMap, n_particles::Int;
                         drift::Bool = false, obs_noise::Float64=5.0)
    # Initialize filter
    state = pf_initialize(bayesian_sir, (0, obs_noise), choicemap(), n_particles)
    model = get_retval(state.traces[1])
    println("Running SMC with data-driven migration proposal...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        # Apply drift rejuvenation.
        if drift && effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        # Upsample transmission counts based on number newly infected
        pf_update!(state, (t, obs_noise), (UnknownChange(), NoChange()),
                   obs, migrate_proposal, (t, obs))
    end
    # Return particles and log ML estimate
    return state
end
