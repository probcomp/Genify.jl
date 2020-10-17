## Data-driven SMC algorithm

@gen function step_proposal(prev_trace, t, newly_infected)
    model = get_retval(prev_trace)
    n_prev_undetected = sum(a.days_infected < model.infection_period &&
                            a.status == :I for a in values(model.agents))
    if newly_infected <= 0 || n_prev_undetected <= 0 return end
    # Compute average newly infected per previously infected agent
    β = newly_infected / n_prev_undetected
    fitted_model = deepcopy(model)
    fitted_model.β_det .= β
    {:step => t => :agents} ~ abm_step!(fitted_model, agent_step!, 1, true)
    return nothing
end

# Gets total observed infection counts for all timesteps from 1 to T
get_infected(obs::ChoiceMap, model::ABM, T::Int) =
    [sum(obs[:step=>t=>:obs=>:infected=>c] for c in 1:model.C) for t in 1:T]

get_recovered(obs::ChoiceMap, model::ABM, T::Int) =
    [sum(obs[:step=>t=>:obs=>:recovered=>c] for c in 1:model.C) for t in 1:T]

get_susceptible(obs::ChoiceMap, model::ABM, T::Int) = nagents(model) .-
    (get_infected(obs, model, T) .+ get_recovered(obs, model, T))

function moving_average(vs, n)
    avgs = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
    return [avgs; fill(avgs[end], n-1)]
end

function data_driven_smc(T::Int, observations::ChoiceMap, n_particles::Int;
                         drift::Bool = false, obs_noise::Float64=5.0)
    # Initialize filter
    state = pf_initialize(bayesian_sir, (0, obs_noise), choicemap(), n_particles)
    model = get_retval(state.traces[1])
    # Compute number of newly infected per person per time step.
    susceptible = get_susceptible(observations, model, T) .|> round
    newly_infected = - diff([nagents(model) - sum(model.Is); susceptible])
    newly_infected = moving_average(newly_infected, 5)
    println("Running SMC with data-driven transmission count proposal...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        # Apply drift rejuvenation.
        if drift && effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        # Upsample transmission counts based on number newly infected
        pf_update!(state, (t, obs_noise), (UnknownChange(), NoChange()),
                   obs, step_proposal, (t, newly_infected[t]))
    end
    # Return particles and log ML estimate
    return state
end
