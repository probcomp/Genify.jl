## Resimulation MH

# MH algorithm that fully resimulates the underlying simulator
function resimulation_mh(T::Int, observations::ChoiceMap, n_iters::Int,
                         tracked_vars=[:β], obs_noise=5.0)
    trs, scores = Vector{Trace}(undef, n_iters) ,Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    # Smart initialization of beta from the data
    init_choices, _, _ = propose(init_beta, (observations, min(T, 10)))
    observations = merge(observations, init_choices)
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
    trs[1], scores[1] = trace, get_score(trace)
    data[1, tracked_vars] = [trace[v] for v in tracked_vars]
    # Get all addreses for the simulator
    abm_addrs = Gen.select([:step => t => :agents for t in 1:T]...)
    println("Running resimulation MH...")
    @showprogress for i in 2:n_iters
        trace, _ = mh(trace, Gen.select(:β)) # Propose new parameters
        trace, _ = mh(trace, gaussian_drift, ()) # Propose new parameters
        trace, _ = mh(trace, abm_addrs) # Resimulate entire SIR model
        trs[i], scores[i] = trace, get_score(trace)
        data[i, tracked_vars] = [trace[v] for v in tracked_vars]
    end
    return trs, scores, data
end
