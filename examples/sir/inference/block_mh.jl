import Base.Iterators: partition

# Block resimulation MH that resimulates agents within each origin city
function block_mh(T::Int, observations::ChoiceMap, n_iters::Int,
                  tracked_vars=[:β], obs_noise=5.0, n_timeblocks=5)
    trs, scores = Vector{Trace}(undef, n_iters) ,Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    # Smart initialization of beta from the data
    init_choices, _, _ = propose(init_beta, (observations, min(T, 10)))
    observations = merge(observations, init_choices)
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
    trs[1], scores[1] = trace, get_score(trace)
    data[1, tracked_vars] = [trace[v] for v in tracked_vars]
    # Block addresses by times and processes (migration or transmission)
    model = get_retval(trace)
    t_blocks = reverse(collect(partition(1:T, div(T, n_timeblocks))))
    block_addrs = [Gen.select([:step => t => :agents => :step! =>
                               :agent_step! => 1 => a => fn
                               for t in ts for a in 1:nagents(model)]...)
                   for ts in t_blocks for fn in [:migrate!, :transmit!]]
    # Alternate between parameter proposal and resimulating blocks
    @showprogress for i in 2:n_iters
        trace, _ = mh(trace, Gen.select(:β))
        trace, _ = mh(trace, gaussian_drift, ())
        for addrs in block_addrs
            trace, accept = mh(trace, addrs)
        end
        trs[i], scores[i] = trace, get_score(trace)
        data[i, tracked_vars] = [trace[v] for v in tracked_vars]
    end
    return trs, scores, data
end
