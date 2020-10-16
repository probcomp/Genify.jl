## Single site MH
#
## MH algorithm that fully resimulates the underlying simulator
function single_site_mh(T::Int, 
                        observations::ChoiceMap, 
                        n_iters::Int,
                        tracked_vars=[:β], 
                        obs_noise=5.0)
    scores = Vector{Float64}(undef, n_iters)
    trs = Vector{Trace}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
    scores[1] = get_score(trace)
    trs[1] = trace
    for v in tracked_vars
        data[1, v] = trace[v]
    end
    abm_addrs = Gen.select([:step => t => :agents for t in 1:T]...)
    println("Running single site MH...")
    @showprogress for i in 2:n_iters
        trace, _ = mh(trace, abm_addrs) # Resimulate entire SIR model
        for t in 1 : T
            trace, _ = mh(trace, Gen.select(:step => t => :agents => :agent_step! => :migrate! => :m))
            trace, _ = mh(trace, Gen.select(:step => t => :agents => :agent_step! => :transmit! => :n))
            trace, _ = mh(trace, Gen.select(:step => t => :agents => :agent_step! => :recover! => :r))
        end
        scores[i] = get_score(trace)
        for v in tracked_vars
            data[i, v] = trace[v]
        end
        trs[i] = trace
    end
    return trs, scores, data
end

