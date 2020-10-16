## Resimulation MH

# Gaussian drift proposal
@gen function gaussian_drift(tr::Trace, params=Dict(:β => (0.3, 0.01, Inf)))
    for (addr, distargs) in params
        if distargs isa Tuple && length(distargs) == 3
            sigma, lb, ub = distargs
            {addr} ~ trunc_normal(tr[addr], sigma, lb, ub)()
        else
            sigma = distargs
            {addr} ~ normal(tr[addr], sigma)()
        end
    end
end

# MH algorithm that fully resimulates the underlying simulator
function resimulation_mh(T::Int, observations::ChoiceMap, n_iters::Int,
                         tracked_vars=[:β], obs_noise=5.0)
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
    println("Running resimulation MH with drift proposals...")
    @showprogress for i in 2:n_iters
        trace, _ = mh(trace, Gen.select(:β)) # Propose new parameters
        trace, _ = mh(trace, gaussian_drift, ()) # Propose new parameters
        trace, _ = mh(trace, abm_addrs) # Resimulate entire SIR model
        scores[i] = get_score(trace)
        for v in tracked_vars
            data[i, v] = trace[v]
        end
        trs[i] = trace
    end
    return trs, scores, data
end
