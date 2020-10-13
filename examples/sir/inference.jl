using Gen, Genify, Distributions
using ProgressMeter: @showprogress

include("model.jl")

trunc_normal(args...) = Genify.WrappedDistribution(TruncatedNormal(args...))

# Gaussian drift proposal
@gen function gaussian_drift(tr::Trace, params=Dict(:β => (0.1, 0, 0.6)))
    for (addr, distargs) in params
        if distargs isa Tuple && length(distargs) == 3
            sigma, minval, maxval = distargs
            {addr} ~ trunc_normal(tr[addr], sigma, minval, maxval)()
        else
            sigma = distargs
            {addr} ~ normal(tr[addr], sigma)()
        end
    end
end

# MH algorithm that fully resimulates the underlying simulator
function resimulation_mh(T::Int, n_iters::Int, observations::ChoiceMap,
                         tracked_vars=[:β])
    scores = Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    trace, _ = generate(bayesian_sir, (T,), observations)
    scores[1] = get_score(trace)
    for v in tracked_vars
        data[1, v] = trace[v]
    end
    @showprogress for i in 2:n_iters
        trace, _ = mh(trace, gaussian_drift, ())
        trace, _ = mh(trace, Gen.select(:step))
        scores[i] = get_score(trace)
        for v in tracked_vars
            data[i, v] = trace[v]
        end
    end
    return trace, scores, data
end

## Generate observations and run resimulation MH
trace, _ = generate(bayesian_sir, (90,), choicemap(:β => 0.3))
observations = choicemap()
set_submap!(observations, :obs, get_submap(get_choices(trace), :obs))

trace, scores, data = resimulation_mh(90, 50, observations)
