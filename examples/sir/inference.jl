using Gen, Genify, GenParticleFilters, Distributions
using ProgressMeter: @showprogress

include("model.jl")

## Helper functions

function obs_dataframe(tr::Trace)
    model, T = get_retval(tr), get_args(tr)[1]
    data = reduce(vcat, tr[:step => t => :obs]' for t in 1:T)
    cnames = reduce(vcat, [[Symbol(:I, i), Symbol(:R, i)] for i in 1:model.C])
    return DataFrame(data, cnames)
end

function plot_obs!(tr::Trace, cities=nothing, legend=false; kwargs...)
    model = get_retval(tr)
    cities = isnothing(cities) ? collect(1:model.C) : cities
    cnames = reduce(vcat, [[Symbol(:I, i), Symbol(:R, i)] for i in cities])
    df = obs_dataframe(tr)[:, cnames]
    plot!(Matrix(df); lw=1.5, label=permutedims(names(df)),
          legend=legend, ylims=[0, maximum(model.Ns)], kwargs...)
end

function plot_obs(tr::Trace, cities=nothing, legend=:topleft; kwargs...)
    plot(); plot_obs!(tr, cities, legend; kwargs...)
end

function plot_obs(trs::AbstractVector{<:Trace}, cities=nothing; kwargs...)
    plt = plot();
    for tr in trs
        plot_obs!(tr, cities, false; kwargs...)
    end
    return plt
end

function plot_obs(trs::AbstractVector{<:Trace}, ws::AbstractVector{Float64},
                  cities=nothing; kwargs...)
    plt = plot();
    max_w = maximum(ws)
    for (tr, w) in zip(trs, ws)
        plot_obs!(tr, cities, false; alpha=w/max_w, kwargs...)
    end
    return plt
end


## Resimulation MH

# Gaussian drift proposal
@gen function gaussian_drift(tr::Trace, params=Dict(:β => (0.3, 0, 0.6)))
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
                         tracked_vars=[:β])
    scores = Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    trace, _ = generate(bayesian_sir, (T,), observations)
    scores[1] = get_score(trace)
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
    end
    return trace, scores, data
end

## Generate observations and run resimulation MH
trace, _ = generate(bayesian_sir, (50,), choicemap(:β => 0.5))
plot_obs(trace, [1])

choices = get_choices(trace)
obs_addrs = Gen.select([:step => t => :obs for t in 1:50]...)
observations = Gen.get_selected(choices, obs_addrs)

trace, scores, data = resimulation_mh(50, observations, 100)

## Particle filter
function particle_filter(T::Int, observations::ChoiceMap, n_particles::Int)
    # Initialize filter
    state = pf_initialize(bayesian_sir, (0,), choicemap(), n_particles)
    # Step through filter
    println("Running particle filter...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t => :obs))
        # maybe_resample!(state, ess_threshold=n_particles/2)
        pf_update!(state, (t,), (UnknownChange(),), obs)
    end
    # Return particles and log ML estimate
    return state
end

pf_state = particle_filter(50, observations, 10);
lml_est = log_ml_estimate(pf_state)
β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
βs = [tr[:β] for tr in pf_state.traces]
trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

plot_obs(trs)
plot_obs(trs, ws, [1])
plot_obs!(trace, [1]; color=:black)
