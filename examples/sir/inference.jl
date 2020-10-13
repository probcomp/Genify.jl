using Gen, Genify, GenParticleFilters, Distributions
using ProgressMeter: @showprogress

include("model.jl")

## Helper functions

function aggregate_obs(trace::Trace)
    T = get_args(trace)[1]
    obs_addrs = Gen.select([:step => t => :obs for t in 1:T]...)
    observations = Gen.get_selected(get_choices(trace), obs_addrs)
    return observations
end

function migration_obs(trace::Trace)
    model, T = get_retval(trace), get_args(trace)[1]
    m_addrs = Gen.select(
        [:step => t => :agents => :step! => :agent_step! => 1 =>
         agt => :migrate! => :m for t in 1:T for agt in 1:nagents(model)]...
    )
    observations = Gen.get_selected(get_choices(trace), m_addrs)
    return observations
end

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
    if !haskey(kwargs, :ls)
        kwargs = Dict{Symbol,Any}(kwargs...)
        kwargs[:ls] = reduce(hcat, [[:solid :dash] for i in cities])
    end
    labels = legend == false ? fill("", length(cnames)) : names(df)
    plot!(Matrix(df); lw=2, labels=permutedims(labels),
          legend=legend, ylims=[0, maximum(model.Ns)*1.5], kwargs...)
end

function plot_obs(tr::Trace, cities=nothing, legend=:topleft; kwargs...)
    plot(); plot_obs!(tr, cities, legend; kwargs...)
end

function plot_obs!(trs::AbstractVector{<:Trace}, cities=nothing; kwargs...)
    for tr in trs plot_obs!(tr, cities, false; kwargs...) end
    return plot!()
end

function plot_obs!(trs::AbstractVector{<:Trace}, ws::AbstractVector{Float64},
                  cities=nothing; kwargs...)
    max_w = maximum(ws)
    for (tr, w) in zip(trs, ws)
        plot_obs!(tr, cities, false; alpha=w/max_w, kwargs...)
    end
    return plot!()
end

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
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
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
trace, _ = generate(bayesian_sir, (50, 0.01), choicemap(:β => 0.5));
observations = aggregate_obs(trace)
plot_obs(trace)

trace, scores, data = resimulation_mh(50, observations, 100)

## Basic SMC algorithm without rejuvenation

function smc_basic(T::Int, observations::ChoiceMap, n_particles::Int)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Step through filter
    println("Running particle filter...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        pf_update!(state, (t, noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end

observations = aggregate_obs(trace)
pf_state = smc_basic(50, observations, 50);
lml_est = log_ml_estimate(pf_state)
β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
βs = [tr[:β] for tr in pf_state.traces]
trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

plot_obs(trace, [1]; color=[:red :blue])
plot_obs!(trs, ws, [1], lw=1, color=:grey)
plot!(legend=:topleft, title="Vanilla SMC, N=50")
savefig("smc_no_migration.png")

observations = aggregate_obs(trace)
m_observations = merge(observations, migration_obs(trace));
pf_state = smc_basic(50, m_observations, 25);
lml_est = log_ml_estimate(pf_state)
β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
βs = [tr[:β] for tr in pf_state.traces]
trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

plot_obs(trace, [1]; color=[:red :blue])
plot_obs!(trs, ws, [1], lw=1, color=:grey)
plot!(legend=:topleft, title="SMC with conditioning on migration history, N=25")
savefig("smc_with_migration.png")

## SMC algorithm with Gaussian drift rejuvenation of parameters

function smc_drift(T::Int, observations::ChoiceMap, n_particles::Int)
    # Initialize filter
    noise = 5.0
    state = pf_initialize(bayesian_sir, (0, noise), choicemap(), n_particles)
    # Step through filter
    println("Running particle filter...")
    @showprogress for t=1:T
        obs = get_selected(observations, Gen.select(:step => t))
        if effective_sample_size(state) < 0.25 * n_particles
            pf_resample!(state, :residual)
            pf_rejuvenate!(state, mh, (gaussian_drift, ()))
        end
        pf_update!(state, (t, noise), (UnknownChange(), NoChange()), obs)
    end
    # Return particles and log ML estimate
    return state
end

pf_state = smc_drift(50, observations, 50);
lml_est = log_ml_estimate(pf_state)
β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
βs = [tr[:β] for tr in pf_state.traces]
trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

plot_obs(trace, [1])
plot_obs!(trs, ws, [1], lw=1, color=:grey)
