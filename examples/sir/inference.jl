module Inference

using Gen, Genify, GenParticleFilters, Distributions, StatsBase
using DataFrames, CSV
using ProgressMeter: @showprogress

include("model.jl")
include("utils.jl")
include("inference/gaussian_drift.jl")
include("inference/resimulation_mh.jl")
include("inference/block_mh.jl")
include("inference/single_site_mh.jl")
include("inference/basic_smc.jl")
include("inference/drift_smc.jl")
include("inference/data_driven_smc.jl")

show_plots = true
save_plots = false

run_resimulation_mh = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    trs, scores, data = resimulation_mh(T, observations, N)
    t_stop = time()

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="Resimulation MH, N=$(N)")
        if save_plots savefig("images/resimulation_mh.png") end
        if show_plots display(plot!()) end
    end

    mean_score = logsumexp(scores) - log(N)
    return (t_stop - t_start), trs, scores, ones(N), data.β, mean_score, plot!()
end

run_block_mh = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    trs, scores, data = block_mh(T, observations, N)
    t_stop = time()

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="Block resimulation MH, N=$(N)")
        if save_plots savefig("images/block_mh.png") end
        if show_plots display(plot!()) end
    end

    mean_score = logsumexp(scores) - log(N)
    return (t_stop - t_start), trs, scores, ones(N), data.β, mean_score, plot!()
end

run_single_site_mh = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    trs, scores, data = single_site_mh(T, observations, N)
    t_stop = time()

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="Single site MH, N=$(N)")
        if save_plots savefig("images/single_site_mh.png") end
        if show_plots display(plot!()) end
    end

    mean_score = logsumexp(scores) - log(N)
    return (t_stop - t_start), trs, scores, ones(N), data.β, mean_score, plot!()
end

run_basic_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    pf_state = basic_smc(T, observations, N);
    t_stop = time()
    lml_est = log_ml_estimate(pf_state)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, ws, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="Vanilla SMC, N=$(N)")
        if save_plots savefig("images/basic_smc.png") end
        if show_plots display(plot!()) end
    end

    scores = collect(get_score.(trs))
    mean_score = logsumexp(sample(scores, StatsBase.weights(ws), N)) - log(N)
    return (t_stop - t_start), trs, scores, ws, βs, mean_score, plot!()
end

run_drift_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    pf_state = drift_smc(T, observations, N);
    t_stop = time()
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, ws, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="SMC with Gaussian drift rejuvenation, N=$(N)")
        if save_plots savefig("images/drift_smc.png") end
        if show_plots display(plot!()) end
    end

    scores = collect(get_score.(trs))
    mean_score = logsumexp(sample(scores, StatsBase.weights(ws), N)) - log(N)
    return (t_stop - t_start), trs, scores, ws, βs, mean_score, plot!()
end

run_data_driven_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    t_start = time()
    pf_state = data_driven_smc(T, observations, N);
    t_stop = time()
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    if show_plots || save_plots
        plot_obs(trace, [1]); plot_obs!(trs, ws, [1], lw=1, color=:grey)
        plot!(legend=:topleft, title="SMC with data-driven migration proposals, N=$(N)")
        if save_plots savefig("images/data_driven_smc.png") end
        if show_plots display(plot!()) end
    end

    scores = collect(get_score.(trs))
    mean_score = logsumexp(sample(scores, StatsBase.weights(ws), N)) - log(N)
    return (t_stop - t_start), trs, scores, ws, βs, mean_score, plot!()
end

run_experiments = (trace, T, repeats) -> begin
    algs = Dict(
        :basic_smc => (run_basic_smc, 100),
        :data_driven_smc => (run_data_driven_smc, 100),
        # :resimulation_mh => (run_resimulation_mh, 100)
    )
    df = DataFrame(alg = Symbol[], dur = Float64[], dur_std = Float64[],
                   score = Float64[], score_std = Float64[],
                   β_hat = Float64[], β_std = Float64[], β_rmse = Float64[])
    for (name, (alg, N)) in algs
        durs = Vector{Float64}(undef, repeats)
        scores = Vector{Float64}(undef, repeats)
        β_hats = Vector{Float64}(undef, repeats)
        for i in 1:repeats
            dur, _, _, ws, data, score, _ = alg(trace, T, N)
            durs[i], scores[i] = dur, score
            β_hats[i] = mean(data, StatsBase.weights(ws))
            GC.gc() # Reclaim unused memory
        end
        β_rmse = mean((β_hats .- trace[:β]).^2).^0.5
        push!(df, [name, mean(durs), std(durs), mean(scores), std(scores),
                   mean(β_hats), std(β_hats), β_rmse])
        println(df[end, :])
    end
    return df
end

# Generate ground truth trace.
trace, _ = generate(bayesian_sir, (50, 0.01), choicemap(:β => 0.5));
plot_obs(trace)

show_plots = false
save_plots = false

df = run_experiments(trace, 50, 1)
CSV.write("sir_inference.csv", df)

# dur, trs, scores, ws, data, plt = run_resimulation_mh(trace, 50, 100);
# dur, trs, scores, ws, data, plt = run_block_mh(trace, 50, 10);
# dur, trs, scores, ws, data, plt = run_single_site_mh(trace, 50, 100);
# dur, trs, scores, ws, data, plt = run_basic_smc(trace, 50, 100);
# dur, trs, scores, ws, data, plt = run_drift_smc(trace, 50, 100);
# dur, trs, scores, ws, data, plt = run_data_driven_smc(trace, 50, 100);

end # module
