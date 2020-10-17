module Inference

using Gen, Genify, GenParticleFilters, Distributions
using ProgressMeter: @showprogress

include("model.jl")
include("utils.jl")
include("inference/gaussian_drift.jl")
include("inference/resimulation_mh.jl")
include("inference/single_site_mh.jl")
include("inference/basic_smc.jl")
include("inference/drift_smc.jl")
include("inference/data_driven_smc.jl")

run_resimulation_mh = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    trs, scores, data = resimulation_mh(T, observations, N)
    plot_obs(trace, [1])
    plot_obs!(trs, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Resimulation MH, N=$(N)")
    savefig("images/resimulation_mh.png")
end

run_single_site_mh = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    trs, scores, data = single_site_mh(T, observations, N)
    plot_obs(trace, [1])
    plot_obs!(trs, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Single site MH, N=$(N)")
    savefig("images/single_site_mh.png")
end

run_basic_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    pf_state = basic_smc(T, observations, N);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1]; color=[:red :blue])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Vanilla SMC, N=$(N)")
    savefig("images/smc_no_migration.png")
end

run_drift_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    pf_state = drift_smc(T, observations, N);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="SMC with Gaussian drift rejuvenation, N=$(N)")
    savefig("images/smc_with_drift_rejuv.png")
end

run_data_driven_smc = (trace, T, N) -> begin
    observations = merge(case_count_obs(trace), location_obs(trace, frac=0.5));
    pf_state = data_driven_smc(T, observations, N);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Data-driven SMC with targeted Poisson, N=$(N)")
    savefig("images/data_driven_smc.png")
end

# Generate ground truth trace.
trace, _ = generate(bayesian_sir, (50, 0.01), choicemap(:β => 0.5));

run_single_site_mh(trace, 50, 100)
run_resimulation_mh(trace, 50, 100)
run_basic_smc(trace, 50, 100)
run_drift_smc(trace, 50, 100)
run_data_driven_smc(trace, 50, 100)

end # module
