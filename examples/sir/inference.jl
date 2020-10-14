module Inference

using Gen, Genify, GenParticleFilters, Distributions
using ProgressMeter: @showprogress

include("model.jl")
include("utils.jl")
include("inference/resimulation_mh.jl")
include("inference/basic_smc.jl")
include("inference/drift_smc.jl")
include("inference/data_driven_smc.jl")

run_resimulation_mh = trace -> begin
    observations = aggregate_obs(trace)
    plot_obs(trace)
    trace, scores, data = resimulation_mh(50, observations, 100)
end

run_basic_smc = trace -> begin
    observations = aggregate_obs(trace)
    pf_state = smc_basic(50, observations, 50);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1]; color=[:red :blue])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Vanilla SMC, N=50")
    savefig("images/smc_no_migration.png")

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
    savefig("images/smc_with_migration.png")
end

run_drift_smc = trace -> begin
    observations = aggregate_obs(trace)
    pf_state = smc_drift(50, observations, 50);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="SMC with Gaussian drift rejuvenation, N=50")
    savefig("images/smc_with_drift_rejuv.png")
end

run_data_driven_smc = trace -> begin
    observations = aggregate_obs(trace)
    pf_state = data_driven_smc(50, observations, 50);
    lml_est = log_ml_estimate(pf_state)
    β_hat, β_var = mean(pf_state, :β), var(pf_state, :β)
    βs = [tr[:β] for tr in pf_state.traces]
    trs, ws = get_traces(pf_state), get_norm_weights(pf_state);

    plot_obs(trace, [1])
    plot_obs!(trs, ws, [1], lw=1, color=:grey)
    plot!(legend=:topleft, title="Data-driven SMC with targeted Poisson and Gaussian drift rejuvenation, N=50")
    savefig("images/data_driven_smc.png")
end

# Generate ground truth trace.
trace, _ = generate(bayesian_sir, (50, 0.01), choicemap(:β => 0.5));

run_resimulation_mh(trace)
run_basic_smc(trace)
run_drift_smc(trace)
run_data_driven_smc(trace)

end # module
