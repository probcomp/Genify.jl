include("simulator.jl")

# ## Bayesian wrapper
using Gen, Genify, Distributions

trunc_normal(μ, σ, lb, ub) =
    Genify.WrappedDistribution(truncated(Normal(μ, σ), lb, ub))

# Set up fixed model parameters
function create_params(; β=0.3)
    Ns = [200; 200; 200]
    migration_rates = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
    β_det = β * ones(3)
    β_und = β_det ./ 10
    death_rate = 0.0
    return @dict(Ns, migration_rates, β_det, β_und, death_rate)
end

# Define observation model
@gen function observe(model::ABM, noise::Float64=5.0)
    agents = values(model.agents)
    obs = zeros(length(nodes(model)) * 2)
    for city in nodes(model)
        infected = count(a.status == :I && a.pos == city for a in agents)
        recovered = count(a.status == :R && a.pos == city for a in agents)
        obs[[city*2-1, city*2]] = [infected, recovered]
        {:infected => city} ~ trunc_normal(infected, noise, 0, Inf)()
        {:recovered => city} ~ trunc_normal(recovered, noise, 0, Inf)()
    end
    return obs
end

# Genify the Agents.step! method
abm_step! = genify(Agents.step!, AgentBasedModel, Any, Int, Bool; recurse=true)

# Define Unfold step
@gen function bayesian_sir_step(t::Int, _::Nothing, model::AgentBasedModel)
    {:agents} ~ abm_step!(model, agent_step!, 1, true)
    {:obs} ~ observe(model)
    return nothing
end

# Wrap SIR step model in Gen model with parameter uncertainty
@gen (static) function bayesian_sir(T::Int)
    β ~ uniform_continuous(0, 2.0)
    params = create_params(β=β)
    model = model_initiation(; params...)
    {:step} ~ Unfold(bayesian_sir_step)(T, nothing, model)
    return model
end

load_generated_functions()
