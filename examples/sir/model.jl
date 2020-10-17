include("simulator.jl")

# ## Bayesian wrapper
using Gen, Genify, Distributions

trunc_normal(μ, σ, lb, ub) =
    Genify.WrappedDistribution(truncated(Normal(μ, σ), lb, ub))

# Set up fixed model parameters
function create_params(; β=0.3)
    Ns = [50; 50; 50]
    migration_rates = [0.95 0.04 0.01; 0.025 0.95 0.025; 0.01 0.04 0.95]
    β_det = β * ones(3)
    β_und = β_det ./ 10
    death_rate = 0.0
    Is = [10; 0; 0]
    return @dict(Ns, migration_rates, β_det, β_und, death_rate, Is)
end

# Define observation model
@gen function observe(model::ABM, noise::Float64=5.0)
    obs = zeros(length(nodes(model)) * 2)
    for city in nodes(model)
        infected = count(a.status == :I for a in get_node_agents(city, model))
        recovered = count(a.status == :R for a in get_node_agents(city, model))
        obs[[city*2-1, city*2]] = [infected, recovered]
        {:infected => city} ~ trunc_normal(infected, noise, 0, Inf)()
        {:recovered => city} ~ trunc_normal(recovered, noise, 0, Inf)()
    end
    for (i, agent) in model.agents
        probs = [c == agent.pos ? 0.95 : 0.05/(model.C-1) for c in 1:model.C]
        {:location => i} ~ Gen.categorical(probs)
    end
    return obs
end

# Genify the Agents.step! method
abm_step! = genify(Agents.step!, AgentBasedModel, Any, Int, Bool; recurse=true)

# Define Unfold step
@gen function bayesian_sir_step!(t::Int, _::Nothing,
                                model::AgentBasedModel, noise::Float64)
    {:agents} ~ abm_step!(model, agent_step!, 1, true)
    {:obs} ~ observe(model, noise)
    return nothing
end

# Wrap SIR step model in Gen model with parameter uncertainty
@gen (static) function bayesian_sir(T::Int, noise::Float64)
    β ~ uniform_continuous(0, 2.0)
    params = create_params(β=β)
    model = model_initiation(; params...)
    {:step} ~ Unfold(bayesian_sir_step!)(T, nothing, model, noise)
    return model
end

load_generated_functions()

# Test run the model
params = create_params()
model = model_initiation(; params...)
step!(model, agent_step!, 1, true)
abm_step!(model, agent_step!, 1, true)
