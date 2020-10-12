using Gen, Genify

include("model.jl")

# Genify the Agents.step! method
genstep! = genify(Agents.step!, ABM, Any, Any, Int)

# Define observation model
@gen function observe(model::ABM, noise::Float64=5.0)
    agents = collect(values(model.agents))
    # TODO: Observe counts for each city
    alive ~ normal(length(agents), noise)
    infected ~ normal(count(a.status == :I for a in agents), noise)
    recovered ~ normal(count(a.status == :R for a in agents), noise)
    return alive, infected, recovered
end

# Wrap in Gen model with parameter uncertainty
@gen function bayesian_sir(n::Int)
    populations = [200; 200; 200]
    migration_rates = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
    β ~ uniform_continuous()
    β_det = β * ones(3)
    β_und = β_det ./ 10
    model = model_initiation(Ns=populations, migration_rates=migration_rates,
                             β_det=β_det, β_und=β_und)
    for i = 1:n
        {:step => i} ~ genstep!(model, agent_step!)
        {:obs => i} ~ observe(model)
    end
    return model
end
