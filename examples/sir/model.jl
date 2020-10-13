# # SIR model for the spread of COVID-19
# Adapted from the Agents.jl repository:
# https://github.com/JuliaDynamics/Agents.jl/blob/master/examples/sir.jl

# This example illustrates how to use `GraphSpace` and how to model agents on an graph
# (network) where the transition probabilities between each node (position) is not constant.
# ## SIR model

# A SIR model tracks the ratio of Susceptible, Infected, and Recovered individuals within a population.
# Here we add one more category of individuals: those who are infected, but do not know it.
# Transmission rate for infected and diagnosed individuals is lower than infected and undetected.
# We also allow a fraction of recovered individuals to catch the disease again, meaning
# that recovering the disease does not bring full immunity.

# ## Model parameters
# Here are the model parameters, some of which have default values.
# * `Ns`: a vector of population sizes per city. The amount of cities is just `C=length(Ns)`.
# * `β_und`: a vector for transmission probabilities β of the infected but undetected per city.
#   Transmission probability is how many susceptible are infected per day by an infected individual.
#   If social distancing is practiced, this number increases.
# * `β_det`: an array for transmission probabilities β of the infected and detected per city.
#   If hospitals are full, this number increases.
# * `infection_period = 30`: how many days before a person dies or recovers.
# * `detection_time = 14`: how many days before an infected person is detected.
# * `death_rate = 0.02`: the probability that the individual will die after the `infection_period`.
# * `reinfection_probability = 0.05`: The probability that a recovered person can get infected again.
# * `migration_rates`: A matrix of migration probability per individual per day from one city to another.
# * `Is = [zeros(C-1)..., 1]`: An array for initial number of infected but undetected people per city.
#   This starts as only one infected individual in the last city.

# Notice that `Ns, β, Is` all need to have the same length, as they are numbers for each
# city. We've tried to add values to the infection parameters similar to the ones you would hear
# on the news about COVID-19.

# The good thing with Agent based models is that you could easily extend the model we
# implement here to also include age as an additional property of each agent.
# This makes ABMs flexible and suitable for research of virus spreading.

# ## Making the model in Agents.jl
# We start by defining the `PoorSoul` agent type and the ABM
# cd(@__DIR__) #src
using Agents, Random, DataFrames, LightGraphs
using Distributions: Poisson, DiscreteNonParametric
using DrWatson: @dict
using Plots
# gr() # hide

mutable struct PoorSoul <: AbstractAgent
    id::Int
    pos::Int
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initiation(;
    Ns,
    migration_rates,
    β_und,
    β_det,
    infection_period = 30,
    reinfection_probability = 0.05,
    detection_time = 14,
    death_rate = 0.02,
    Is = [zeros(Int, length(Ns) - 1)..., 1],
    seed = 0,
)

    # Random.seed!(seed)
    @assert length(Ns) ==
    length(Is) ==
    length(β_und) ==
    length(β_det) ==
    size(migration_rates, 1) "length of Ns, Is, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    C = length(Ns)
    ## normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims = 2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    properties = @dict(
        Ns,
        Is,
        β_und,
        β_det,
        β_det,
        migration_rates,
        infection_period,
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        death_rate
    )
    space = GraphSpace(complete_digraph(C))
    model = ABM(PoorSoul, space; properties = properties)

    ## Add initial individuals
    for city in 1:C, n in 1:Ns[city]
        ind = add_agent!(city, model, 0, :S) # Susceptible
    end
    ## add infected individuals
    for city in 1:C
        inds = get_node_contents(city, model)
        for n in 1:Is[city]
            agent = model[inds[n]]
            agent.status = :I # Infected
            agent.days_infected = 1
        end
    end
    return model
end
nothing # hide

# We will make a function that starts a model with `C` number of cities,
# and creates the other parameters automatically by attributing some random
# values to them. You could directly use the above constructor and specify all
# `Ns, β`, etc. for a given set of cities.

# All cities are connected with each other, while it is more probable to travel from a city
# with small population into a city with large population.

using LinearAlgebra: diagind

# function create_params(;
#     C,
#     max_travel_rate,
#     infection_period = 30,
#     reinfection_probability = 0.05,
#     detection_time = 14,
#     death_rate = 0.02,
#     Is = [zeros(Int, C - 1)..., 1],
#     seed = 19,
# )
#
#     Random.seed!(seed)
#     Ns = rand(50:5000, C)
#     β_und = rand(0.3:0.02:0.6, C)
#     β_det = β_und ./ 10
#
#     Random.seed!(seed)
#     migration_rates = zeros(C, C)
#     for c in 1:C
#         for c2 in 1:C
#             migration_rates[c, c2] = (Ns[c] + Ns[c2]) / Ns[c]
#         end
#     end
#     maxM = maximum(migration_rates)
#     migration_rates = (migration_rates .* max_travel_rate) ./ maxM
#     migration_rates[diagind(migration_rates)] .= 1.0
#
#     params = @dict(
#         Ns,
#         β_und,
#         β_det,
#         migration_rates,
#         infection_period,
#         reinfection_probability,
#         detection_time,
#         death_rate,
#         Is
#     )
#
#     return params
# end

function agent_step!(agent, model)
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function migrate!(agent, model)
    pid = agent.pos
    d = DiscreteNonParametric(1:(model.C), model.migration_rates[pid, :])
    m = rand(d)
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    d = Poisson(rate)
    n = rand(d)
    n == 0 && return

    for contactID in get_node_contents(agent.pos, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand() ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n == 0 && return
        end
    end
end

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand() ≤ model.death_rate
            kill_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

# ## Bayesian wrapper
using Gen, Genify, Distributions

# Genify the Agents.step! method
genstep! = genify(Agents.step!, ABM, Any; recurse=true)

# Set up fixed model parameters
function create_params(; β=0.3)
    Ns = [200; 200; 200]
    migration_rates = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9]
    β_det = β * ones(3)
    β_und = β_det ./ 10
    death_rate = 0.0
    return @dict(Ns, migration_rates, β_det, β_und, death_rate)
end

# Define observation model
@gen function observe(model::ABM, noise::Float64=5.0)
    agents = collect(values(model.agents))
    for city in nodes(model)
        alive = length(get_node_contents(city, model))
        infected = count(a.status == :I && a.pos == city &&
            a.days_infected >= model.detection_time for a in agents)
        recovered = count(a.status == :R && a.pos == city for a in agents)
        {:alive => city} ~ normal(alive, noise)
        {:infected => city} ~ normal(infected, noise)
        {:recovered => city} ~ normal(recovered, noise)
    end
    return model
end

# Wrap SIR step model in Gen model with parameter uncertainty
@gen function bayesian_sir(T::Int)
    β ~ uniform_continuous(0, 0.6)
    params = create_params(β=β)
    model = model_initiation(**params)
    for t = 1:T
        {:step => t} ~ genstep!(model, agent_step!)
        {:obs => t} ~ observe(model)
    end
    return model
end
