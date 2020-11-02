using Gen, Genify
using Distributions: Normal

##  Define model of noisy weighing scale in Julia

function scale(n::Int)
    mass = rand(Normal(5, 1))
    obs = measure(mass, n)
end

function measure(mass::Real, n::Int)
    obs = Float64[]
    for i in 1:n
        m = rand(Normal(mass, 2))
        push!(obs, m)
    end
    return obs
end

## Genify the model

genscale = genify(scale, Int)

## Construct choicemap of noisy measurements

observations = choicemap(((:obs => :m => i, 4) for i in 1:10)...)

## Perform importance sampling to estimate the true mass

trs, ws, _ = importance_sampling(genscale, (10,), observations, 100)
mass_est = sum(tr[:mass] * exp(w) for (tr, w) in zip(trs, ws))
