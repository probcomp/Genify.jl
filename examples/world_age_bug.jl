module WorldAgeBug

using Random, Distributions, Gen
include("../src/Genify.jl")
using .Genify
using .Genify: unwrap, transform!

# This hits a world age bug.

function foo()
    x = [1.0, 3.0, 5.0] .+ rand(Normal(0.0, 1.0))
end

genfoo = genify(foo; useslots=true)
choices, _, _ = propose(genfoo, ())
display(choices)

end
