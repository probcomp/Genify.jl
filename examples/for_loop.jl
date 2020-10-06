module ForLoop

using Random, Distributions, Gen
using Genify
using Genify: unwrap, transform!

# This hits a world age bug.

function foo(N::Int)
    for i in 1 : N
        for j in 1 : N
            x = rand(Normal(0.0, 1.0))
        end
    end
end

genfoo = genify(foo, Int; useslots=true)
choices, _, _ = propose(genfoo, (5, ))
display(choices)

end
