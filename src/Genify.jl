module Genify

export genify

using Gen, Distributions, Random
using MacroTools
using IRTools
using IRTools: domtree, blocks, branches

const Address = Union{Symbol, Pair{Symbol}}

include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
