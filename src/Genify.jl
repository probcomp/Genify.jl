module Genify

export genify

using Gen, Distributions, Random
using MacroTools
using IRTools

# Loop detection.
using IRTools: domtree, blocks, branches, Block, IR

const Address = Union{Symbol, Pair{Symbol}}

include("utils.jl")
include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
