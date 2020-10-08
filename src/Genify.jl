module Genify

export genify

using Gen, Distributions, StatsBase, Random
using MacroTools, IRTools

# Loop detection.
using IRTools: domtree, blocks, branches, Block, IR

const Address = Union{Symbol, Pair{Symbol}}
const Setlike = Union{AbstractSet, AbstractDict}
const Indexable = Union{AbstractArray, Tuple, AbstractString}

include("utils.jl")
include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
