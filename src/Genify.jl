module Genify

export genify

using Gen, Distributions, StatsBase, Random, IRTools
using IRTools:
    IR, Branch, arguments, argument!, deletearg!, recurse!, xcall, @dynamo,
    block, blocks, block!, deleteblock!, branch!, branches,
    successors, predecessors, dominators

const Address = Union{Symbol, Pair{Symbol}}
const Setlike = Union{AbstractSet, AbstractDict}
const Indexable = Union{AbstractArray, Tuple, AbstractString}

include("utils.jl")
include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
