module Genify

export genify

using Gen, Distributions, StatsBase, Random, IRTools
using IRTools:
    IR, arguments, argument!, deletearg!, recurse!, xcall, @dynamo,
    dominators, block ,blocks, successors, predecessors, deleteblock!    

const Address = Union{Symbol, Pair{Symbol}}
const Setlike = Union{AbstractSet, AbstractDict}
const Indexable = Union{AbstractArray, Tuple, AbstractString}

include("utils.jl")
include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
