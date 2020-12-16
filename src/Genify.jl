module Genify

export genify, genified

using Gen, Distributions, StatsBase, Random, IRTools
using IRTools:
    IR, Block, Branch, Variable,
    arguments, argument!, deletearg!, recurse!, xcall, @dynamo,
    block, blocks, block!, deleteblock!, branch!, branches,
    successors, predecessors, dominators

const Address = Union{Symbol, Pair{Symbol}}
const Setlike = Union{AbstractSet, AbstractDict}
const Indexable = Union{AbstractArray, Tuple, AbstractString}
struct ManualAddress end # Placeholder for manual addressing scheme

include("utils.jl")
include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
