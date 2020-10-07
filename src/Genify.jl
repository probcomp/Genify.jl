module Genify

export genify

using Gen, Distributions, Random
using MacroTools, IRTools

const Address = Union{Symbol, Pair{Symbol}}
const Setlike = Union{AbstractSet, AbstractDict}
const Indexable = Union{AbstractArray, Tuple, AbstractString}

include("transform.jl")
include("distributions.jl")
include("primitives.jl")

end
