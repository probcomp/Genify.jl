using Test, Genify, IRTools
using Random, TracedRandom, StatsBase
using Distributions, Gen

using Genify: unwrap, transform!
using MacroTools: isexpr
using IRTools: IR

include("loops.jl")
include("transform.jl")
include("primitives.jl")
