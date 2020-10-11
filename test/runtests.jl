using Test, Genify, IRTools
using Random, StatsBase, Distributions, Gen

using Genify: unwrap, transform!
using MacroTools: isexpr
using IRTools: IR

include("loops.jl")
include("transform.jl")
include("primitives.jl")
