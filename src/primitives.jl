## Rewrites random primitives as Gen distributions
const randprims = Set([
    rand, randn, randexp, randperm, shuffle, sample
])

## rand ##

# Strip away RNGs supplied to rand
@inline trace(options::Options, state, addr::Address, ::typeof(rand), rng::AbstractRNG, args...) =
    trace(options, state, addr::Address, rand, args...)

# Default to uniformly sampling Float64 values from [0, 1]
@inline trace(options::Options, state, addr::Address, ::typeof(rand)) =
    trace(options, state, addr::Address, rand, Float64)
@inline trace(options::Options, state, addr::Address, ::typeof(rand), d::Integer, dims::Integer...) =
    trace(options, state, addr::Address, rand, Float64, d, dims...)

# rand for numeric types
@inline trace(::Options, state, addr::Address, ::typeof(rand), T::Type{<:Real}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), T::Type{<:Real}, ::Tuple{}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), T::Type{<:Real}, dims::Dims) =
    Gen.traceat(state, TypedArrayDistribution(T, dims), (), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), T::Type{<:Real}, d::Integer, dims::Integer...) =
    Gen.traceat(state, TypedArrayDistribution(T, d, dims...), (), addr)

# rand for indexable collections
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T) where {T <: Indexable} =
    Gen.traceat(state, LabeledUniform{eltype(T)}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, ::Tuple{}) where {T <: Indexable} =
    Gen.traceat(state, LabeledUniform{eltype(T)}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, dims::Dims) where {T <: Indexable} =
    Gen.traceat(state, ArrayedDistribution(LabeledUniform{eltype(T)}(), dims), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, d::Integer, dims::Integer...) where {T <: Indexable} =
    Gen.traceat(state, ArrayedDistribution(LabeledUniform{eltype(T)}(), d, dims...), (c,), addr)

# rand for set-like collections
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T) where {T <: Setlike} =
    Gen.traceat(state, SetUniform{eltype(T)}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, ::Tuple{}) where {T <: Setlike} =
    Gen.traceat(state, SetUniform{eltype(T)}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, dims::Dims) where {T <: Setlike} =
    Gen.traceat(state, ArrayedDistribution(SetUniform{eltype(T)}(), dims), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::T, d::Integer, dims::Integer...) where {T <: Setlike} =
    Gen.traceat(state, ArrayedDistribution(SetUniform{eltype(T)}(), d, dims...), (c,), addr)

# rand for Distributions.jl distributions
@inline trace(::Options, state, addr::Address, ::typeof(rand), dist::D) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(dist), params(dist), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), dist::D, ::Tuple{}) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(dist), params(dist), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), dist::D, dims::Dims) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), dims), params(dist), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), dist::D, d::Integer, dims::Integer...) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), d, dims...), params(dist), addr)

## randn ##

# Strip away RNGs supplied to randn
@inline trace(options::Options, state, addr::Address, ::typeof(randn), rng::AbstractRNG, args...) =
    trace(options, state, addr::Address, randn, args...)

# Default to sampling Float64 values
@inline trace(options::Options, state, addr::Address, ::typeof(randn)) =
    trace(options, state, addr::Address, randn, Float64)
@inline trace(options::Options, state, addr::Address, ::typeof(randn), dims::Dims) =
    trace(options, state, addr::Address, randn, Float64, dims)
@inline trace(options::Options, state, addr::Address, ::typeof(randn), d::Integer, dims::Integer...) =
    trace(options, state, addr::Address, randn, Float64, d, dims...)

# Forward sampling of all other Float types
@inline trace(::Options, state, addr::Address, ::typeof(randn), T::Type{<:AbstractFloat}, args...) =
    trace(options, state, addr::Address, randn, Float64, args...)

# Sampling of Float64 values
@inline trace(::Options, state, addr::Address, ::typeof(randn), T::Type{Float64}) =
    Gen.traceat(state, Gen.normal, (0, 1), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randn), T::Type{Float64}, ::Tuple{}) =
    Gen.traceat(state, Gen.normal, (0, 1), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randn), T::Type{Float64}, dims::Dims) =
    Gen.traceat(state, ArrayedDistribution(Gen.normal, dims), (0, 1), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randn), T::Type{Float64}, d::Integer, dims::Integer...) =
    Gen.traceat(state, ArrayedDistribution(Gen.normal, d, dims...), (0, 1), addr)

## randexp ##

# Strip away RNGs supplied to randexp
@inline trace(options::Options, state, addr::Address, ::typeof(randexp), rng::AbstractRNG, args...) =
    trace(options, state, addr::Address, randexp, args...)

# Default to sampling Float64 values
@inline trace(options::Options, state, addr::Address, ::typeof(randexp)) =
    trace(options, state, addr::Address, randexp, Float64)
@inline trace(options::Options, state, addr::Address, ::typeof(randexp), dims::Dims) =
    trace(options, state, addr::Address, randexp, Float64, dims)
@inline trace(options::Options, state, addr::Address, ::typeof(randexp), d::Integer, dims::Integer...) =
    trace(options, state, addr::Address, randexp, Float64, d, dims...)

# Forward sampling of all other Float types
@inline trace(::Options, state, addr::Address, ::typeof(randexp), T::Type{<:AbstractFloat}, args...) =
    trace(options, state, addr::Address, randexp, Float64, args...)

# Sampling of Float64 values
@inline trace(::Options, state, addr::Address, ::typeof(randexp), T::Type{Float64}) =
    Gen.traceat(state, Gen.exponential, (1,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randexp), T::Type{Float64}, ::Tuple{}) =
    Gen.traceat(state, Gen.exponential, (1,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randexp), T::Type{Float64}, dims::Dims) =
    Gen.traceat(state, ArrayedDistribution(Gen.exponential, dims), (1,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(randexp), T::Type{Float64}, d::Integer, dims::Integer...) =
    Gen.traceat(state, ArrayedDistribution(Gen.exponential, d, dims...), (1,), addr)

## sample ##

# Strip away RNGs supplied to randexp
@inline trace(options::Options, state, addr::Address, ::typeof(sample), rng::AbstractRNG, args...) =
    trace(options, state, addr::Address, sample, args...)

# Unweighted sample
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}) where {T} =
    Gen.traceat(state, LabeledUniform{T}(), (a,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}, n::Integer) where {T} =
    Gen.traceat(state, ArrayedDistribution(LabeledUniform{T}(), n), (a,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}, dims::Dims) where {T} =
    Gen.traceat(state, ArrayedDistribution(LabeledUniform{T}(), dims), (a,), addr)

# Weighted sample
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}, w::AbstractWeights) where {T} =
    Gen.traceat(state, LabeledCategorical{T}(), (a, w), addr)
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}, w::AbstractWeights, n::Integer) where {T} =
    Gen.traceat(state, ArrayedDistribution(LabeledCategorical{T}(), n), (a, w), addr)
@inline trace(::Options, state, addr::Address, ::typeof(sample), a::AbstractArray{T}, w::AbstractWeights, dims::Dims) where {T} =
    Gen.traceat(state, ArrayedDistribution(LabeledCategorical{T}(), dims), (a, w), addr)
