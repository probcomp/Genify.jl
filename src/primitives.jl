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
const Indexable = Union{AbstractArray, AbstractRange, Tuple, AbstractString}
flatten(c::Indexable) = c
flatten(c::AbstractArray) = vec(c)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::Indexable) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::Indexable, ::Tuple{}) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::Indexable, dims::Dims) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, dims), (flatten(c),), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::Indexable, d::Integer, dims::Integer...) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, d, dims...), (flatten(c),), addr)

# rand for set-like collections
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractSet{T}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractSet{T}, ::Tuple{}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractSet{T}, dims::Dims) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), dims), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractSet{T}, d::Integer, dims::Integer...) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), d, dims...), (c,), addr)

@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, ::Tuple{}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, dims::Dims) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), dims), (c,), addr)
@inline trace(::Options, state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, d::Integer, dims::Integer...) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), d, dims...), (c,), addr)

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
