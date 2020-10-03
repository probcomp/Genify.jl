"Rewritten call to `rand` that supports traced execution in Gen."
# Strip away RNGs supplied to rand
trace(state, addr::Address, ::typeof(rand), rng::AbstractRNG, args...) =
    trace(state, addr::Address, rand, args...)

# rand for numeric types
trace(state, addr::Address, ::typeof(rand)) =
    trace(state, addr::Address, rand, Float64)
trace(state, addr::Address, ::typeof(rand), T::Type{<:Real}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
trace(state, addr::Address, ::typeof(rand), T::Type{<:Real}, ::Tuple{}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
trace(state, addr::Address, ::typeof(rand), T::Type{<:Real}, dims::Dims) =
    Gen.traceat(state, TypedArrayDistribution(T, dims), (), addr)
trace(state, addr::Address, ::typeof(rand), T::Type{<:Real}, d::Integer, dims::Integer...) =
    Gen.traceat(state, TypedArrayDistribution(T, d, dims...), (), addr)

# rand for indexable collections
const Indexable = Union{AbstractArray, AbstractRange, Tuple, AbstractString}
flatten(c::Indexable) = c
flatten(c::AbstractArray) = vec(c)
trace(state, addr::Address, ::typeof(rand), c::Indexable) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
trace(state, addr::Address, ::typeof(rand), c::Indexable, ::Tuple{}) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
trace(state, addr::Address, ::typeof(rand), c::Indexable, dims::Dims) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, dims), (flatten(c),), addr)
trace(state, addr::Address, ::typeof(rand), c::Indexable, d::Integer, dims::Integer...) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, d, dims...), (flatten(c),), addr)

# rand for set-like collections
trace(state, addr::Address, ::typeof(rand), c::AbstractSet{T}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractSet{T}, ::Tuple{}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractSet{T}, dims::Dims) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), dims), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractSet{T}, d::Integer, dims::Integer...) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), d, dims...), (c,), addr)

trace(state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, ::Tuple{}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, dims::Dims) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), dims), (c,), addr)
trace(state, addr::Address, ::typeof(rand), c::AbstractDict{T,U}, d::Integer, dims::Integer...) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), d, dims...), (c,), addr)

# rand for Distributions.jl distributions
trace(state, addr::Address, ::typeof(rand), dist::D) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(dist), params(dist), addr)
trace(state, addr::Address, ::typeof(rand), dist::D, ::Tuple{}) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(dist), params(dist), addr)
trace(state, addr::Address, ::typeof(rand), dist::D, dims::Dims) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), dims), params(dist), addr)
trace(state, addr::Address, ::typeof(rand), dist::D, d::Integer, dims::Integer...) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), d, dims...), params(dist), addr)
