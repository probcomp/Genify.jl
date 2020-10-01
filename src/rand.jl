"Rewritten call to `rand` that supports traced execution in Gen."
# Strip away RNGs supplied to rand
genrand(state, addr, rng::AbstractRNG, args...) =
    genrand(state, addr, args...)

# rand for numeric types
genrand(state, addr) =
    genrand(state, addr, Float64)
genrand(state, addr, T::Type{<:Real}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
genrand(state, addr, T::Type{<:Real}, dims::Dims) =
    Gen.traceat(state, TypedArrayDistribution(T, dims), (), addr)
genrand(state, addr, T::Type{<:Real}, d::Integer, dims::Integer...) =
    Gen.traceat(state, TypedArrayDistribution(T, d, dims...), (), addr)

# rand for indexable collections
const Indexable = Union{AbstractArray, AbstractRange, Tuple, AbstractString}
flatten(c::Indexable) = c
flatten(c::AbstractArray) = vec(c)
genrand(state, addr, c::Indexable) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
genrand(state, addr, c::Indexable, dims::Dims) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, dims), (flatten(c),), addr)
genrand(state, addr, c::Indexable, d::Integer, dims::Integer...) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, d, dims...), (flatten(c),), addr)

# rand for set-like collections
genrand(state, addr, c::AbstractSet{T}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
genrand(state, addr, c::AbstractSet{T}, dims::Dims) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), dims), (c,), addr)
genrand(state, addr, c::AbstractSet{T}, d::Integer, dims::Integer...) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), d, dims...), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}, dims::Dims) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), dims), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}, d::Integer, dims::Integer...) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), d, dims...), (c,), addr)

# rand for Distributions.jl distributions
genrand(state, addr, dist::D) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(dist), params(dist), addr)
genrand(state, addr, dist::D, dims::Dims) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), dims), params(dist), addr)
genrand(state, addr, dist::D, d::Integer, dims::Integer...) where {D <: Distributions.Distribution} =
    Gen.traceat(state, ArrayedDistribution(WrappedDistribution(dist), d, dims...), params(dist), addr)
