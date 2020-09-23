"Wraps Distributions.jl distributions as Gen.jl distributions."
struct WrappedDistribution{T,D <: Distributions.Distribution} <: Gen.Distribution{T}
    dist::D
end
WrappedDistribution(d::D) where {D <: UnivariateDistribution} =
    WrappedDistribution{eltype(D),D}(d)
WrappedDistribution(d::D) where {D <: MultivariateDistribution} =
    WrappedDistribution{Vector{eltype(D)},D}(d)
WrappedDistribution(d::D) where {D <: MatrixDistribution} =
    WrappedDistribution{Matrix{eltype(D)},D}(d)

(d::WrappedDistribution)(args...) = Gen.random(d, args...)

Gen.random(d::WrappedDistribution{T,D}, args...) where {T,D} =
    rand(d.dist)
Gen.logpdf(d::WrappedDistribution{T,D}, x, args...) where {T,D} =
    Distributions.logpdf(d.dist, x)
Gen.logpdf_grad(d::WrappedDistribution{T,D}, x, args...) where {T,D} =
    tuple(nothing, (nothing for p in params(d.dist))...)
Gen.has_output_grad(::WrappedDistribution{T,D}) where {T,D} =
    false
Gen.has_argument_grads(::WrappedDistribution{T,D}) where {T,D} =
    tuple((false for p in params(D()))...)

Gen.is_discrete(::WrappedDistribution) = false
Gen.is_discrete(::WrappedDistribution{T,D}) where {T,D <: DiscreteDistribution} = true

"Repeats a distribution across an array of arbitrary dimensions."
struct ArrayedDistribution{T} <: Gen.Distribution{T}
    dist::Gen.Distribution
    dims::Dims
end
ArrayedDistribution(D::Type{<:Gen.Distribution{T}}, dims::Dims) where {T} =
    ArrayedDistribution{Array{T,length(dims)}}(D(), dims)
ArrayedDistribution(D::Type{<:Gen.Distribution{T}}, dims...) where {T} =
    ArrayedDistribution{Array{T,length(dims)}}(D(), dims)
ArrayedDistribution(d::Gen.Distribution{T}, dims::Dims) where {T} =
    ArrayedDistribution{Array{T,length(dims)}}(d, dims)
ArrayedDistribution(d::Gen.Distribution{T}, dims...) where {T} =
    ArrayedDistribution{Array{T,length(dims)}}(d, dims)

(d::ArrayedDistribution)(args...) = Gen.random(d, args...)

Gen.random(d::ArrayedDistribution{T}, args...) where {T} =
    broadcast(x -> Gen.random(d.dist, args...), zeros(d.dims...))
Gen.logpdf(d::ArrayedDistribution{T}, x::AbstractArray, args...) where {T} =
    size(x) != d.dims ? throw(DimensionMismatch("size should be $(d.dims)")) :
    sum(broadcast(y -> Gen.logpdf(d.dist, y, args...), x))
Gen.logpdf_grad(d::ArrayedDistribution{T}, x::AbstractArray, args...) where {T} =
    size(x) != d.dims ? throw(DimensionMismatch("size should be $(d.dims)")) :
    tuple(nothing, (nothing for a in Gen.has_argument_grads(d.dist)...))
Gen.has_output_grad(::ArrayedDistribution{T}) where {T} =
    false
Gen.has_argument_grads(::ArrayedDistribution{T}) where {T} =
    (false for a in Gen.has_argument_grads(d.dist))

"Typed scalar distribution, equivalent to `rand(T)` for type `T`."
struct TypedScalarDistribution{T<:Real} <: Gen.Distribution{T} end
TypedScalarDistribution(T::Type) = TypedScalarDistribution{T}()

(d::TypedScalarDistribution)() = Gen.random(d)

Gen.random(::TypedScalarDistribution{T}) where {T} =
    rand(T)

# Integers are sampled uniformly from [typemin(T), typemax(T)]
Gen.logpdf(d::TypedScalarDistribution{T}, x::Integer) where {T <: Integer} =
    typemin(T) <= x <= typemax(T) ? (-sizeof(T) * 8 * log(2)) : -Inf
# Floats are sampled uniformly from (0, 1)
Gen.logpdf(d::TypedScalarDistribution{T}, x::Real) where {T <: AbstractFloat} =
    0 <= x <= 1 ? 0.0 : -Inf

Gen.logpdf_grad(::TypedScalarDistribution, x) =
    (nothing,)
Gen.has_output_grad(::TypedScalarDistribution) =
    false
Gen.has_argument_grads(::TypedScalarDistribution) =
    ()

"Typed array distribution, equivalent to `rand(T, dims...)` for type `T`."
struct TypedArrayDistribution{T<:Real,N} <: Gen.Distribution{Array{T,N}}
    dims::Dims{N}
end
TypedArrayDistribution(T::Type, dims::Dims{N}) where {N} =
    TypedArrayDistribution{T,N}(dims)
TypedArrayDistribution(T::Type, dims...) =
    TypedArrayDistribution{T,length(dims)}(dims)

(d::TypedArrayDistribution)() = Gen.random(d)

Gen.random(d::TypedArrayDistribution{T}) where {T} =
    rand(T, d.dims...)

# Integers are sampled uniformly from [typemin(T), typemax(T)]
Gen.logpdf(d::TypedArrayDistribution{T}, x::AbstractArray{<:Integer}) where {T <: Integer} =
    size(x) != d.dims ? throw(DimensionMismatch("size should be $(d.dims)")) :
    all(typemin(T) .<= x .<= typemax(T)) ? -sizeof(T)*8*log(2)*prod(d.dims) : -Inf
# Floats are sampled uniformly from (0, 1)
Gen.logpdf(d::TypedArrayDistribution{T}, x::AbstractArray{<:Real}) where {T <: AbstractFloat} =
    size(x) != d.dims ? throw(DimensionMismatch("size should be $(d.dims)")) :
    all(0 .<= x .<= 1) ? 0.0 : -Inf

Gen.logpdf_grad(::TypedArrayDistribution, x) =
    (nothing,)
Gen.has_output_grad(::TypedArrayDistribution) =
    false
Gen.has_argument_grads(::TypedArrayDistribution) =
    ()

"Labeled uniform distribution over some indexable collection."
@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

"Labeled uniform distribution over set-like collections."
struct SetUniformDistribution{T} <: Gen.Distribution{T} end
SetUniformDistribution() = SetUniformDistribution{Any}()

(d::SetUniformDistribution)(args...) = Gen.random(d, args...)

Gen.random(::SetUniformDistribution{T}, support::AbstractSet{T}) where {T} =
    rand(support)
Gen.random(::SetUniformDistribution{Pair{T,U}}, support::AbstractDict{T,U}) where {T,U} =
    rand(support)

Gen.logpdf(d::SetUniformDistribution{T}, x::T, support::AbstractSet{T}) where {T} =
    x in support ? -log(length(support)) : -Inf
Gen.logpdf(d::SetUniformDistribution{Pair{T,U}}, x::Pair{T,U}, support::AbstractDict{T,U}) where {T,U} =
    x in support ? -log(length(support)) : -Inf

Gen.logpdf_grad(::SetUniformDistribution, x, support) =
    (nothing, nothing)
Gen.has_output_grad(::SetUniformDistribution) =
    false
Gen.has_argument_grads(::SetUniformDistribution) =
    (nothing,)
