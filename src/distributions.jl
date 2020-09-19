"Wraps Distributions.jl distributions as Gen.jl distributions."
struct WrappedDistribution{T,D <: Distributions.Distribution} <: Gen.Distribution{T} end
WrappedDistribution(D::Type{<:UnivariateDistribution}) =
    WrappedDistribution{eltype(D),D}()
WrappedDistribution(D::Type{<:MultivariateDistribution}) =
    WrappedDistribution{AbstractVector{eltype(D)},D}()
WrappedDistribution(d::Type{<:MatrixDistribution}) =
    WrappedDistribution{AbstractMatrix{eltype(D)},D}()

(d::WrappedDistribution)(args...) = Gen.random(d, args...)

Gen.logpdf(d::WrappedDistribution{T,D}, x::T, args...) where {T,D} =
    Distributions.logpdf(D(args...), x)
Gen.logpdf_grad(d::WrappedDistribution{T,D}, x::T, args...) where {T,D} =
    tuple(nothing, (nothing for p in params(D(args...)))...)
Gen.random(d::WrappedDistribution{T,D}, args...) where {T,D} =
    rand(D(args...))
Gen.has_output_grad(::WrappedDistribution{T,D}) where {T,D} =
    false
Gen.has_argument_grads(::WrappedDistribution{T,D}) where {T,D} =
    tuple((false for p in params(D()))...)

Gen.is_discrete(::WrappedDistribution) = false
Gen.is_discrete(::WrappedDistribution{T,D}) where {T,D <: DiscreteDistribution} = true

"Repeats a distribution across an array of arbitrary dimensions."
struct ArrayDistribution{T,D <: Gen.Distribution} <: Gen.Distribution{T}
    dims::Vector{Int}
end
ArrayDistribution(D::Type{<:Gen.Distribution{T}}, dims) where {T} =
    ArrayDistribution{AbstractArray{T,length(dims)}, D}(collect(dims))
ArrayDistribution(d::Gen.Distribution{T}, dims) where {T} =
    ArrayDistribution{AbstractArray{T,length(dims)}, typeof(d)}(collect(dims))

(d::ArrayDistribution)(args...) = Gen.random(d, args...)

Gen.logpdf(d::ArrayDistribution{T,D}, x::T, args...) where {T,D} =
    sum(broadcast(y -> Gen.logpdf(D(), y, args...), x))
Gen.logpdf_grad(d::ArrayDistribution{T,D}, x::T, args...) where {T,D} =
    tuple(nothing, (nothing for a in Gen.has_argument_grads(D()))...)
Gen.random(d::ArrayDistribution{T,D}, args...) where {T,D} =
    broadcast(x -> Gen.random(D(), args...), zeros(d.dims...))
Gen.has_output_grad(::ArrayDistribution{T,D}) where {T,D} =
    false
Gen.has_argument_grads(::ArrayDistribution{T,D}) where {T,D} =
    (false for a in Gen.has_argument_grads(D()))
