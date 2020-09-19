struct WrappedDistribution{T,D} <: Gen.Distribution{T} end
WrappedDistribution(d::Type{<:UnivariateDistribution}) =
    WrappedDistribution{eltype(d),d}()

(d::WrappedDistribution)() = Gen.random(d)

Gen.logpdf(d::WrappedDistribution{T,D}, x::T, args...) where {T,D} =
    Distributions.logpdf(D(args...), x)
Gen.logpdf_grad(d::WrappedDistribution{T,D}, x::T, args...) where {T,D} =
    tuple(nothing, (nothing for p in params(D(args...)))...)
Gen.random(d::WrappedDistribution{T,D}, args...) where {T,D} =
    rand(D(args...))
Gen.has_output_grad(::WrappedDistribution{T,D}) where {T,D} =
    false
Gen.has_argument_grads(::WrappedDistribution{T,D}) where {T,D} =
    tuple((nothing for p in params(D()))...)

Gen.is_discrete(::WrappedDistribution{Int,D}) where {D} = true
Gen.is_discrete(::WrappedDistribution{Float64,D}) where {D} = false
