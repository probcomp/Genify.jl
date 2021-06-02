import SpecialFunctions: logfactorial

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
WrappedDistribution(d::Truncated{D}) where {D} =
    WrappedDistribution{eltype(D),typeof(d)}(d)

(d::WrappedDistribution)(args...) = Gen.random(d, args...)

@inline Gen.random(d::WrappedDistribution{T,D}, args...) where {T,D} =
    rand(d.dist)
@inline Gen.logpdf(d::WrappedDistribution{T,D}, x, args...) where {T,D} =
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

@inline Gen.random(d::ArrayedDistribution{T}, args...) where {T} =
    broadcast(x -> Gen.random(d.dist, args...), zeros(d.dims...))
@inline Gen.logpdf(d::ArrayedDistribution{T}, x::AbstractArray, args...) where {T} =
    size(x) != d.dims ? error("size should be $(d.dims)") :
    sum(broadcast(y -> Gen.logpdf(d.dist, y, args...), x))
Gen.logpdf_grad(d::ArrayedDistribution{T}, x::AbstractArray, args...) where {T} =
    size(x) != d.dims ? error("size should be $(d.dims)") :
    tuple(nothing, (nothing for a in Gen.has_argument_grads(d.dist)...))
Gen.has_output_grad(::ArrayedDistribution{T}) where {T} =
    false
Gen.has_argument_grads(::ArrayedDistribution{T}) where {T} =
    (false for a in Gen.has_argument_grads(d.dist))

"Typed scalar distribution, equivalent to `rand(T)` for type `T`."
struct TypedScalarDistribution{T<:Real} <: Gen.Distribution{T} end
TypedScalarDistribution(T::Type) = TypedScalarDistribution{T}()

(d::TypedScalarDistribution)() = Gen.random(d)

@inline Gen.random(::TypedScalarDistribution{T}) where {T} =
    rand(T)

# Integers are sampled uniformly from [typemin(T), typemax(T)]
@inline Gen.logpdf(d::TypedScalarDistribution{T}, x::Integer) where {T <: Integer} =
    typemin(T) <= x <= typemax(T) ? (-sizeof(T) * 8 * log(2)) : -Inf
# Floats are sampled uniformly from (0, 1)
@inline Gen.logpdf(d::TypedScalarDistribution{T}, x::Real) where {T <: AbstractFloat} =
    0 <= x <= 1 ? 0.0 : -Inf

Gen.logpdf_grad(::TypedScalarDistribution{T}, x::Integer) where {T <: Integer} =
    (nothing,)
Gen.logpdf_grad(::TypedScalarDistribution{T}, x::Real) where {T <: AbstractFloat} =
    0.0

Gen.has_output_grad(::TypedScalarDistribution{T}) where {T <: Integer} =
    false
Gen.has_output_grad(::TypedScalarDistribution{T}) where {T <: AbstractFloat} =
    true

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

@inline Gen.random(d::TypedArrayDistribution{T}) where {T} =
    rand(T, d.dims...)

# Integers are sampled uniformly from [typemin(T), typemax(T)]
@inline Gen.logpdf(d::TypedArrayDistribution{T}, x::AbstractArray{<:Integer}) where {T <: Integer} =
    size(x) != d.dims ? error("size should be $(d.dims)") :
    all(typemin(T) .<= x .<= typemax(T)) ? -sizeof(T)*8*log(2)*prod(d.dims) : -Inf
# Floats are sampled uniformly from (0, 1)
@inline Gen.logpdf(d::TypedArrayDistribution{T}, x::AbstractArray{<:Real}) where {T <: AbstractFloat} =
    size(x) != d.dims ? error("size should be $(d.dims)") :
    all(0 .<= x .<= 1) ? 0.0 : -Inf

Gen.logpdf_grad(d::TypedArrayDistribution{T}, x::AbstractArray{<:Integer}) where {T <: Integer} =
    size(x) != d.dims ? error("size should be $(d.dims)") : (nothing,)
Gen.logpdf_grad(d::TypedArrayDistribution{T}, x::AbstractArray{<:Real}) where {T <: AbstractFloat} =
    size(x) != d.dims ? error("size should be $(d.dims)") :
    zeros(d.dims)

Gen.has_output_grad(::TypedArrayDistribution{T}) where {T <: Integer} =
    false
Gen.has_output_grad(::TypedArrayDistribution{T}) where {T <: AbstractFloat} =
    true

Gen.has_argument_grads(::TypedArrayDistribution) =
    ()

"Labeled uniform distribution over indexable collections."
struct LabeledUniform{T} <: Gen.Distribution{T} end
LabeledUniform() = LabeledUniform{Any}()

(d::LabeledUniform)(args...) = Gen.random(d, args...)

@inline Gen.random(::LabeledUniform{T}, labels::AbstractArray{T}) where {T} =
    sample(labels)
@inline Gen.random(::LabeledUniform{T}, labels::Tuple{Vararg{T}}) where {T} =
    rand(labels)
@inline Gen.random(::LabeledUniform{T}, labels::AbstractString) where {T <: AbstractChar} =
    rand(labels)
@inline Gen.logpdf(::LabeledUniform{T}, x::T, labels::Indexable) where {T} =
    log(sum(x == l for l in labels) / length(labels))

Gen.logpdf_grad(::LabeledUniform, x, labels) =
    (nothing, nothing)
Gen.has_output_grad(::LabeledUniform) =
    false
Gen.has_argument_grads(::LabeledUniform) =
    (false,)

"Labeled uniform distribution over set-like collections."
struct SetUniform{T} <: Gen.Distribution{T} end
SetUniform() = SetUniform{Any}()

(d::SetUniform)(args...) = Gen.random(d, args...)

@inline Gen.random(::SetUniform{T}, support::AbstractSet{T}) where {T} =
    rand(support)
@inline Gen.random(::SetUniform{Pair{T,U}}, support::AbstractDict{T,U}) where {T,U} =
    rand(support)

@inline Gen.logpdf(::SetUniform{T}, x::T, support::AbstractSet{T}) where {T} =
    x in support ? -log(length(support)) : -Inf
@inline Gen.logpdf(::SetUniform{Pair{T,U}}, x::Pair{T,U}, support::AbstractDict{T,U}) where {T,U} =
    x in support ? -log(length(support)) : -Inf

Gen.logpdf_grad(::SetUniform, x, support) =
    (nothing, nothing)
Gen.has_output_grad(::SetUniform) =
    false
Gen.has_argument_grads(::SetUniform) =
    (false,)

"Categorical distribution over an array of labels."
struct LabeledCategorical{T} <: Gen.Distribution{T} end
LabeledCategorical() = LabeledCategorical{Any}()

(d::LabeledCategorical)(args...) = Gen.random(d, args...)

@inline Gen.random(::LabeledCategorical{T}, labels::AbstractArray{T}, weights::AbstractWeights) where {T} =
    sample(labels, weights)
@inline Gen.logpdf(::LabeledCategorical{T}, x::T, labels::AbstractArray{T}, weights::AbstractWeights) where {T} =
    log(sum((x .== vec(labels)) .* Vector(weights))) - log(sum(weights))
@inline Gen.logpdf(::LabeledCategorical{T}, x::T, labels::AbstractArray{T}, weights::UnitWeights) where {T} =
    log(sum(x .== labels) / length(labels))

function Gen.logpdf_grad(::LabeledCategorical, x, labels, weights::AbstractWeights)
    grad = (x .== vec(labels)) .* (1. ./ Vector(weights)) .- 1.0 ./ sum(weights)
    return (nothing, nothing, grad)
end

Gen.has_output_grad(::LabeledCategorical) =
    false
Gen.has_argument_grads(::LabeledCategorical) =
    (false, true)

"Uniform distribution over permutations of length N."
struct RandomPermutation{T,N} <: Gen.Distribution{T} end
RandomPermutation(n::T) where {T <: Integer} = RandomPermutation{Vector{T},n}()

(d::RandomPermutation)() = Gen.random(d)

@inline Gen.random(::RandomPermutation{T,N}) where {T,N} =
    randperm(N)
@inline Gen.logpdf(::RandomPermutation{T,N}, x::T) where {T,N} =
    length(x) == N && isperm(x) ? -logfactorial(N) : -Inf

Gen.logpdf_grad(::RandomPermutation, x) =
    (nothing,)
Gen.has_output_grad(::RandomPermutation) =
    false
Gen.has_argument_grads(::RandomPermutation) =
    ()

"Uniform distribution over cyclic permutations of length N."
struct RandomCycle{T,N} <: Gen.Distribution{T} end
RandomCycle(n::T) where {T <: Integer} = RandomCycle{Vector{T},n}()

(d::RandomCycle)() = Gen.random(d)

function iscycle(v::Vector{<:Integer})
    n = length(v); seen = falses(n); i = 1
    for k in 1:n
        (0 < i <= n) && (seen[i] ⊻= true) || return false
        i = v[i]
    end
    return true
end

@inline Gen.random(::RandomCycle{T,N}) where {T,N} =
    randcycle(N)
@inline Gen.logpdf(::RandomCycle{T,N}, x::T) where {T,N} =
    length(x) == N && iscycle(x) ? -logfactorial(N-1) : -Inf

Gen.logpdf_grad(::RandomCycle, x) =
    (nothing,)
Gen.has_output_grad(::RandomCycle) =
    false
Gen.has_argument_grads(::RandomCycle) =
    ()

"Uniform distribution over permutations (shuffles) of an array."
struct RandomShuffle{T <: AbstractArray} <: Gen.Distribution{T}
    v::T # Original array to be shuffled
end

(d::RandomShuffle)() = Gen.random(d)

@inline Gen.random(d::RandomShuffle) =
    shuffle(d.v)
function Gen.logpdf(d::RandomShuffle{T}, xs::T) where {T}
    if size(xs) != size(d.v) return -Inf end
    v_counts = countmap(d.v)
    score = sum(logfactorial.(values(v_counts))) - logfactorial(length(d.v))
    for x in xs
        x in keys(v_counts) || return -Inf
        v_counts[x] -= 1
    end
    return all(values(v_counts) .== 0) ? score : -Inf
end

Gen.logpdf_grad(::RandomShuffle, x) =
    (nothing,)
Gen.has_output_grad(::RandomShuffle) =
    false
Gen.has_argument_grads(::RandomShuffle) =
    ()
