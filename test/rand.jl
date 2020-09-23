@testset "rand(::Type{<:Integer}) statements" begin

function foo(n::Int)
    a = rand(Int8)
    b = rand(Int16, n)
    c = rand(Int32, (n, n))
end

genfoo = genify(foo, Int)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Int8
@test trace[:b] isa Vector{Int16}
@test trace[:c] isa Matrix{Int32}

# Test array sizes
@test size(trace[:a]) == ()
@test size(trace[:b]) == (10,)
@test size(trace[:c]) == (10, 10)

# Test sample range
@test all(typemin(Int8) .<= trace[:a] .<= typemax(Int8))
@test all(typemin(Int16) .<= trace[:b] .<= typemax(Int16))
@test all(typemin(Int32) .<= trace[:c] .<= typemax(Int32))

# Test generate with constraints
trace, weight = generate(genfoo, (10,), choicemap(:a => 0))
@test weight ≈ -8*log(2)
trace, weight = generate(genfoo, (10,), choicemap(:b => zeros(Int16, 10)))
@test weight ≈ -16*log(2) * 10
trace, weight = generate(genfoo, (10,), choicemap(:c => zeros(Int16, 10, 10)))
@test weight ≈ -32*log(2) * 10 * 10

end

@testset "rand(::Type{<:AbstractFloat}) statements" begin

function foo(n::Int)
    a = rand(Float16)
    b = rand(Float32, n)
    c = rand(Float64, n, n)
end

genfoo = genify(foo, Int)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Float16
@test trace[:b] isa Vector{Float32}
@test trace[:c] isa Matrix{Float64}

# Test array sizes
@test size(trace[:a]) == ()
@test size(trace[:b]) == (10,)
@test size(trace[:c]) == (10, 10)

# Test ranges
@test all(0 .<= trace[:a] .<= 1)
@test all(0 .<= trace[:b] .<= 1)
@test all(0 .<= trace[:c] .<= 1)

# Test generate with constraints
trace, weight = generate(genfoo, (10,), choicemap(:a => 0.5))
@test weight == 0
trace, weight = generate(genfoo, (10,), choicemap(:b => [-1; fill(0, 9)]))
@test weight == -Inf
trace, weight = generate(genfoo, (10,), choicemap(:c => rand(Float64, 10, 10)))
@test weight == 0

end

@testset "rand(::Indexable|Setlike) statements" begin

function foo(dims...)
    a = rand([:h, :e, :l, :l, :o], dims...)
    b = rand("world", dims...)
    c = rand(1:5, dims...)
    d = rand((9, 6, 42), dims...)
    e = rand(Set("hello"), dims...)
    f = rand(Dict(zip("world", "hello")), dims...)
end

genfoo = genify(foo)

# Check types
trace = simulate(genfoo, ())
@test trace[:a] isa Symbol
@test trace[:b] isa Char
@test trace[:c] isa Int
@test trace[:d] isa Int
@test trace[:e] isa Char
@test trace[:f] isa Pair{Char,Char}

# Test array sizes
trace = simulate(genfoo, (10, 10))
@test size(trace[:a]) == (10, 10)
@test size(trace[:b]) == (10, 10)
@test size(trace[:c]) == (10, 10)
@test size(trace[:d]) == (10, 10)
@test size(trace[:e]) == (10, 10)
@test size(trace[:f]) == (10, 10)

# Test ranges
@test all(x in [:h, :e, :l, :l, :o] for x in trace[:a])
@test all(x in "world" for x in trace[:b])
@test all(x in 1:5 for x in trace[:c])
@test all(x in (9, 6, 42) for x in trace[:d])
@test all(x in Set("hello") for x in trace[:e])
@test all(x in Dict(zip("world", "hello")) for x in trace[:f])

# Test generate with constraints
trace, weight = generate(genfoo, (), choicemap(:a => :l))
@test weight ≈ log(2/5)
trace, weight = generate(genfoo, (), choicemap(:b => 'w'))
@test weight ≈ log(1/5)
trace, weight = generate(genfoo, (), choicemap(:c => 3))
@test weight ≈ log(1/5)
trace, weight = generate(genfoo, (), choicemap(:d => 42))
@test weight ≈ log(1/3)
trace, weight = generate(genfoo, (), choicemap(:e => 'l'))
@test weight ≈ log(1/4)
trace, weight = generate(genfoo, (), choicemap(:f => ('o' => 'e')))
@test weight ≈ log(1/5)

end

@testset "rand(::Distributions.Distribution) statements" begin

function foo(dims...)
    a = rand(Dirichlet(5, 2))
    b = rand(Categorical(a))
    c = rand(Wishart(b+2, [1.0 0.0; 0.0 1.0]))
    d = rand(Normal(0, 1), dims...)
end

genfoo = genify(foo)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Vector{Float64}
@test trace[:b] isa Int
@test trace[:c] isa Matrix{Float64}
@test trace[:d] isa Vector{Float64}

# Test array sizes
@test size(trace[:a]) == (5,)
@test size(trace[:b]) == ()
@test size(trace[:c]) == (2, 2)
@test size(trace[:d]) == (10,)

# Test ranges
@test all(0 .<= trace[:a] .<= 1)
@test all(1 .<= trace[:b] .<= 5)
@test insupport(Wishart, trace[:c])

# Test generate with constraints
trace, weight = generate(genfoo, (), choicemap(:a => ones(5)/5))
@test weight ≈ Distributions.logpdf(Dirichlet(5, 2), ones(5)/5)
trace, weight = generate(genfoo, (), choicemap(:a => ones(5)/5, :b => 5))
@test weight ≈ Distributions.logpdf(Dirichlet(5, 2), ones(5)/5) + log(1/5)
trace, weight = generate(genfoo, (), choicemap(:c => [1 0; 0 1]))
@test weight ≈ Distributions.logpdf(Wishart(trace[:b]+2, [1 0; 0 1]), [1 0; 0 1])
trace, weight = generate(genfoo, (10, 10), choicemap(:d => zeros(10, 10)))
@test weight ≈ Gen.logpdf(normal, 0, 0, 1) * 100

end
