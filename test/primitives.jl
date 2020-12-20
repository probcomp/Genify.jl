@testset "Random primitives" begin

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

@testset "randn() and randexp() statements" begin

function foo(dims...)
    a = randn()
    b = randexp()
    c = randn(Float64, dims)
    d = randexp(dims...)
end

genfoo = genify(foo)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Float64
@test trace[:b] isa Float64
@test trace[:c] isa Vector{Float64}
@test trace[:d] isa Vector{Float64}

# Test array sizes
@test size(trace[:a]) == ()
@test size(trace[:b]) == ()
@test size(trace[:c]) == (10,)
@test size(trace[:d]) == (10,)

# Test ranges
@test all(trace[:b] .>= 0)
@test all(trace[:d] .>= 0)

# Test generate with constraints
trace, weight = generate(genfoo, (), choicemap(:a => 1, :b => 1))
@test weight ≈ Gen.logpdf(normal, 0, 1, 1) + Gen.logpdf(exponential, 1, 1)
trace, weight = generate(genfoo, (10,), choicemap(:c => ones(10), :d => ones(10)))
@test weight ≈ 10 * (Gen.logpdf(normal, 0, 1, 1) + Gen.logpdf(exponential, 1, 1))

end

@testset "randperm(), randcycle(), and shuffle() statements" begin

function foo(n::Int)
    for i in 1:100
        a = randperm(n)
        b = randcycle(n)
    end
    c = shuffle([:a, :b, :c, :d, :e])
    d = shuffle(['h', 'e', 'l', 'l', 'o', 'o'])
end

genfoo = genify(foo, Int)

# Check types
trace = simulate(genfoo, (5,))
@test all(trace[:a => i] isa Vector{Int} for i in 1:100)
@test all(trace[:b => i] isa Vector{Int} for i in 1:100)
@test trace[:c] isa Vector{Symbol}
@test trace[:d] isa Vector{Char}

# Test array sizes
@test all(length(trace[:a => i]) == 5 for i in 1:100)
@test all(length(trace[:b => i]) == 5 for i in 1:100)
@test length(trace[:c]) == 5
@test length(trace[:d]) == 6

# Test ranges
@test all(isperm(trace[:a => i]) for i in 1:100)
@test all(Genify.iscycle(trace[:b => i]) for i in 1:100)
@test Set(trace[:c]) == Set([:a, :b, :c, :d, :e])
@test sort(trace[:d]) == ['e', 'h', 'l', 'l', 'o', 'o']

# Test generate with constraints
choices = choicemap((:a => 1, randperm(5)), (:b => 1, randcycle(5)))
trace, weight = generate(genfoo, (5,), choices)
@test weight ≈ log(1/factorial(5)) + log(1/factorial(4))
trace, weight = generate(genfoo, (5,), choicemap((:c,  [:e, :b, :d, :c, :a])))
@test weight == log(1/factorial(5))
trace, weight = generate(genfoo, (5,), choicemap((:d,  ['h', 'l', 'l', 'o', 'e', 'o'])))
@test weight == log((factorial(2) * factorial(2)) / factorial(6))

trace, weight = generate(genfoo, (5,), choicemap((:a => 1,  ones(Int, 5))))
@test weight == -Inf
trace, weight = generate(genfoo, (5,), choicemap((:b => 1,  [1, 2, 3, 4, 5])))
@test weight == -Inf
trace, weight = generate(genfoo, (5,), choicemap((:c,  [:e, :b, :d, :c, :f])))
@test weight == -Inf
trace, weight = generate(genfoo, (5,), choicemap((:d,  ['h', 'l', 'l', 'e', 'o'])))
@test weight == -Inf

end

@testset "sample() statements" begin

function foo()
    a = sample([:h, :e, :l, :l, :o], weights([1, 1, 5, 1, 1]))
    b = sample(collect("world"), (10, 10))
end

genfoo = genify(foo)

# Check types
trace = simulate(genfoo, ())
@test trace[:a] isa Symbol
@test trace[:b] isa Matrix{Char}

# Test array sizes
trace = simulate(genfoo, ())
@test size(trace[:b]) == (10, 10)

# Test ranges
@test trace[:a] in [:h, :e, :l, :l, :o]
@test all(x in "world" for x in trace[:b])

# Test generate with constraints
trace, weight = generate(genfoo, (), choicemap(:a => :l))
@test weight ≈ log(6/9)
trace, weight = generate(genfoo, (), choicemap(:b => fill('w', (10, 10))))
@test weight ≈ log(1/5) * 100

end

@testset "Primitives with RNGs" begin

function foo(n::Int)
    rng = MersenneTwister(0)
    a = rand(rng, Int8)
    b = rand(rng, Float32, n)
    c = rand(rng, "world")
    d = rand(rng, Wishart(5, [1.0 0.0; 0.0 1.0]))
    e = randexp(rng)
    f = randn(rng)
    g = randperm(rng, n)
    h = randcycle(rng, n)
end

genfoo = genify(foo, Int)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Int8
@test trace[:b] isa Vector{Float32}
@test trace[:c] isa Char
@test trace[:d] isa Matrix{Float64}
@test trace[:e] isa Float64
@test trace[:f] isa Float64
@test trace[:g] isa Vector{Int}
@test trace[:h] isa Vector{Int}

end

@testset "In-place primitives" begin

function foo(A::Vector{Float64}, B::Vector{Int})
    a = rand!(A)
    b = randn!(A)
    c = randexp!(A)
    d = randperm!(B)
    e = randcycle!(B)
    f = shuffle!(B)
    g = sample!(B, A)
end

genfoo = genify(foo, Vector{Float64}, Vector{Int})

# Check that arrays were modified
A, B = Vector{Float64}(undef, 10), Vector{Int}(undef, 10)
trace = simulate(genfoo, (A, B))
@test A == trace[:g]
@test B == trace[:f]

# Check types
@test trace[:a] isa Vector{Float64}
@test trace[:b] isa Vector{Float64}
@test trace[:c] isa Vector{Float64}
@test trace[:d] isa Vector{Int}
@test trace[:e] isa Vector{Int}
@test trace[:f] isa Vector{Int}
@test trace[:g] isa Vector{Int}
@test trace[] isa Vector{Float64}

# Test ranges
@test all(0 .<= trace[:a] .<= 1)
@test all(-Inf .<= trace[:b] .<= Inf)
@test all(0 .<= trace[:c] .<= Inf)
@test isperm(trace[:d])
@test Genify.iscycle(trace[:e])
@test isperm(trace[:f])
@test all(1 .<= trace[:g] .<= 10)

end

end
