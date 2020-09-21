using Test, Genify
using Random, Distributions, Gen

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
@test isapprox(weight, -8*log(2))
trace, weight = generate(genfoo, (10,), choicemap(:b => zeros(Int16, 10)))
@test isapprox(weight, -16*log(2) * 10)
trace, weight = generate(genfoo, (10,), choicemap(:c => zeros(Int16, 10, 10)))
@test isapprox(weight, -32*log(2) * 10 * 10)

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
