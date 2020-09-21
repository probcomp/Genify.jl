using Test, Genify
using Random, Distributions, Gen

@testset "rand(::Type{<:Integer}) statements" begin

function foo()
    a = rand(Int8)
    b = rand(Int16)
    c = rand(Int32)
end

genfoo = genify(foo)

# Check types
trace = simulate(genfoo, ())
@test trace[:a] isa Int8
@test trace[:b] isa Int16
@test trace[:c] isa Int32

# Test ranges for various Integer types
traces = [simulate(genfoo, ()) for i in 1:100]
@test all(typemin(Int8) .<= getindex.(traces, :a) .<= typemax(Int8))
@test all(typemin(Int16) .<= getindex.(traces, :b) .<= typemax(Int16))
@test all(typemin(Int32) .<= getindex.(traces, :c) .<= typemax(Int32))

# Test generate with constraints inside the support
trace, weight = generate(genfoo, (), choicemap(:a => 0))
@test isapprox(weight, -8*log(2))
trace, weight = generate(genfoo, (), choicemap(:b => 0))
@test isapprox(weight, -16*log(2))
trace, weight = generate(genfoo, (), choicemap(:c => 0))
@test isapprox(weight, -32*log(2))

function foo(n::Int)
    a = rand(Int8, n)
    b = rand(Int8, (n, n))
    c = rand(Int8, n, n, n)
end

genfoo = genify(foo, Int)

# Check types
trace = simulate(genfoo, (10,))
@test trace[:a] isa Array{Int8,1}
@test trace[:b] isa Array{Int8,2}
@test trace[:c] isa Array{Int8,3}

# Test array sizes
@test size(trace[:a]) == (10,)
@test size(trace[:b]) == (10, 10)
@test size(trace[:c]) == (10, 10, 10)

# Test sample range
@test all(typemin(Int8) .<= trace[:a] .<= typemax(Int8))
@test all(typemin(Int8) .<= trace[:b] .<= typemax(Int8))
@test all(typemin(Int8) .<= trace[:c] .<= typemax(Int8))

end
