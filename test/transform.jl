using Genify: unwrap, transform!
using MacroTools: isexpr
using IRTools: IR

@testset "IR transform" begin

function check_ir(ir_in::IR, ir_out::IR)
    for (x, stmt) in ir_in
        e_in, e_out = stmt.expr, ir_out[x].expr
        if !isexpr(e_in, :call) continue end
        if unwrap(e_in.args[1]) == :rand
            if !(isexpr(e_out, :call) &&
                 e_out.args[1] == GlobalRef(Genify, :trace))
                error("$e_in is not traced")
            end
        elseif e_in.args[1] == GlobalRef(Core, :_apply)
            if !(isexpr(e_out, :call) &&
                 e_out.args[1] == GlobalRef(Core, :_apply) &&
                 e_out.args[2] == GlobalRef(Genify, :trace))
                error("$e_in is not traced")
            end
        end
    end
    return true
end

# Test straight line code
function foo(x::Int)
    p = rand(Beta(1, 1))
    probs = [p, 1-p]
    y = rand(Categorical(probs))
    return (x+y)
end

ir_in = IR(typeof(foo), Int)
ir_out = transform!(copy(ir_in))
@test check_ir(ir_in, ir_out)

# Test branching code
function foo(x::Int)
    if x > 0
        y = rand(Uniform(0, 1))
    else
        y = rand(Normal(0, 1))
    end
    return y
end

ir_in = IR(typeof(foo), Int)
ir_out = transform!(copy(ir_in))
@test check_ir(ir_in, ir_out)

# Test rand with varargs
function foo(dims...)
    p = rand(Beta(1, 1))
    probs = [p, 1-p]
    y = rand(Categorical(probs), dims...)
    return y
end

ir_in = IR(typeof(foo))
ir_out = transform!(copy(ir_in))
@test check_ir(ir_in, ir_out)

end

@testset "Generative function building" begin

function foo(alpha::Real, beta::Real)
    p = rand(Beta(alpha, beta))
    coin = rand(Bernoulli(p))
    return coin
end

genfoo = genify(foo, Int, Float64)
julia_fn = genfoo.julia_function
@test Genify.signature(julia_fn) == Tuple{Real, Real}
@test genfoo.arg_types == [Real, Real]

# Test simulate
trace = simulate(genfoo, (1, 1))
@test 0 <= trace[:p] <= 1
@test trace[:coin] in [true, false]

# Test generate
trace, weight = generate(genfoo, (1, 1), choicemap(:p => 0.25, :coin => false))
@test trace[:p] == 0.25
@test trace[:coin] == false

# Test update
trace, weight, _, discard =
    update(trace, (1, 1), (NoChange(), NoChange()), choicemap(:coin => true))
@test trace[:coin] == true
@test discard[:coin] == false
@test weight == log(0.25) - log(0.75)

end

@testset "Automatic random variable naming" begin

# Test straight line code
function foo(x::Int)
    p = rand(Beta(1, 1))
    probs = [p, 1-p]
    y = rand(Categorical(probs))
    return (x+y)
end

genfoo = genify(foo, Int; useslots=true)
choices, _, _ = propose(genfoo, (0,))
@test has_value(choices, :p) && has_value(choices, :y)

# Test branching code
function foo(x::Bool)
    if x
        y = rand(Uniform(0, 1))
    else
        y = rand(Normal(0, 1))
    end
end

genfoo = genify(foo, Bool; useslots=true)
choices, _, _ = propose(genfoo, (true,))
@test has_value(choices, :y) && !has_value(choices, :y_1)
choices, _, _ = propose(genfoo, (false,))
@test !has_value(choices, :y) && has_value(choices, :y_1)

# Test broadcast in code
function foo()
    x = [1.0, 3.0, 5.0] .+ rand(Normal(0.0, 1.0))
end
genfoo = genify(foo; useslots=true)
choices, _, _ = propose(genfoo, ())
@test has_value(choices, :x)

# Test for loop in code
function foo(N::Int)
    for i in 1 : N
        for j in 1 : N
            x = rand(Normal(0.0, 1.0))
        end
    end
end

genfoo = genify(foo, Int; useslots=true)
choices, _, _ = propose(genfoo, (5, ))

end

@testset "Nested generative functions" begin

    function model(mu::Real)
        theta = rand(LogNormal(mu, 1))
        obs = observe(theta)
    end

    function observe(theta::Real)
        m1 = rand(Normal(theta, 1))
        m2 = rand(Normal(theta, 2))
        m3 = rand(Normal(theta, 3))
        return (m1, m2, m3)
    end

    genmodel = genify(model, Real)

    # Test simulate
    trace = simulate(genmodel, (1,))
    @test trace[:theta] >= 0
    @test trace[:obs] isa Tuple{Float64, Float64, Float64}
    @test trace[:obs => :m1] isa Float64
    @test trace[:obs => :m2] isa Float64
    @test trace[:obs => :m3] isa Float64

    # Test generate
    observations = choicemap((:obs => :m1, 3), (:obs => :m2, 3), (:obs => :m3, 3))
    trace, weight = generate(genmodel, (1,), observations)
    @test trace[:obs => :m1] == 3
    @test trace[:obs => :m2] == 3
    @test trace[:obs => :m3] == 3
    @test weight ≈ sum(Gen.logpdf(normal, 3, trace[:theta], i) for i in 1:3)

    # Test regenerate
    prev_trace = trace
    trace, weight, _ = regenerate(trace, (1,), (NoChange(),), select(:theta))
    @test trace[:obs] == prev_trace[:obs]

    prev_trace = trace
    trace, weight, _ = regenerate(trace, (1,), (NoChange(),), select(:obs))
    @test trace[:theta] == prev_trace[:theta]

    # Test non-recursion
    genmodel = genify(model, Real; recurse=false)
    choices, _, _ = propose(genmodel, (1,))
    @test :obs ∉ keys(nested_view(choices))

end
