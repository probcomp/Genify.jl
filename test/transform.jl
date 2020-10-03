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
ir_out = transform!(copy(ir_in); autoname=false)
@test check_ir(ir_in, ir_out)

# Test rand with varargs
function foo(dims...)
    p = rand(Beta(1, 1))
    probs = [p, 1-p]
    y = rand(Categorical(probs), dims...)
    return y
end

ir_in = IR(typeof(foo))
ir_out = transform!(copy(ir_in); autoname=false)
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
@test methods(julia_fn).ms[1].sig == Tuple{typeof(julia_fn), Any, Real, Real}
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

genfoo = genify(foo, Int; scheme=Genify.SlotNaming())
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

genfoo = genify(foo, Bool; scheme=Genify.SlotNaming())
choices, _, _ = propose(genfoo, (true,))
@test has_value(choices, :y) && !has_value(choices, :y_1)
choices, _, _ = propose(genfoo, (false,))
@test !has_value(choices, :y) && has_value(choices, :y_1)

end
