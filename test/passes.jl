using Genify: resolve, genify_ir
using MacroTools: isexpr
using IRTools: IR

@testset "IR transform" begin

function check_ir(ir_in::IR, ir_out::IR)
    for (x, stmt) in ir_in
        e_in, e_out = stmt.expr, ir_out[x].expr
        if !isexpr(e_in, :call) continue end
        if resolve(e_in.args[1]) == Base.rand
            if !(isexpr(e_out, :call) && e_out.args[1] == Genify.genrand)
                return false
            end
        elseif e_in.args[1] == GlobalRef(Core, :_apply)
            if !(isexpr(e_out, :call) &&
                 e_out.args[1] == GlobalRef(Core, :_apply) &&
                 e_out.args[2] == Genify.genrand)
                return false
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
ir_out = genify_ir(ir_in; autoname=false)
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
ir_out = genify_ir(ir_in; autoname=false)
@test check_ir(ir_in, ir_out)

# Test rand with varargs
function foo(dims...)
    p = rand(Beta(1, 1))
    probs = [p, 1-p]
    y = rand(Categorical(probs), dims...)
    return y
end

ir_in = IR(typeof(foo))
ir_out = genify_ir(ir_in; autoname=false)
@test check_ir(ir_in, ir_out)

end
