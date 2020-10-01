module Genify

export genify

using Gen, Distributions, Random
using MacroTools, IRTools

include("distributions.jl")
include("rand.jl")
include("passes.jl")

"Transforms a Julia method to a dynamic Gen function."
function genify(fn::Function, arg_types...;
                autoname::Bool=true, verbose::Bool=false, return_gf::Bool=true)
    # Get name and IR of function
    fn_name = nameof(fn)
    n_args = length(arg_types)
    # Construct and transform IR
    ir = IR(typeof(fn), arg_types...; slots=true)
    if isnothing(ir) error("No IR available for this method signature.") end
    arg_types = collect(ir.meta.method.sig.parameters)[2:end]
    if verbose println("== Original IR =="); display(ir) end
    ir = genify_ir!(ir; autoname=autoname)
    if verbose println("== Transformed IR =="); display(ir) end
    # Build new Julia function
    traced_fn = build_func(ir, gensym(fn_name), [Any; arg_types])
    if !return_gf return traced_fn end
    # Embed Julia function within dynamic generative function
    arg_defaults = fill(nothing, n_args)
    has_argument_grads = fill(false, n_args)
    accepts_output_grad = false
    gen_fn = Gen.DynamicDSLFunction{Any}(
        Dict(), Dict(), arg_types, false, arg_defaults,
        traced_fn, has_argument_grads, accepts_output_grad)
    return gen_fn
end

"Build Julia function from IR."
function build_func(ir::IR, name::Symbol, types=nothing)
    argnames = [Symbol(:arg, i) for i=1:length(arguments(ir))]
    if isnothing(types) types = [Any for i=1:length(arguments(ir))] end
    argsigs = [Expr(:(::), n, QuoteNode(t)) for (n, t) in zip(argnames, types)]
    fn = @eval @generated function $(name)($(argsigs...))
        return IRTools.Inner.build_codeinfo($ir)
    end
    return fn
end

end
