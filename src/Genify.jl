module Genify

export genify

using Gen, Distributions, Random
using MacroTools, IRTools
using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall

include("distributions.jl")

resolve(val::Any) = val
resolve(gr::GlobalRef) = Core.eval(gr.mod, gr.name)

"Rewritten call to `rand` that supports traced execution in Gen."
genrand(state, addr, rng::AbstractRNG, args...) =
    genrand(state, addr, args...)
genrand(state, addr, dist::D) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(D), params(dist), addr)
genrand(state, addr) =
    genrand(state, addr, Float64)
genrand(state, addr, T::Type{<:AbstractFloat}) =
    Gen.traceat(state, uniform_continuous, (0, 1), addr)
genrand(state, addr, T::Type{<:AbstractFloat}, dims...) =
    Gen.traceat(state, ArrayDistribution(uniform_continuous, dims), (0, 1), addr)
genrand(state, addr, T::Type{<:Integer}) =
    Gen.traceat(state, uniform_discrete, (typemin(T), typemax(T)), addr)
genrand(state, addr, T::Type{<:Integer}, dims...) =
    Gen.traceat(state, ArrayDistribution(uniform_discrete, dims), (typemin(T), typemax(T)), addr)

"Transforms a Julia method to a dynamic Gen function."
function genify(fn::Function, arg_types...;
                autoname::Bool=true, verbose::Bool=false, return_gf::Bool=true)
    # Get name and IR of function
    fn_name = nameof(fn)
    n_args = length(arg_types)
    # Construct and transform IR
    ir = IR(typeof(fn), arg_types...; slots=true)
    if isnothing(ir) error("No IR available for this method signature.") end
    if verbose println("== Original IR =="); display(ir) end
    ir = genify_ir(ir; autoname=autoname)
    if verbose println("== Transformed IR =="); display(ir) end
    # Build new Julia function
    traced_fn = build_func(ir, gensym(fn_name))
    if !return_gf return traced_fn end
    # Embed Julia function within dynamic generative function
    arg_defaults = fill(nothing, n_args)
    has_argument_grads = fill(false, n_args)
    accepts_output_grad = false
    gen_fn = Gen.DynamicDSLFunction{Any}(
        Dict(), Dict(), collect(arg_types), false, arg_defaults,
        traced_fn, has_argument_grads, accepts_output_grad)
    return gen_fn
end

"Transforms IR of a Julia method to support traced execution in Gen."
function genify_ir(ir::IR; autoname::Bool=true)
    # Modify arguments
    deletearg!(ir, 1) # Remove argument that refers to function object
    state = argument!(ir; at=1) # Add argument that refers to GFI state
    randvars = Set{IRTools.Variable}() # Track which IR variables are random
    # Iterate over IR
    for (x, stmt) in ir
        if isexpr(stmt.expr, :call)
            # Replace calls to Base.rand with calls to genrand
            fn, args = stmt.expr.args[1], stmt.expr.args[2:end]
            if resolve(fn) !== Base.rand continue end
            push!(randvars, x) # Remember if this variable is a call to rand
            addr = QuoteNode(gensym())
            ir[x] = xcall(genrand, state, addr, args...)
        elseif isexpr(stmt.expr, :(=)) && autoname
            # Attempt to automatically generate address names from slot names
            slot, v = stmt.expr.args
            if !isa(slot, IRTools.Slot) || !(v in randvars) continue end
            slot_num = parse(Int, string(slot.id)[2:end])
            addr = ir.meta.code.slotnames[slot_num] # Look up name in CodeInfo
            ir[v].expr.args[3] = QuoteNode(addr) # Replace gensym-ed address
        end
    end
    return ir
end

"Build Julia function from IR."
function build_func(ir::IR, name::Symbol)
    argnames = [Symbol(:arg, i) for i=1:length(arguments(ir))]
    fn = @eval @generated function $(name)($(argnames...))
        return IRTools.Inner.build_codeinfo($ir)
    end
    return fn
end

end
