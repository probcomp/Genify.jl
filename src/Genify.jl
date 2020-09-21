module Genify

export genify

using Gen, Distributions, Random
using MacroTools, IRTools
using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall

include("distributions.jl")

resolve(val::Any) = val
resolve(gr::GlobalRef) = Core.eval(gr.mod, gr.name)

"Rewritten call to `rand` that supports traced execution in Gen."
# Strip away RNGs supplied to rand
genrand(state, addr, rng::AbstractRNG, args...) =
    genrand(state, addr, args...)

# rand for numeric types
genrand(state, addr) =
    genrand(state, addr, Float64)
genrand(state, addr, T::Type{<:Real}) =
    Gen.traceat(state, TypedScalarDistribution(T), (), addr)
genrand(state, addr, T::Type{<:Real}, dims::Tuple) =
    Gen.traceat(state, TypedArrayDistribution(T, dims), (), addr)
genrand(state, addr, T::Type{<:Real}, dims...) =
    Gen.traceat(state, TypedArrayDistribution(T, dims...), (), addr)

# rand for indexable collections
const Indexable = Union{AbstractArray, AbstractRange, Tuple, AbstractString}
flatten(c::Indexable) = c
flatten(c::AbstractArray) = vec(c)
genrand(state, addr, c::Indexable) =
    Gen.traceat(state, labeled_uniform, (flatten(c),), addr)
genrand(state, addr, c::Indexable, dims::Tuple) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, dims), (flatten(c),), addr)
genrand(state, addr, c::Indexable, dims...) =
    Gen.traceat(state, ArrayedDistribution(labeled_uniform, dims), (flatten(c),), addr)

# rand for set-like collections
genrand(state, addr, c::AbstractSet{T}) where {T} =
    Gen.traceat(state, SetUniformDistribution{T}(), (c,), addr)
genrand(state, addr, c::AbstractSet{T}, dims::Tuple) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), dims), (c,), addr)
genrand(state, addr, c::AbstractSet{T}, dims...) where {T} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{T}(), dims), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}) where {T,U} =
    Gen.traceat(state, SetUniformDistribution{Pair{T,U}}(), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}, dims::Tuple) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), dims), (c,), addr)
genrand(state, addr, c::AbstractDict{T,U}, dims...) where {T,U} =
    Gen.traceat(state, ArrayedDistribution(SetUniformDistribution{Pair{T,U}}(), dims), (c,), addr)

# rand for Distributions.jl distributions
genrand(state, addr, dist::D) where {D <: Distributions.Distribution} =
    Gen.traceat(state, WrappedDistribution(D), params(dist), addr)

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
    ir = genify_ir(ir; autoname=autoname)
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
