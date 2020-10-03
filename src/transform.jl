using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall, @dynamo

abstract type NamingScheme end
struct NoNaming <: NamingScheme end
struct SlotNaming <: NamingScheme end

const gen_fn_cache = Dict{Tuple,Gen.DynamicDSLFunction}()
const randprims = Set([:rand, :randn, :randexp, :randperm, :shuffle])
const untraced = setdiff(Set([names(Base); names(Core)]), randprims)

unwrap(val) = val
unwrap(ref::GlobalRef) = ref.name

"Transforms a Julia method to a dynamic Gen function."
function genify(fn, arg_types...;
                scheme::NamingScheme=SlotNaming(), return_gf::Bool=true)
    # Get name and IR of function
    fn_name = nameof(fn)
    n_args = length(arg_types)
    # Get type information
    meta = IRTools.meta(Tuple{typeof(fn), arg_types...})
    if isnothing(meta) error("No IR available for this method signature.") end
    arg_types = collect(meta.method.sig.parameters)[2:end]
    # Build new Julia function
    arg_names = [Symbol(:arg, i) for i=1:length(arg_types)]
    args = [:($(n)::$(QuoteNode(t))) for (n, t) in zip(arg_names, arg_types)]
    traced_fn = @eval function $(gensym(fn_name))(state, $(args...))
        return splice($scheme, state, $fn, $(arg_names...))
    end
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

"Memoized call to genify."
function genified(fn, arg_types...; scheme::NamingScheme=SlotNaming())
    gen_fn = genify(fn, arg_types...; scheme=scheme)
    fn_type = typeof(fn)
    gen_fn_cache[(fn_type, arg_types...)] = gen_fn
    args = [:(::$(QuoteNode(Type{T}))) for T in arg_types]
    @eval function genified(fn::$fn_type, $(args...))
        return gen_fn_cache[($fn_type, $(arg_types...))]
    end
    return gen_fn
end

"""
Trace an arbitrary method without nesting under an address namespace. This is
done by transforming the IR to wrap each sub-call with a call to `trace`(@ref),
generating the corresponding address names, and adding an extra argument
for the GFI state.
"""
function splice end

@dynamo function splice(scheme::T, state, fn, args...) where {T<:NamingScheme}
    ir = IR(fn, args...; slots=(scheme==SlotNaming))
    if ir == nothing return end
    return transform!(ir; scheme_arg=true, autoname=(scheme==SlotNaming))
end

"""
Trace random primitives or arbitrary methods. Random primitives are traced
by constructing the appropriate `Gen.Distribution`. Arbitrary methods are
traced by converting them to `Gen.DynamicDSLFunction`s via a memoized
call to [`genify`](@ref).
"""
function trace end

@generated function trace(state, addr::Address, fn, args...)
    meta = IRTools.meta(Tuple{fn, args...})
    if isnothing(meta) return :(fn(args...)) end
    arg_types = collect(meta.method.sig.parameters)[2:end]
    gen_fn = :($(GlobalRef(Genify, :genified))(fn, $(arg_types...)))
    return :($(GlobalRef(Gen, :traceat))(state, $gen_fn, args, addr))
end

# Make sure Distribution constructors aren't traced
trace(state, addr::Address, D::Type{<:Distributions.Distribution}, args...) =
    D(args...)

"Transform the IR by wrapping subcalls in `trace`(@ref)."
function transform!(ir::IR; scheme_arg::Bool=false, autoname::Bool=false)
    # Modify arguments
    state = argument!(ir; at=1) # Add argument that refers to GFI state
    if (scheme_arg) argument!(ir; at=1) end # Add argument for naming scheme
    rand_addrs = Dict() # Map from rand statements to address statements
    # Iterate over IR
    for (x, stmt) in ir
        if isexpr(stmt.expr, :call)
            # Replace calls to rand with calls to genrand
            fn, args = stmt.expr.args[1], stmt.expr.args[2:end]
            is_apply = fn == GlobalRef(Core, :_apply)
            if (is_apply) fn, args = args[1], args[2:end] end
            # TODO: More careful filtering
            if fn.mod in [Core, Base] && !(fn.name in randprims) continue end
            if fn.name in untraced continue end
            addr = insert!(ir, x, QuoteNode(gensym(fn.name)))
            rand_addrs[x] = addr # Remember IRVar for address
            if is_apply
                a = insert!(ir, x, xcall(GlobalRef(Core, :tuple), state, addr, fn))
                ir[x] = xcall(GlobalRef(Core, :_apply), GlobalRef(Genify, :trace), a, args...)
            else
                ir[x] = xcall(GlobalRef(Genify, :trace), state, addr, fn, args...)
            end
        end
    end
    if (autoname) ir = autoname!(ir, rand_addrs) end
    return ir
end

"Automatically generate user-friendly address names."
function autoname!(ir::IR, rand_addrs::Dict)
    slotnames = ir.meta.code.slotnames
    slotcount = Dict{Int,Int}()
    for (x, stmt) in ir
        if isexpr(stmt.expr, :(=))
            # Attempt to automatically generate address names from slot names
            slot, v = stmt.expr.args
            if !isa(slot, IRTools.Slot) || !(v in keys(rand_addrs)) continue end
            slot_id = parse(Int, string(slot.id)[2:end])
            addr = slotnames[slot_id] # Look up name in CodeInfo
            if get!(slotcount, slot_id, 0) > 0 # Ensure uniqueness
                addr =Symbol(addr, :_, slotcount[slot_id])
            end
            slotcount[slot_id] += 1
            ir[rand_addrs[v]] = QuoteNode(addr) # Replace gensym-ed address
        end
    end
    return ir
end
