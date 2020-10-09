using MacroTools: isexpr

"""
    Options{recurse::Bool, useslots::Bool, scheme::Symbol}

Option type that specifies how Julia methods should be transformed into
generative functions. These options are passed as type parameters so
that they are accessible within `@generated` functions.
"""
struct Options{R, U, S} end
const MinimalOptions = Options{false, false, :static}
const DefaultOptions = Options{true, true, :static}

Options(recurse::Bool, useslots::Bool, scheme::Symbol) =
    Options{recurse, useslots, scheme}()

"Unpack option type parameters as a tuple."
unpack(::Options{R,U,S}) where {R, U, S} = (R, U, S)

"""
    TracedFunction{S <: Tuple}

Wraps a Julia function or callable object with method signature `S`.
When called, performs traced execution of the underlying function `fn` by
passing `fn` to the [`splice`](@ref) dynamo.
"""
struct TracedFunction{S <: Tuple} <: Function
    fn # Original function
    options::Options # Genification options
end

TracedFunction(fn, sig::Type{<:Tuple}) =
    TracedFunction{sig}(fn, DefaultOptions())
TracedFunction(fn, sig::Type{<:Tuple}, options::Options) =
    TracedFunction{sig}(fn, options)

(tf::TracedFunction{S})(state, args...) where {S} =
    splice(tf.options, state, tf.fn, args::S...)

"Returns the argument signature of the matching method."
signature(::TracedFunction{S}) where {S} = S
function signature(fn_type::Type, arg_types::Type...)
    meta = IRTools.meta(Tuple{fn_type, arg_types...})
    if isnothing(meta) return nothing end
    sig_types = collect(Base.unwrap_unionall(meta.method.sig).parameters[2:end])
    for (i, a) in enumerate(arg_types)
        if !(sig_types[i] isa TypeVar) continue end
        sig_types[i] = a
    end
    return sig_types
end

"Unwrap `GlobalRef`s, returning the unqualified name."
unwrap(ref::GlobalRef) = ref.name
unwrap(val) = val

"""
    genify(fn, arg_types...; kwargs...)

Transforms a Julia method into a dynamic Gen function. `fn` can be a `Function`
or any other callable object, and `arg_types` are the types of the corresponding
arguments.

# Arguments:
- `recurse::Bool=true`: recursively `genify` methods called by `fn` if true.
- `useslots::Bool=true`: if true, use slot (i.e. variable) names as trace
        addresses where possible.
- `scheme::Symbol=:static`: scheme for generating address names, defaults to
        `static` generation at compile time.
- `options=nothing`: the above options can also be provided as parameters in an
        [`Options`](@ref) struct, overriding any other values specified.
"""
function genify(fn, arg_types::Type...; options=nothing, recurse::Bool=true,
                useslots::Bool=true, scheme::Symbol=:static)
    # Get name and IR of function
    fn_name = nameof(fn)
    n_args = length(arg_types)
    # Get type information
    fn_type = fn isa Type ? Type{fn} : typeof(fn)
    arg_types = signature(fn_type, arg_types...)
    if isnothing(arg_types) error("No method definition available for $fn.") end
    # Construct traced function
    options = isnothing(options) ? Options(recurse, useslots, scheme) : options
    traced_fn = TracedFunction(fn, Tuple{arg_types...}, options)
    # Embed traced function within dynamic generative function
    arg_defaults = fill(nothing, n_args)
    has_argument_grads = fill(false, n_args)
    accepts_output_grad = false
    gen_fn = Gen.DynamicDSLFunction{Any}(
        Dict(), Dict(), arg_types, false, arg_defaults,
        traced_fn, has_argument_grads, accepts_output_grad)
    return gen_fn
end

"Memoized [`genify`](@ref) that compiles specialized versions of itself."
function genified(options::Options, fn, arg_types::Type...)
    gen_fn = genify(fn, arg_types...; options=options)
    op_type, fn_type = typeof(options), typeof(fn)
    args = [:(::$(QuoteNode(Type{T}))) for T in arg_types]
    @eval function genified(options::$op_type, fn::$fn_type, $(args...))
        return $gen_fn
    end
    return gen_fn
end

"""
    splice(state, fn, args...)
    splice(options, state, fn, args...)

Trace an arbitrary method without nesting under an address namespace. This is
done by transforming the method's IR to wrap each sub-call with a call
to `trace`(@ref), generating the corresponding address names, and adding an
extra argument for the GFI `state`.
"""
function splice end

@dynamo function splice(options::T, state, fn, args...) where {T<:Options}
    _, useslots, _ = unpack(options())
    ir = IR(fn, args...; slots=useslots)
    if ir == nothing return end
    return transform!(ir, options())
end

"""
    trace(options, state, addr, fn, args...)

Trace random primitives or arbitrary methods. Random primitives are traced
by constructing the appropriate `Gen.Distribution`. Arbitrary methods are
traced by converting them to `Gen.DynamicDSLFunction`s via a memoized
call to [`genify`](@ref).
"""
function trace end

@generated function trace(options::Options, state, addr::Address, fn, args...)
    recurse, _, _ = unpack(options())
    if !recurse return :(fn(args...)) end
    arg_types = signature(fn, args...)
    if isnothing(arg_types) return :(fn(args...)) end
    gen_fn = :($(GlobalRef(Genify, :genified))(options, fn, $(arg_types...)))
    return :($(GlobalRef(Gen, :traceat))(state, $gen_fn, args, addr))
end

"Transform the IR by wrapping sub-calls in `trace`(@ref)."
function transform!(ir::IR, options::Options=MinimalOptions())
    recurse, useslots, scheme = unpack(options)
    # Modify arguments
    state = argument!(ir; at=1) # Add argument that refers to GFI state
    optarg = argument!(ir; at=1) # Add argument for options
    rand_addrs = Dict() # Map from rand statements to address statements
    # Iterate over IR
    for (x, stmt) in ir
        if !isexpr(stmt.expr, :call) continue end
        # Unpack call
        fn, args, calltype = unpack_call(stmt.expr)
        # Filter out untraced calls
        if !is_traced(ir, fn, recurse) continue end
        # Generate address name from function name
        addr = unwrap(fn) isa Symbol ? gensym(unwrap(fn)) : gensym()
        addr = insert!(ir, x, QuoteNode(addr))
        rand_addrs[x] = addr # Remember IRVar for address
        # Rewrite statement by wrapping call within `trace`
        rewrite!(ir, x, calltype, QuoteNode(options), state, addr, fn, args)
    end
    if (useslots) ir = autoname!(ir, rand_addrs) end
    return ir
end

"Unpack calls, special casing `Core._apply` and `Core._apply_iterate`."
function unpack_call(expr::Expr)
    fn, args, calltype = expr.args[1], expr.args[2:end], :call
    if fn == GlobalRef(Core, :_apply)
        fn, args, calltype = args[1], args[2:end], :apply
    elseif fn == GlobalRef(Core, :_apply_iterate)
        fn, args, calltype = args[2], args[3:end], :apply_iterate
    end
    return fn, args, calltype
end

"Determine whether a called function should be traced."
function is_traced(ir, fn::GlobalRef, recurse::Bool)
    if !isdefined(fn.mod, fn.name) return error("$fn not defined.") end
    val = getfield(fn.mod, fn.name)
    if val in randprims return true end # Primitives are always traced
    if !recurse return false end # Only trace primitives if not recursing
    for m in (Base, Core, Core.Intrinsics) # Filter out Base, Core, etc.
        if isdefined(m, fn.name) && getfield(m, fn.name) == val return false end
    end
    if val isa Type && val <: Sampleable return false end # Filter distributions
    return true
end
is_traced(ir, fn::Function, recurse::Bool) = # Handle injected functions
    fn in randprims || recurse
is_traced(ir, fn::IRTools.Variable, recurse::Bool) = # Handle IR variables
    !haskey(ir, fn) || is_traced(ir, ir[fn], recurse)
is_traced(ir, fn::Expr, recurse::Bool) = # Handle keyword functions
    isexpr(fn, :call) && fn.args[1] == GlobalRef(Core, :kwfunc) ?
        is_traced(ir, fn.args[2], recurse) : true
is_traced(ir, fn, recurse::Bool) = # Return true by default, to be safe
    true

"Rewrites the statement by wrapping call within `trace`."
function rewrite!(ir, var, calltype, options, state, addr, fn, args)
    if calltype == :call # Handle basic calls
        ir[var] = xcall(GlobalRef(Genify, :trace),
                        options, state, addr, fn, args...)
    elseif calltype == :apply # Handle `Core._apply`
        preargs = xcall(GlobalRef(Core, :tuple), options, state, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(GlobalRef(Core, :_apply),
                        GlobalRef(Genify, :trace), preargs, args...)
    elseif calltype == :apply_iterate # Handle `Core._apply_iterate`
        preargs = xcall(GlobalRef(Core, :tuple), options, state, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(GlobalRef(Core, :_apply_iterate),
                        GlobalRef(Base, :iterate),
                        GlobalRef(Genify, :trace), preargs, args...)
    end
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
