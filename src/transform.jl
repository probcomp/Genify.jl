using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall, @dynamo

const gen_fn_cache = Dict{Tuple,Gen.DynamicDSLFunction}()
const randprims = Set([:rand, :randn, :randexp, :randperm, :shuffle])
const untraced = setdiff(Set([names(Base); names(Core)]), randprims)

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

"Returns the argument signature of a traced function."
signature(::TracedFunction{S}) where {S} = S

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
    meta = IRTools.meta(Tuple{typeof(fn), arg_types...})
    if isnothing(meta) error("No IR available for this method signature.") end
    arg_types = collect(meta.method.sig.parameters)[2:end]
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
    gen_fn_cache[(op_type, fn_type, arg_types...)] = gen_fn
    args = [:(::$(QuoteNode(Type{T}))) for T in arg_types]
    @eval function genified(options::$op_type, fn::$fn_type, $(args...))
        return gen_fn_cache[($op_type, $fn_type, $(arg_types...))]
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
    trace(state, addr, fn, args...)
    trace(options, state, addr, fn, args...)

Trace random primitives or arbitrary methods. Random primitives are traced
by constructing the appropriate `Gen.Distribution`. Arbitrary methods are
traced by converting them to `Gen.DynamicDSLFunction`s via a memoized
call to [`genify`](@ref).
"""
function trace end

@generated function trace(options::Options, state, addr::Address, fn, args...)
    meta = IRTools.meta(Tuple{fn, args...})
    if isnothing(meta) return :(fn(args...)) end
    arg_types = collect(meta.method.sig.parameters)[2:end]
    gen_fn = :($(GlobalRef(Genify, :genified))(options, fn, $(arg_types...)))
    return :($(GlobalRef(Gen, :traceat))(state, $gen_fn, args, addr))
end

# Ensure type constructors aren't traced (assumes constructors are not random)
trace(options::Options, state, addr::Address, T::Type, args...) = T(args...)

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
        fn, args = stmt.expr.args[1], stmt.expr.args[2:end]
        is_apply = fn == GlobalRef(Core, :_apply)
        if (is_apply) fn, args = args[1], args[2:end] end
        # Filter out calls in Core, Base, and non-primitives if not recursing
        if (!recurse && !(unwrap(fn) in randprims) ||
            fn isa GlobalRef && fn.mod in (Core, Base) ||
            unwrap(fn) in untraced)
            continue
        end
        # Generate address name from function name
        addr = unwrap(fn) isa Symbol ? gensym(unwrap(fn)) : gensym()
        addr = insert!(ir, x, QuoteNode(addr))
        rand_addrs[x] = addr # Remember IRVar for address
        # Wrap call within `trace`
        if is_apply
            trace_apply!(ir, x, QuoteNode(options), state, addr, fn, args)
        else
            trace_call!(ir, x, QuoteNode(options), state, addr, fn, args)
        end
    end
    if (useslots) ir = autoname!(ir, rand_addrs) end
    return ir
end

function trace_apply!(ir, var, options, state, addr, fn, args)
    preargs = unwrap(fn) in randprims ?
        xcall(GlobalRef(Core, :tuple), state, addr, fn) :
        xcall(GlobalRef(Core, :tuple), options, state, addr, fn)
    preargs = insert!(ir, var, preargs)
    ir[var] = xcall(GlobalRef(Core, :_apply), GlobalRef(Genify, :trace),
                    preargs, args...)
end

function trace_call!(ir, var, options, state, addr, fn, args)
    ir[var] = unwrap(fn) in randprims ?
        xcall(GlobalRef(Genify, :trace), state, addr, fn, args...) :
        xcall(GlobalRef(Genify, :trace), options, state, addr, fn, args...)
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
