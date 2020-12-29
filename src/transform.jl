using MacroTools: isexpr, iscall

"""
    Options{recurse::Bool, useslots::Bool, naming::Symbol}

Option type that specifies how Julia methods should be transformed into
generative functions. These options are passed as type parameters so
that they are accessible within `@generated` functions.
"""
struct Options{R, U, S} end
const MinimalOptions = Options{false, false, :static}
const DefaultOptions = Options{true, true, :static}
const ManualOptions = Options{true, false, :manual}
const named_options = Dict{Symbol,Options}(
    :minimal => MinimalOptions(),
    :default => DefaultOptions(),
    :manual => ManualOptions()
)

Options(recurse::Bool, useslots::Bool, naming::Symbol) =
    Options{recurse, useslots, naming}()

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

signature(::TracedFunction{S}) where {S} = S

"""
    genify(fn, arg_types...; kwargs...)
    genify(options, fn, arg_types...)

Transforms a Julia method into a dynamic Gen function.

# Arguments:
- `fn`: a `Function`, `Type` constructor, or (if the second form is used)
        any other callable object.
- `arg_types`: The types of the arguments for the method to be transformed.

# Keyword Arguments:
- `recurse::Bool=true`: recursively `genify` methods called by `fn` if true.
- `useslots::Bool=true`: if true, use slot (i.e. variable) names as trace
        addresses where possible.
- `naming::Symbol=:static`: scheme for generating address names, defaults to
        static generation at compile time. Use `:manual` for user-specified
        addresses (e.g., `rand(:z, Normal(0, 1))`)
- `options`: the above options can also be provided as parameters in an
        [`Options`](@ref) struct, or as a `Symbol` from the list of named
        option sets overriding any other values specified:
  - `:minimal` corresponds to `recurse=false, useslots=false, naming=:static`
  - `:default` corresponds to `recurse=true, useslots=true, naming=:static`
  - `:manual` corresponds to `recurse=true, useslots=false, naming=:manual`
"""
function genify(options::Options, fn, arg_types::Type...)
    # Get name and IR of function
    fn_name = nameof(fn)
    n_args = length(arg_types)
    # Get type information
    fn_type = fn isa Type ? Type{fn} : typeof(fn)
    arg_types = signature(fn_type, arg_types...)
    if isnothing(arg_types) error("No method definition available for $fn.") end
    # Construct traced function
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

function genify(options::Symbol, fn, arg_types::Type...)
    return genify(named_options[options], fn, arg_types...)
end

function genify(fn::Union{Function,Type}, arg_types::Type...;
                options=nothing, recurse::Bool=true,
                useslots::Bool=true, naming::Symbol=:static)
    options = isnothing(options) ? Options(recurse, useslots, naming) : options
    return genify(options, fn, arg_types...)
end

"Memoized [`genify`](@ref) that compiles specialized versions of itself."
function genified(options::Options, fn, arg_types::Type...)
    gen_fn = genify(options, fn, arg_types...)
    op_type, fn_type = typeof(options), typeof(fn)
    args = [:(::$(QuoteNode(Type{T}))) for T in arg_types]
    @eval function genified(options::$op_type, fn::$fn_type, $(args...))
        return $gen_fn
    end
    return gen_fn
end

function genified(options::Symbol, fn, arg_types::Type...)
    return genified(named_options[options], fn, arg_types...)
end

function genified(fn::Union{Function,Type}, arg_types::Type...;
                  options=nothing, recurse::Bool=true,
                  useslots::Bool=true, naming::Symbol=:static)
    options = isnothing(options) ? Options(recurse, useslots, naming) : options
    return genified(options, fn, arg_types...)
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
call to [`genify`](@ref). Constructor methods are not traced.
"""
function trace end

# Trace arbitrary methods
@generated function trace(options::Options, state, addr::Address, fn, args...)
    recurse, _, naming = unpack(options())
    if !recurse return :(fn(args...)) end
    arg_types = signature(fn, args...)
    if isnothing(arg_types) return :(fn(args...)) end
    gen_fn = :($(GlobalRef(Genify, :genified))(options, fn, $(arg_types...)))
    return :($(GlobalRef(Gen, :traceat))(state, $gen_fn, args, addr))
end

# Avoid tracing constructor methods
@inline trace(::Options, state, ::Address, fn::Type, args...) = fn(args...)

"Transform the IR by wrapping sub-calls in `trace`(@ref)."
function transform!(ir::IR, options::Options=MinimalOptions())
    recurse, useslots, naming = unpack(options)
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
        if !istraced(ir, fn, recurse && naming != :manual) continue end
        if naming == :static
            # Statically generate address name from function name and arguments
            addr = genaddr(ir, fn, unpack_args(ir, args, calltype))
            addr = insert!(ir, x, QuoteNode(addr))
            rand_addrs[x] = addr # Remember IRVar for address
        elseif naming == :manual
            addr = ManualAddress() # Handle manual addressing via dispatch
        else
            error("Unrecognized naming scheme :$naming")
        end
        # Rewrite statement by wrapping call within `trace`
        rewrite!(ir, x, calltype, QuoteNode(options), state, addr, fn, args)
    end
    if naming != :manual
        if (useslots) slotaddrs!(ir, rand_addrs) end # Name addresses with slots
        uniqueaddrs!(ir, rand_addrs) # Ensure uniqueness of random addresses
        loopaddrs!(ir, rand_addrs) # Add loop indices to addresses
    end
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

"Unpack tuples in IR."
unpack_tuple(ir, v::Variable) = haskey(ir, v) ?
    unpack_tuple(ir, ir[v].expr) : nothing
unpack_tuple(ir, e) = iscall(e, GlobalRef(Core, :tuple)) ?
    e.args[2:end] : nothing

"Unpack arguments, special casing `Core._apply` and `Core._apply_iterate`."
function unpack_args(ir, args, calltype)
    if calltype == :call return args end
    unpacked = [unpack_tuple(ir, a) for a in args]
    filter!(a -> !isnothing(a), unpacked)
    return reduce(vcat, unpacked; init=[])
end

"Determine whether a called function should be traced."
function istraced(ir, fn::GlobalRef, recurse::Bool)
    if !isdefined(fn.mod, fn.name) return error("$fn not defined.") end
    val = getfield(fn.mod, fn.name)
    if val in randprims return true end # Primitives are always traced
    if !recurse return false end # Only trace primitives if not recursing
    for m in (Base, Core, Core.Intrinsics) # Filter out Base, Core, etc.
        if isdefined(m, fn.name) && getfield(m, fn.name) == val return false end
    end
    if val isa Type && val <: Sampleable return false end # Filter distributions
    if val in (truncated, product_distribution) return false end
    return true
end
istraced(ir, fn::Function, recurse::Bool) = # Handle injected functions
    fn in randprims || recurse
istraced(ir, fn::Variable, recurse::Bool) = # Handle IR variables
    !haskey(ir, fn) || istraced(ir, ir[fn].expr, recurse)
istraced(ir, fn::Expr, recurse::Bool) = # Handle keyword functions
    iscall(fn, GlobalRef(Core, :kwfunc)) ?
        istraced(ir, fn.args[2], recurse) : true
istraced(ir, fn, recurse::Bool) = # Return true by default, to be safe
    true

"Static generation of address names."
function genaddr(ir, fn::Symbol, args)
    if length(args) == 0 || !(fn in primnames) return fn end
    argsym = argaddr(ir, args[1])
    return isnothing(argsym) ? fn : Symbol(fn, :_, argsym)
end
function genaddr(ir, fn::Expr, args)
    if iscall(fn, GlobalRef(Core, :kwfunc))
        return genaddr(ir, fn.args[2], args[3:end])
    elseif isexpr(fn, :call) && fn.args[1] isa GlobalRef
        return genaddr(ir, Symbol(Expr(:call, fn.args[1].name)), args)
    else
        return :unknown
    end
end
genaddr(ir, fn::GlobalRef, args) =
    genaddr(ir, fn.name, args)
genaddr(ir, fn::Function, args) =
    genaddr(ir, nameof(fn), args)
genaddr(ir, fn::Variable, args) =
    genaddr(ir, haskey(ir, fn) ? ir[fn].expr : argname(ir, fn), args)
genaddr(ir, fn, args) =
    :unknown

"Generate partial address from argument."
argaddr(ir, arg::Variable) =
    haskey(ir, arg) ? argaddr(ir, ir[arg].expr) : argname(ir, arg)
argaddr(ir, arg::Expr) =
    isexpr(arg, :call) ? argaddr(ir, arg.args[1]) : nothing
argaddr(ir, arg::GlobalRef) =
    arg.name
argaddr(ir, arg) =
    Symbol(arg)

"Get argument slot name from IR."
argname(ir::IR, v::Variable) =
    argname(ir.meta, v)
argname(meta::IRTools.Meta, v::Variable) =
    1 <= v.id <= meta.nargs ? meta.code.slotnames[v.id] : nothing
argname(meta::Any, v::Variable) =
    nothing

"Rewrites the statement by wrapping call within `trace`."
function rewrite!(ir, var, calltype, options, state, addr, fn, args)
    if calltype == :call # Handle basic calls
        ir[var] = xcall(Genify, :trace, options, state, addr, fn, args...)
    elseif calltype == :apply # Handle `Core._apply`
        preargs = xcall(Core, :tuple, options, state, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(Core, :_apply,
                        GlobalRef(Genify, :trace), preargs, args...)
    elseif calltype == :apply_iterate # Handle `Core._apply_iterate`
        preargs = xcall(Core, :tuple, options, state, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(Core, :_apply_iterate, GlobalRef(Base, :iterate),
                        GlobalRef(Genify, :trace), preargs, args...)
    end
end

"Generate trace addresses from slotnames where possible."
function slotaddrs!(ir::IR, rand_addrs::Dict)
    slotnames = ir.meta.code.slotnames
    for (x, stmt) in ir
        if isexpr(stmt.expr, :(=))
            # Attempt to automatically generate address names from slot names
            slot, v = stmt.expr.args
            if !isa(slot, IRTools.Slot) || !(v in keys(rand_addrs)) continue end
            slot_id = parse(Int, string(slot.id)[2:end])
            addr = slotnames[slot_id] # Look up name in CodeInfo
            ir[rand_addrs[v]] = QuoteNode(addr) # Replace previous address
        end
    end
    return ir
end

"Ensure that all addresses have unique names."
function uniqueaddrs!(ir::IR, rand_addrs::Dict)
    counts = Dict{Symbol,Int}()
    firstuses = Dict{Symbol,Variable}()
    addrvars = sort(collect(values(rand_addrs)), by = v -> ir.defs[v.id])
    for v in addrvars # Number all uses after first occurrence
        addr = ir[v].expr.value
        counts[addr] = get(counts, addr, 0) + 1
        if (counts[addr] == 1) firstuses[addr] = v; continue end
        ir[v] = QuoteNode(Symbol(addr, :_, counts[addr]))
    end
    for (addr, c) in counts # Go back and number first use of name
        if c == 1 continue end
        ir[firstuses[addr]] = QuoteNode(Symbol(addr, :_, 1))
    end
    return ir
end

"Add loop indices to addresses."
function loopaddrs!(ir::IR, rand_addrs::Dict)
    # Add count variables for each loop in IR
    loops, countvars = loopcounts!(ir)
    # Append loop count to addresses in each loop body
    for (loop, count) in zip(loops, countvars)
        for addrvar in values(rand_addrs)
            if !(block(ir, addrvar).id in loop.body) continue end
            if iscall(ir[addrvar].expr, GlobalRef(Base, :Pair))
                head, tail = ir[addrvar].expr.args[2:3]
                tail = insert!(ir, addrvar, xcall(Base, :Pair, tail, count))
                ir[addrvar] = xcall(Base, :Pair, head, tail)
            else
                head = insert!(ir, addrvar, ir[addrvar])
                ir[addrvar] = xcall(Base, :Pair, head, count)
            end
        end
    end
    return ir
end
