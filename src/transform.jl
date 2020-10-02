using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall, @dynamo

unwrap(val) = val
unwrap(ref::GlobalRef) = ref.name

"Transforms a Julia method to a dynamic Gen function."
function genify(fn, arg_types...; autoname::Bool=true, return_gf::Bool=true)
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
    if autoname
        traced_fn = @eval function $(gensym(fn_name))(state, $(args...))
            return named_splicer(state, $fn, $(arg_names...))
        end
    else
        traced_fn = @eval function $(gensym(fn_name))(state, $(args...))
            return splicer(state, $fn, $(arg_names...))
        end
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

@dynamo function splicer(state, fn, args...)
    ir = IR(fn, args...)
    if ir == nothing return end
    return transform!(ir)
end

@dynamo function named_splicer(state, fn, args...)
    ir = IR(fn, args...; slots=true)
    if ir == nothing return end
    return transform!(ir; autoname=true)
end

function transform!(ir::IR; autoname::Bool=false)
    # Modify arguments
    state = argument!(ir; at=1) # Add argument that refers to GFI state
    rand_addrs = Dict() # Map from rand statements to address statements
    # Iterate over IR
    for (x, stmt) in ir
        if isexpr(stmt.expr, :call)
            # Replace calls to rand with calls to genrand
            fn, args = stmt.expr.args[1], stmt.expr.args[2:end]
            is_apply = fn == GlobalRef(Core, :_apply)
            if (is_apply) fn, args = args[1], args[2:end] end
            if fn.name != :rand continue end
            addr = insert!(ir, x, QuoteNode(gensym()))
            rand_addrs[x] = addr # Remember IRVar for address
            if is_apply
                a = insert!(ir, x, xcall(GlobalRef(Core, :tuple), state, addr, fn))
                ir[x] = xcall(GlobalRef(Core, :_apply), GlobalRef(Genify, :tracer), a, args...)
            else
                ir[x] = xcall(GlobalRef(Genify, :tracer), state, addr, fn, args...)
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
