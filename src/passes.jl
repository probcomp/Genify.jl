using IRTools: IR, arguments, argument!, deletearg!, recurse!, xcall

"Resolves GlobalRefs to their values."
resolve(val::Any) = val
resolve(ref::GlobalRef) = Core.eval(ref.mod, ref.name)

const genrand_ref = GlobalRef(@__MODULE__, :genrand)

"Transforms IR of a Julia method to support traced execution in Gen."
function genify_ir!(ir::IR; autoname::Bool=true)
    # Modify arguments
    deletearg!(ir, 1) # Remove argument that refers to function object
    state = argument!(ir; at=1) # Add argument that refers to GFI state
    rand_addrs = Dict() # Map from rand statements to address statements
    # Iterate over IR
    for (x, stmt) in ir
        if isexpr(stmt.expr, :call)
            # Replace calls to Base.rand with calls to genrand
            fn, args = stmt.expr.args[1], stmt.expr.args[2:end]
            is_apply = fn == GlobalRef(Core, :_apply)
            if (is_apply) fn, args = args[1], args[2:end] end
            if resolve(fn) !== Base.rand continue end
            addr = insert!(ir, x, QuoteNode(gensym()))
            rand_addrs[x] = addr # Remember IRVar for address
            if is_apply
                a = insert!(ir, x, xcall(GlobalRef(Core, :tuple), state, addr))
                ir[x] = xcall(GlobalRef(Core, :_apply), genrand_ref, a, args...)
            else
                ir[x] = xcall(genrand_ref, state, addr, args...)
            end
        end
    end
    if (autoname) ir = autoname_ir!(ir, rand_addrs) end
    return ir
end

"Automatically generate user-friendly address names."
function autoname_ir!(ir::IR, rand_addrs::Dict)
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
