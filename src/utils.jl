# Loop detection.
using IRTools:
    dominators, block ,blocks, successors, predecessors, deleteblock!

struct NaturalLoop
    header::Int
    backedge::Int
    body::Set{Int}
end

@inline header(nl::NaturalLoop) = getfield(nl, :header)
@inline backedge(nl::NaturalLoop) = getfield(nl, :backedge)
@inline body(nl::NaturalLoop) = getfield(nl, :body)

function Base.display(nl::NaturalLoop)
    println("--- Natural loop ---")
    println(" Header : $(nl.header)")
    println(" Blocks : $(nl.body)")
    println(" Backedge : $(nl.backedge)")
    println()
end

Base.display(ls::Vector{NaturalLoop}) = display.(ls)

IRTools.dominators(ir::IR) = dominators(IRTools.CFG(ir))

"Returns all edges in the control flow graph."
function edges(ir::IR; reverse=false)
    neighbors = reverse ? predecessors : successors
    return Dict(s.id => Set(t.id for t in neighbors(s)) for s in blocks(ir))
end

"Prunes unreachable blocks."
function pruneblocks!(ir::IR)
    unreachable = [b.id for b in blocks(ir) if isempty(predecessors(b))][2:end]
    map(i -> deleteblock!(ir, i), reverse(unreachable))
    return ir
end

"Builds a single loop from the IR, header block, and final block."
function buildloop(ir::IR, header::Int, backedge::Int)
    body = Set{Int}([header])
    stack = [backedge]
    while !isempty(stack)
        i = pop!(stack)
        if i in body continue end
        push!(body, i)
        append!(stack, [b.id for b in predecessors(block(ir, i))])
    end
    return NaturalLoop(header, backedge, body)
end

"Detect the set of natural loops for a piece of IR."
function detectloops(ir::IR)
    loops = NaturalLoop[]
    domdict = dominators(ir)
    for (id, doms) in domdict # Loop over dominators of each block
        for header in successors(block(ir, id)) # Check for backedges to headers
            if !(header.id in doms) continue end
            push!(loops, buildloop(ir, header.id, id))
        end
    end
    # Sort outermost to innermost, then by header index
    sort!(loops, lt=(l1, l2) -> l2.body in l1.body || l1.header < l2.header)
    return loops
end

"Detect the set of natural loops for a piece of IR, pruning unreachable blocks."
detectloops!(ir) = detectloops(pruneblocks!(ir))
