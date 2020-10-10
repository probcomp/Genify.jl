# Loop detection.

struct NaturalLoop
    header::Int # Unique entry point to loop
    latch::Int # In-loop block that has backedge to header
    body::Set{Int} # The set of all blocks in the loop
    backedges::Set{Int} # All blocks in CFG with backedges to the header
end

function Base.display(nl::NaturalLoop)
    println("--- Natural loop ---")
    println(" Header : $(nl.header)")
    println(" Blocks : $(nl.body)")
    println(" Latch : $(nl.latch)")
    println(" Backedges : $(nl.backedges)")
    println()
end

Base.display(ls::AbstractVector{NaturalLoop}) = display.(ls)

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

"Builds a single loop from the IR, header block, and final block (latch)."
function buildloop(ir::IR, header::Int, latch::Int, backedges::Set{Int})
    body = Set{Int}([header])
    stack = [latch]
    while !isempty(stack)
        i = pop!(stack)
        if i in body continue end
        push!(body, i)
        append!(stack, [b.id for b in predecessors(block(ir, i))])
    end
    return NaturalLoop(header, latch, body, backedges)
end

"Detect the set of natural loops for a piece of IR."
function detectloops(ir::IR)
    loops = NaturalLoop[]
    domdict = dominators(ir)
    backedges = Dict{Int,Set{Int}}()
    # Find all backedges
    for (id, doms) in domdict, header in successors(block(ir, id))
        if !(header.id in doms) continue end
        push!(get!(backedges, header.id, Set{Int}()), id)
    end
    # Create loop for each backedge
    loops = [buildloop(ir, header, latch, edges)
             for (header, edges) in backedges for latch in edges]
    # Sort outermost to innermost, then by header index
    sort!(loops, lt=(l1, l2) -> l2.body in l1.body || l1.header < l2.header)
    return loops
end

"Detect the set of natural loops for a piece of IR, pruning unreachable blocks."
detectloops!(ir::IR) = detectloops(pruneblocks!(IRTools.explicitbranch!(ir)))

"Insert preheader for a block in a piece of IR."
function preheader!(ir::IR, i::Int, backedges::AbstractVector{Int}=Int[])
    # Insert preheader, add arguments
    preheader = block!(ir, i)
    header = block(ir, i+1)
    args = [argument!(preheader, insert = false) for a in arguments(header)]
    # Rewire branches from entering blocks to preheader
    backedges = map(e -> e >= i ? e+1 : e, backedges)
    entrances = [p for p in predecessors(header) if !(p.id in backedges)]
    for e in entrances, (bi, br) in enumerate(branches(e))
        if br.block != header.id continue end
        branches(e)[bi] = Branch(br, block=preheader.id)
    end
    # Branch from preheader to header
    branch!(preheader, header, args...)
    return preheader
end

"Insert preheader for a loop."
preheader!(ir::IR, loop::NaturalLoop) =
    preheader!(ir, loop.header, collect(loop.backedges))

"Insert preheaders for every loop in a piece of IR."
function preheaders!(ir::IR, loops::AbstractVector{NaturalLoop})
    preheaders, inserts = Block[], Int[]
    for (n, loop) in enumerate(sort!(loops, by=l->l.header, rev=true))
        backedges = map(e -> e + sum(inserts .<= e), collect(loop.backedges))
        preheader!(ir, loop.header, backedges)
        pushfirst!(preheaders, block(ir, loop.header + length(loops) - n))
        push!(inserts, loop.header) # Track which indices have been inserted
    end
    return preheaders
end

"Add loop counters to each natural loop in a piece of IR."
function loopcounts!(ir::IR)
    # Prune unreachable blocks, detect loops, and add preheaders
    preheaders!(ir, detectloops!(ir))
    # Iterate over loops in topological order
    prev_header = -1
    loops, countvars = NaturalLoop[], Variable[]
    for loop in detectloops(ir)
        # Skip nested loops with the same header
        if loop.header == prev_header continue end
        # Create loop count variables, and pass them along blocks
        preheader, header = block(ir, loop.header-1), block(ir, loop.header)
        push!(arguments(branches(preheader)[1]), 0) # Initialize loop count
        count = argument!(header, insert=false) # Pass to header
        count = pushfirst!(header, xcall(Base, :+, count, 1)) # Increment count
        for b in (block(ir, i) for i in loop.backedges)
            for (bi, br) in enumerate(branches(b)) # Pass count along backedges
                if br.block != loop.header continue end
                push!(arguments(branches(b)[bi]), count)
            end
        end
        push!(loops, loop); push!(countvars, count)
        prev_header = loop.header
    end
    return loops, countvars
end
