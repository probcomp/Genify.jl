struct NaturalLoop
    header::Int
    loop_body::Set{Int}
    backedge::Int
end

@inline header(nl::NaturalLoop) = getfield(nl, :header)
@inline body(nl::NaturalLoop) = getfield(nl, :loop_body)
@inline backedge(nl::NaturalLoop) = getfield(nl, :backedge)

function Base.display(nl::NaturalLoop)
    println(" ________________________ \n")
    println("       Natural loop\n")
    println(" Header : $(nl.header)")
    println(" Blocks : $(nl.loop_body)")
    println(" Backedge : $(nl.backedge)")
    println("\n ________________________ \n")
end

const Loops = Vector{NaturalLoop}
Base.display(ls::Loops) = for l in ls
    display(l)
end

# Computes all the control flow edges between basic blocks in the IR.
function edges(ir::IR)
    edges = Dict{Int, Vector{Int}}()
    for bb in blocks(ir)
        for tg in blocks(ir)
            if !isempty(branches(bb, tg))
                if haskey(edges, bb.id)
                    push!(edges[bb.id], tg.id)
                else
                    edges[bb.id] = [tg.id]
                end
            end
        end
    end
    edges
end

# Filters the set of edges down to the set of backedges. Nodes are their own backedges.
function backedges(all_edges)
    backedges = Dict{Int, Vector{Int}}()
    for (k, v) in all_edges
        backedges[k] = filter(v) do el
            el <= k
        end
    end
    backedges
end

@inline backedges(ir::IR) = (all_edges = edges(ir); backedges(all_edges))

# Depth first search.
function dfs!(loop_body, domtree, source, target)
    source == target && return true
    any(map(collect(filter(key -> source > key, keys(domtree)))) do k
        if source in domtree[k] && dfs!(loop_body, domtree, k, target)
            push!(loop_body, k)
            true
        else
            false
        end
    end)
end

# Does "single" loop detection given a header block, a backedge block (with link to header), and the reference domtree.
function loop_detection(header, backedge, domtree)
    loop_body = Set{Int}([header, backedge])
    dfs!(loop_body, domtree, backedge, header)
    NaturalLoop(header, loop_body, backedge)
end

# Utility which flattens the Pair domtree to a Dict.
function flatten!(d, par, domtree::Pair)
    hd, tl = domtree
    d[hd] = Int[hd]
    for (k, v) in tl
        (push!(d[hd], k); push!(d[par], k);)
        flatten!(d, hd, k => v)
    end
end

function flatten(domtree::Pair)
    d = Dict{Int, Vector{Int}}()
    hd, tl = domtree
    d[hd] = Int[hd]
    for (k, v) in tl
        push!(d[hd], k)
        flatten!(d, hd, k => v)
    end
    return d
end

# Computes the set of natural loops for a piece of IR.
function loop_detection(ir; debug = false)
    dtree = flatten(domtree(ir, entry = 1))
    all_edges = edges(ir)
    b_edges = filter(backedges(all_edges)) do (k, v)
        !isempty(v)
    end
    debug && (println("All edges:"); display(all_edges); 
              println("\nBackedges:"); display(b_edges); 
              println("\nDomtree:"); display(dtree))
    Iterators.flatten(map(collect(b_edges)) do (k, v)
                          [loop_detection(head, k, dtree) for head in v]
                      end) |> collect
end
