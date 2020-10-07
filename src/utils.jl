struct NaturalLoop
    header::Block
    in_loop::Dict{Block, Bool}
    backedge::Block
end

const Loops = Vector{NaturalLoop}

function edges(ir)
    edges = Dict{Block, Block}()
    visited = Block[]
    for bb in blocks(ir)
        for tg in filter(x -> !(x in visited), blocks(ir))
            !isempty(branches(bb, tg)) && edges[bb] = tg
        end
    end
    edges
end

function loop_detection(ir)
    dtree = domtree(ir, entry = 1)
    all_edges = edges(ir)
    println(all_edges)
end
