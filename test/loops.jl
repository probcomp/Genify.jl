# ------------ Test funcs ------------ #

function foo1(N::Int, K::Int)
    for i in 1 : N
        i < 20 && continue
        for j in 1 : K
            x = rand(Normal(i + j, 1.0))
        end
    end
end

function foo2(N::Int)
    for i in 1 : N
        if i > 10
            x = rand(Normal(0.0, 1.0))
        else
            x = rand(Normal(0.0, 1.0))
        end
    end
end

function foo3(N::Int)
    for i in 1 : N
        for j in 1 : i
            for k in i : j
                for l in 1 : 10
                    println("LOOOOOOOPY")
                end
            end
        end
    end
end

# ------------ Testing ------------ #

@testset "Loop detection" begin

ir = @code_ir foo1(30, 10)
l1, l2 = Genify.detectloops!(ir)
@test l2.body ⊆ l1.body
preheaders = Genify.preheaders!(ir, [l1, l2])
newloops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in newloops)

ir = @code_ir foo2(10)
loops = Genify.detectloops!(ir)
@test length(loops) == 1
preheaders = Genify.preheaders!(ir, loops)
newloops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in newloops)

ir = @code_ir foo3(10)
l1, l2, l3, l4 = Genify.detectloops!(ir)
@test l4.body ⊆ l3.body ⊆ l2.body ⊆ l1.body
preheaders = Genify.preheaders!(ir, [l1, l2, l3, l4])
newloops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in newloops)

end
