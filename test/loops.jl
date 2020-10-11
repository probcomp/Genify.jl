@testset "Loop utilities" begin

function looper1(N::Int, K::Int)
    x = 0
    for i in 1:N
        if (i < 20) continue end
        for j in 1:K
            x += 1
        end
    end
    return x
end

function looper2(N::Int)
    x = 0
    for i in 1 : N
        if (i > 10)
            x += 1
        else
            x += 2
        end
    end
    return x
end

function looper3(N::Int)
    x = 0
    for i in 1 : N
        for j in 1 : i
            for k in i : j
                for l in 1 : 10
                    x += 1
                end
            end
        end
    end
    return x
end

@testset "Loop detection" begin

ir = IR(typeof(looper1), Int, Int)
l1, l2 = Genify.detectloops!(ir)
@test l2.body ⊆ l1.body
@test looper1(10, 10) == IRTools.evalir(ir, nothing, 10, 10)

ir = IR(typeof(looper2), Int)
loops = Genify.detectloops!(ir)
@test length(loops) == 1
@test looper2(20) == IRTools.evalir(ir, nothing, 20)

ir = IR(typeof(looper3), Int)
l1, l2, l3, l4 = Genify.detectloops!(ir)
@test l4.body ⊆ l3.body ⊆ l2.body ⊆ l1.body
@test looper3(20) == IRTools.evalir(ir, nothing, 20)

end

@testset "Preheader insertion" begin

ir = IR(typeof(looper1), Int, Int)
preheaders = Genify.preheaders!(ir, Genify.detectloops!(ir))
loops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in loops)
@test looper1(10, 10) == IRTools.evalir(ir, nothing, 10, 10)

ir = IR(typeof(looper2), Int)
preheaders = Genify.preheaders!(ir, Genify.detectloops!(ir))
loops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in loops)
@test looper2(20) == IRTools.evalir(ir, nothing, 20)

ir = IR(typeof(looper3), Int)
preheaders = Genify.preheaders!(ir, Genify.detectloops!(ir))
loops = Genify.detectloops(ir)
@test all(IRTools.block(ir, l.header-1) in preheaders for l in loops)
@test looper3(20) == IRTools.evalir(ir, nothing, 20)

end

@testset "Loop count insertion" begin

ir = IR(typeof(looper1), Int, Int)
loops, countvars = Genify.loopcounts!(ir)
@test length(loops) == length(countvars) == 2
@test looper1(10, 10) == IRTools.evalir(ir, nothing, 10, 10)

ir = IR(typeof(looper2), Int)
loops, countvars = Genify.loopcounts!(ir)
@test length(loops) == length(countvars) == 1
@test looper2(20) == IRTools.evalir(ir, nothing, 20)

ir = IR(typeof(looper3), Int)
loops, countvars = Genify.loopcounts!(ir)
@test length(loops) == length(countvars) == 4
@test looper3(20) == IRTools.evalir(ir, nothing, 20)

end

end
