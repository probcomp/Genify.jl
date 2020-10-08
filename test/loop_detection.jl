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
    ir = @code_ir foo1(5, 10)
    l1, l2 = Genify.loop_detection(ir)
    @test 4 == Genify.header(l1)
    @test 4 in Genify.body(l1)
    @test 4 == Genify.backedge(l1)

    ir = @code_ir foo2(10)
    l = Genify.loop_detection(ir)[1]
    @test 2 == Genify.header(l)
    @test 2 in Genify.body(l)
    @test 5 in Genify.body(l)
    @test 5 == Genify.backedge(l)
end
