using Test
#using Distributions
#using StableRNGs
#using StatsBase
using LinearAlgebra
#using Random

using RandomFeatures.Utilities


@testset "Utilities" begin

    # batch generator

    x = [i + j for i in 1:10, j in 1:234]

    batch_size = 10
    x1 = batch_generator(x, batch_size) #default dims=1
    @test length(x1) == Int(ceil(size(x, 1) / batch_size))
    @test x1[1] == x

    x2 = batch_generator(x, 10, dims = 2)
    @test length(x2) == Int(ceil(size(x, 2) / batch_size))

    for i in 1:length(x2)
        @test x2[i] == (
            i < length(x2) ? x[:, ((i - 1) * batch_size + 1):(i * batch_size)] : x[:, ((i - 1) * batch_size + 1):end]
        )
    end

    x3 = batch_generator(x, 0) #default dims=1
    @test length(x3) == 1
    @test x3[1] == x


    # Decomposition
    N = 30
    x = [i + j for i in 1:N, j in 1:N]
    x = 1 ./ (x + x') + 1e-3 * I

    b = ones(N)

    xsolve = x \ b

    xsvd = Decomposition(x, "svd")
    @test get_decomposition(xsvd) == svd(x)
    @test get_decomposition_is_inverse(xsvd) == false
    @test get_full_matrix(xsvd) == x

    xqr = Decomposition(x, "qr")
    xlu = Decomposition(x, "lu")
    @test isposdef(x)
    xchol = Decomposition(x, "cholesky")
    xpinv = Decomposition(x, "pinv")

    xsvdsolve = linear_solve(xsvd, b)
    xqrsolve = linear_solve(xqr, b)
    xlusolve = linear_solve(xlu, b)
    xcholsolve = linear_solve(xchol, b)
    xpinvsolve = linear_solve(xpinv, b)

    @test xsvdsolve ≈ xsolve
    @test xqrsolve ≈ xsolve
    @test xlusolve ≈ xsolve
    @test xcholsolve ≈ xsolve
    @test xpinvsolve ≈ xsolve

    y = [x for i in 1:N, x in 1:N]
    ypinv = Decomposition(y, "pinv")
    @test get_decomposition(ypinv) == pinv(y)
    @test get_decomposition_is_inverse(ypinv) == true
    @test get_full_matrix(ypinv) == y

    # to show pinv gets the right solution in a singular problem
    @test_throws SingularException y \ b
    ysolve = pinv(y) * b
    @test linear_solve(ypinv, b) ≈ ysolve


end
