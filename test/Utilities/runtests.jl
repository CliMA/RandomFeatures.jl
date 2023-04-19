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


    # Decomposition - only pinv, chol and svd available

    M = 30
    x = [i + j for i in 1:M, j in 1:M]
    x = 1 ./ (x + x') + 1e-3 * I
    # internally RHS are stored with three indices.
    # (n_samples, output_dim, n_features)
    b = ones(1, 1, M)
    xsolve = zeros(size(b))
    xsolve[1, 1, :] = x \ b[1, 1, :]

    xsvd = Decomposition(x, "svd")
    @test get_decomposition(xsvd) == svd(x)
    @test get_parametric_type(xsvd) == Factor
    @test get_full_matrix(xsvd) == x
    @test get_inv_decomposition(xsvd) ≈ inv(svd(x))

    xpinv = Decomposition(x, "pinv")
    @test isposdef(x)
    xchol = Decomposition(x, "cholesky")
    @test_throws ArgumentError Decomposition(x, "qr")

    xbad = [1.0 1.0; 1.0 0.0] # not pos def
    @test_logs (:info,) Decomposition(xbad, "cholesky")
    xbadchol = Decomposition(xbad, "cholesky")
    @test isposdef(get_full_matrix(xbadchol))

    xsvdsolve = linear_solve(xsvd, b)
    xcholsolve = linear_solve(xchol, b)
    xpinvsolve = linear_solve(xpinv, b)

    @test xsvdsolve ≈ xsolve
    @test xcholsolve ≈ xsolve
    @test xpinvsolve ≈ xsolve

    xsvdvecsolve = linear_solve(xsvd, vec(b))
    xcholvecsolve = linear_solve(xchol, vec(b))
    xpinvvecsolve = linear_solve(xpinv, vec(b))

    @test xsvdvecsolve ≈ vec(xsolve)
    @test xcholvecsolve ≈ vec(xsolve)
    @test xpinvvecsolve ≈ vec(xsolve)


    y = Float64[x for i in 1:M, x in 1:M]
    ypinv = Decomposition(y, "pinv")
    @test get_decomposition(ypinv) == pinv(y)
    @test get_parametric_type(ypinv) == PseInv
    @test get_full_matrix(ypinv) == y

    # to show pinv gets the right solution in a singular problem
    @test_throws SingularException y \ b[1, 1, :]
    ysolve = zeros(size(b))
    ysolve[1, 1, :] = pinv(y) * b[1, 1, :]
    @test linear_solve(ypinv, b) ≈ ysolve
    ysolvevec = pinv(y) * vec(b)
    @test linear_solve(ypinv, vec(b)) ≈ ysolvevec

end
