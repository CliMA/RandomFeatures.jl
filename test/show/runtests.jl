using Test
using LinearAlgebra
using Random
using StableRNGs
using Distributions

using RandomFeatures.Utilities
using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods
using RandomFeatures.DataContainers
using RandomFeatures.ParameterDistributions

seed = 9999

function check_full(x, typename)
    out = sprint(show, MIME("text/plain"), x)
    @test occursin(typename, out)
    @test count(==('\n'), out) <= 10
end

function check_compact(x, typename)
    out2 = sprint(show, x)
    @test occursin(typename, out2)
    @test !occursin('\n', out2)
    out3 = sprint(show, MIME("text/plain"), x; context = :compact => true)
    @test out2 == out3
end

function check_summary(x, typename)
    out = sprint(summary, x)
    @test occursin(typename, out)
    @test !occursin('\n', out)
end

@testset "show" begin

    rng = StableRNG(seed)
    M = 8
    mat = Matrix(1.0I, M, M) + 0.1 * (rand(rng, M, M) + rand(rng, M, M)')

    @testset "Decomposition (svd)" begin
        dc = Decomposition(mat, "svd")
        check_full(dc, "Decomposition")
        check_compact(dc, "Decomposition")
        check_summary(dc, "Decomposition")
    end

    @testset "Decomposition (cholesky)" begin
        dc = Decomposition(mat, "cholesky")
        check_full(dc, "Decomposition")
        check_compact(dc, "Decomposition")
        check_summary(dc, "Decomposition")
    end

    @testset "Decomposition (pinv)" begin
        dc = Decomposition(mat, "pinv")
        check_full(dc, "Decomposition")
        check_compact(dc, "Decomposition")
        check_summary(dc, "Decomposition")
    end

    μ_c = 0.0
    σ_c = 1.0
    pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
    fsampler = FeatureSampler(pd, rng = copy(rng))

    @testset "Sampler" begin
        check_full(fsampler, "Sampler")
        check_compact(fsampler, "Sampler")
        check_summary(fsampler, "Sampler")
    end

    n_features = 64

    @testset "ScalarFeature (Cosine)" begin
        sf = ScalarFourierFeature(n_features, FeatureSampler(pd, rng = copy(rng)))
        check_full(sf, "ScalarFeature")
        check_compact(sf, "ScalarFeature")
        check_summary(sf, "ScalarFeature")
        # hint contains feature count
        @test occursin(string(n_features), sprint(show, sf))
    end

    @testset "ScalarFeature (Relu)" begin
        sf = ScalarNeuronFeature(n_features, FeatureSampler(pd, rng = copy(rng)))
        check_full(sf, "ScalarFeature")
        check_compact(sf, "ScalarFeature")
        check_summary(sf, "ScalarFeature")
    end

    output_dim = 3
    pd_nd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf, repeats = 4)
    vfsampler = FeatureSampler(pd_nd, output_dim, rng = copy(rng))

    @testset "VectorFeature" begin
        vf = VectorFourierFeature(n_features, output_dim, vfsampler)
        check_full(vf, "VectorFeature")
        check_compact(vf, "VectorFeature")
        check_summary(vf, "VectorFeature")
        # hint contains output_dim
        @test occursin(string(output_dim), sprint(show, vf))
    end

    @testset "RandomFeatureMethod" begin
        sf = ScalarFourierFeature(n_features, FeatureSampler(pd, rng = copy(rng)))
        rfm = RandomFeatureMethod(sf)
        check_full(rfm, "RandomFeatureMethod")
        check_compact(rfm, "RandomFeatureMethod")
        check_summary(rfm, "RandomFeatureMethod")
        @test occursin(string(n_features), sprint(show, rfm))
    end

    @testset "Fit" begin
        sf = ScalarFourierFeature(n_features, FeatureSampler(pd, rng = copy(rng)))
        rfm = RandomFeatureMethod(sf)
        n_data = 30
        x = rand(rng, 1, n_data)
        y = rand(rng, 1, n_data)
        io_pairs = PairedDataContainer(x, y, data_are_columns = true)
        f = fit(rfm, io_pairs)
        check_full(f, "Fit")
        check_compact(f, "Fit")
        check_summary(f, "Fit")
    end

end
