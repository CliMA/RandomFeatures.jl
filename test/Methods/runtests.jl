using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random

using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods

seed = 2023

@testset "RandomFeatureMethods" begin

    rng = StableRNG(seed)

    #problem formulation
    n_data = 30
    x = rand(rng, Uniform(0,2*pi), n_data)
    noise = rand(rng, Normal(0,0.1), n_data) 
    y = sin.(x) + noise

    #specify features
    μ_c = 0.0
    σ_c = 2.0
    pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
    feature_sampler = FeatureSampler(pd, rng=copy(rng))

    n_features = 100
    sigma_fixed = Dict("sigma" => 10.0)
    sff = ScalarFourierFeature(
        n_features,
        feature_sampler,
        hyper_fixed =  sigma_fixed
    )

    # configure the method, and fit
    batch_sizes = Dict(
        "train" => 100,
        "test" => 100,
        "features" => 100
    )

    lambda_warn = -1
    lambda = 1e-4
    
    @test_throws ArgumentError RandomFeatureMethod(
        sff,
        batch_sizes,
        regularization = lambda
    )

    

    rfm = RandomFeatureMethod(
        sff,
        batch_sizes,
        regularization = lambda
    )

    
    

    

    
end
