using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random
using EnsembleKalmanProcesses.DataContainers

using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods


seed = 2023

@testset "Methods" begin

    @testset "construction of RFM" begin
        rng = StableRNG(seed)
        
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
        batch_sizes_err = Dict(
            "train" => 100,
            "test" => 100,
            "NOT_FEATURES" => 100
        )
        batch_sizes = Dict(
            "train" => 100,
            "test" => 100,
            "feature" => 100
        )
        lambda_warn = -1
        lambda = 1e-4
        
        @test_throws ArgumentError RandomFeatureMethod(
            sff,
            regularization = lambda,
            batch_sizes = batch_sizes_err,
        )
        
        rfm_warn = RandomFeatureMethod(
            sff,
            regularization = lambda_warn,
            batch_sizes = batch_sizes,
        )
        @test get_regularization(rfm_warn) ≈ 1e12*eps()
        
        rfm = RandomFeatureMethod(
            sff,
            regularization = lambda,
            batch_sizes = batch_sizes
        )
        
        @test get_batch_sizes(rfm) == batch_sizes
        rf_test = get_random_feature(rfm)
        #too arduous right now to check rf_test == sff will wait until "==" is overloaded for ParameterDistribution

        rfm_default = RandomFeatureMethod(sff)

        @test get_batch_sizes(rfm_default) == Dict("train" => 0, "test" => 0, "feature" => 0)
        @test get_regularization(rfm_default) ≈ 1e12*eps()
    end

    @testset "Fit and predict" begin
        rng = StableRNG(seed)
        
        #problem formulation
        n_data = 30
        x = rand(rng, Uniform(0,2*pi), n_data)
        noise = rand(rng, Normal(0,0.1), n_data) 
        y = sin.(x) + noise
        io_pairs = PairedDataContainer(
            reshape(x,1,:),
            reshape(y,1,:),
            data_are_columns=true
        ) #matrix input
        
        #specify features 
        μ_c = 0.0
        σ_c = 2.0
        pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
        feature_sampler = FeatureSampler(pd, rng=copy(rng))
        
        n_features = 100
        sigma_fixed = Dict("sigma" => 1.0)
        sff = ScalarFourierFeature(
            n_features,
            feature_sampler,
            hyper_fixed =  sigma_fixed
        )

        #first case without batches
        rfm = RandomFeatureMethod(sff)
        
        fitted_features = fit(rfm, io_pairs)
        decomp = get_feature_factors(fitted_features)
        @test typeof(decomp) == Decomposition
        @test typeof(get_decomposition(decomp)) <: SVD
        
        coeffs = get_coeffs(fitted_features)

        #second case with batching

        batch_sizes = Dict("train" => 10, "test" => 0, "feature" => 20)
        
        rfm_batch = RandomFeatureMethod(sff, batch_sizes=batch_sizes)

        fitted_batched_features = fit(rfm_batch, io_pairs)
        coeffs_batched = get_coeffs(fitted_batched_features)
        @test coeffs ≈ coeffs_batched

        # test prediction
        

        

    end
    

    

    
end
