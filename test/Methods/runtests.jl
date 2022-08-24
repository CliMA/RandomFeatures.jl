using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random

using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

using RandomFeatures.Utilities
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
        n_data = 20
        x = rand(rng, Uniform(-3,3), n_data)
        noise_sd = 0.1
        noise = rand(rng, Normal(0,noise_sd), n_data)

        # looks like a 4th order polynomial near 0, then is damped to 0 toward +/- inf
        ftest(x::AbstractVecOrMat) = exp.(-x.^2) .* (x.^4 - x.^2 + 3*x .- 1)

        y = ftest(x) + noise
        io_pairs = PairedDataContainer(
            reshape(x,1,:),
            reshape(y,1,:),
            data_are_columns=true
        ) #matrix input

        xtestvec = collect(-3:0.1:3)
        ntest = length(xtestvec) #extended domain
        xtest = DataContainer(reshape(xtestvec,1,:), data_are_columns=true)
        ytest = ftest(get_data(xtest))
        
        #specify features (lengthscale and sigma are magic numbers)
        μ_c = 0.0
        σ_c = 1.0
        pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
        feature_sampler = FeatureSampler(pd, rng=copy(rng))
        feature_sampler_snf = FeatureSampler(pd, rng=copy(rng))
        feature_sampler_ssf = FeatureSampler(pd, rng=copy(rng))
        
        n_features = 100
        sigma_fixed = Dict("sigma" => 3.0)
        sff = ScalarFourierFeature(
            n_features,
            feature_sampler,
            hyper_fixed =  sigma_fixed
        )

        snf = ScalarNeuronFeature(
            n_features,
            feature_sampler_snf,
            hyper_fixed =  sigma_fixed
        )

        ssf = ScalarFeature(
            n_features,
            feature_sampler_ssf,
            Sigmoid(),
            hyper_fixed =  sigma_fixed,
        )
        #first case without batches
        lambda = noise_sd^2
        rfm = RandomFeatureMethod(sff, regularization=lambda)
        
        fitted_features = fit(rfm, io_pairs)
        decomp = get_feature_factors(fitted_features)
        @test typeof(decomp) == Decomposition
        @test typeof(get_decomposition(decomp)) <: SVD
        
        coeffs = get_coeffs(fitted_features)

        #second case with batching
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        
        rfm_batch = RandomFeatureMethod(sff, batch_sizes=batch_sizes, regularization=lambda)

        fitted_batched_features = fit(rfm_batch, io_pairs)
        coeffs_batched = get_coeffs(fitted_batched_features)
        @test coeffs ≈ coeffs_batched
        
        # test prediction with different features
        pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)

        rfm_relu = RandomFeatureMethod(snf, batch_sizes=batch_sizes, regularization=lambda)
        fitted_relu_features = fit(rfm_relu, io_pairs)
        pred_mean_relu, pred_cov_relu = predict(rfm_relu, fitted_relu_features, xtest)

        rfm_sig = RandomFeatureMethod(ssf, batch_sizes=batch_sizes, regularization=lambda)
        fitted_sig_features = fit(rfm_sig, io_pairs)
        pred_mean_sig, pred_cov_sig = predict(rfm_sig, fitted_sig_features, xtest)

        prior_mean, prior_cov = predict_prior(rfm_batch, xtest) # predict inputs from unfitted features
        prior_mean_relu, prior_cov_relu = predict_prior(rfm_relu, xtest)
        prior_mean_sig, prior_cov_sig = predict_prior(rfm_sig, xtest)
        
        # added Plots for these different predictions:
        if TEST_PLOT_FLAG
            
            clrs = map(x->get(colorschemes[:hawaii],x),[0.25,0.5, 0.75])
            plt = plot(get_data(xtest)', ytest', show=false, color="black", linewidth=5, size = (600,600), legend=:topleft, label="Target" )
            #plot!(get_data(xtest)', prior_mean', linestyle=:dash, ribbon = [2*sqrt.(prior_cov); 2*sqrt.(prior_cov)]',label="", alpha=0.5, color=clrs[1])        
            plot!(get_data(xtest)', pred_mean', ribbon = [2*sqrt.(pred_cov); 2*sqrt.(pred_cov)]', label="Fourier", color=clrs[1]) 
            #plot!(get_data(xtest)', prior_mean_relu', ribbon = [2*sqrt.(prior_cov_relu); 2*sqrt.(prior_cov_relu)]', linestyle=:dash, label="", alpha=0.5, color=clrs[2])
            plot!(get_data(xtest)', pred_mean_relu', ribbon = [2*sqrt.(pred_cov_relu); 2*sqrt.(pred_cov_relu)]', label="Relu", color=clrs[2])
            #plot!(get_data(xtest)', prior_mean_sig', ribbon = [2*sqrt.(prior_cov_sig); 2*sqrt.(prior_cov_sig)]', linestyle=:dash, label="", alpha=0.5, color=clrs[3])
            plot!(get_data(xtest)', pred_mean_sig', ribbon = [2*sqrt.(pred_cov_sig); 2*sqrt.(pred_cov_sig)]', label="Sigmoid", color=clrs[3])
            
            scatter!(x, y, markershape=:x, label="", color="black", markersize=6)
            savefig(plt, joinpath(@__DIR__,"Methods_test_1.pdf"))
        end
        println("L^2 error: ", sqrt(sum((ytest - pred_mean).^2)))
        println("normalized L^2 error: ", sqrt(sum(1 ./ pred_cov .*(ytest - pred_mean).^2)))
         
    end
    

    

    
end
