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
        feature_sampler = FeatureSampler(pd, rng = copy(rng))

        n_features = 100
        sigma_fixed = Dict("sigma" => 10.0)

        sff = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed)

        # configure the method, and fit 
        batch_sizes_err = Dict("train" => 100, "test" => 100, "NOT_FEATURES" => 100)
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        lambda_warn = -1
        lambda = 1e-4

        @test_throws ArgumentError RandomFeatureMethod(sff, regularization = lambda, batch_sizes = batch_sizes_err)

        rfm_warn = RandomFeatureMethod(sff, regularization = lambda_warn, batch_sizes = batch_sizes)
        @test get_regularization(rfm_warn) ≈ 1e12 * eps()

        rfm = RandomFeatureMethod(sff, regularization = lambda, batch_sizes = batch_sizes)

        @test get_batch_sizes(rfm) == batch_sizes
        rf_test = get_random_feature(rfm)
        #too arduous right now to check rf_test == sff will wait until "==" is overloaded for ParameterDistribution

        rfm_default = RandomFeatureMethod(sff)

        @test get_batch_sizes(rfm_default) == Dict("train" => 0, "test" => 0, "feature" => 0)
        @test get_regularization(rfm_default) ≈ 1e12 * eps()
    end

    @testset "Fit and predict: 1D -> 1D" begin
        rng_base = StableRNG(seed)

        # looks like a 4th order polynomial near 0, then is damped to 0 toward +/- inf
        ftest(x::AbstractVecOrMat) = exp.(-0.5 * x .^ 2) .* (x .^ 4 - x .^ 3 - x .^ 2 + x .- 1)

        exp_range = [1, 2, 4]
        n_data_exp = 8 * exp_range


        L2err = zeros(length(exp_range), 3)
        weightedL2err = zeros(length(exp_range), 3)

        # values with 1/var learning in 1d_to_1d_regression
        σ_c_vec = [0.93, 3.06, 2.45]
        σ_c_snf_vec = [0.97, 4.18, 3.57]
        σ_c_ssf_vec = [1.65, 4.62, 5.41]
        for (exp_idx, n_data, σ_c, σ_c_snf, σ_c_ssf) in
            zip(1:length(exp_range), n_data_exp, σ_c_vec, σ_c_snf_vec, σ_c_ssf_vec)

            rng = copy(rng_base)
            #problem formulation
            x = rand(rng, Uniform(-3, 3), n_data)
            noise_sd = 0.1
            noise = rand(rng, Normal(0, noise_sd), n_data)


            y = ftest(x) + noise
            io_pairs = PairedDataContainer(reshape(x, 1, :), reshape(y, 1, :), data_are_columns = true) #matrix input

            xtestvec = collect(-3:0.01:3)
            ntest = length(xtestvec) #extended domain
            xtest = DataContainer(reshape(xtestvec, 1, :), data_are_columns = true)
            ytest = ftest(get_data(xtest))

            # specify feature distributions
            # NB we optimize hyperparameter values (σ_c,"sigma") in examples/Learn_hyperparameters/1d_to_1d_regression.jl
            # Such values may change with different ftest and different noise_sd

            sigma_fixed = Dict("sigma" => 1.0)
            n_features = 80

            μ_c = 0.0
            pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
            feature_sampler = FeatureSampler(pd, rng = copy(rng))

            sff = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed)

            sff = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed)

            pd_snf = constrained_gaussian("xi", μ_c, σ_c_snf, -Inf, Inf)
            feature_sampler_snf = FeatureSampler(pd_snf, rng = copy(rng))
            snf = ScalarNeuronFeature(n_features, feature_sampler_snf, hyper_fixed = sigma_fixed)

            pd_ssf = constrained_gaussian("xi", μ_c, σ_c_ssf, -Inf, Inf)
            feature_sampler_ssf = FeatureSampler(pd_ssf, rng = copy(rng))
            ssf = ScalarFeature(n_features, feature_sampler_ssf, Sigmoid(), hyper_fixed = sigma_fixed)
            #first case without batches
            lambda = noise_sd^2
            rfm = RandomFeatureMethod(sff, regularization = lambda)

            fitted_features = fit(rfm, io_pairs)
            decomp = get_feature_factors(fitted_features)
            @test typeof(decomp) == Decomposition{Factor}
            @test typeof(get_decomposition(decomp)) <: SVD

            coeffs = get_coeffs(fitted_features)

            #second case with batching
            batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)

            rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = lambda)

            fitted_batched_features = fit(rfm_batch, io_pairs)
            coeffs_batched = get_coeffs(fitted_batched_features)
            @test coeffs ≈ coeffs_batched

            # test prediction with different features
            pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)

            rfm_relu = RandomFeatureMethod(snf, batch_sizes = batch_sizes, regularization = lambda)
            fitted_relu_features = fit(rfm_relu, io_pairs)
            pred_mean_relu, pred_cov_relu = predict(rfm_relu, fitted_relu_features, xtest)

            rfm_sig = RandomFeatureMethod(ssf, batch_sizes = batch_sizes, regularization = lambda)
            fitted_sig_features = fit(rfm_sig, io_pairs)
            pred_mean_sig, pred_cov_sig = predict(rfm_sig, fitted_sig_features, xtest)

            prior_mean, prior_cov = predict_prior(rfm_batch, xtest) # predict inputs from unfitted features
            prior_mean_relu, prior_cov_relu = predict_prior(rfm_relu, xtest)
            prior_mean_sig, prior_cov_sig = predict_prior(rfm_sig, xtest)

            # added Plots for these different predictions:
            if TEST_PLOT_FLAG

                clrs = map(x -> get(colorschemes[:hawaii], x), [0.25, 0.5, 0.75])
                plt = plot(
                    get_data(xtest)',
                    ytest',
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )
                #plot!(get_data(xtest)', prior_mean', linestyle=:dash, ribbon = [2*sqrt.(prior_cov); 2*sqrt.(prior_cov)]',label="", alpha=0.5, color=clrs[1])        
                plot!(
                    get_data(xtest)',
                    pred_mean',
                    ribbon = [2 * sqrt.(pred_cov); 2 * sqrt.(pred_cov)]',
                    label = "Fourier",
                    color = clrs[1],
                )
                #plot!(get_data(xtest)', prior_mean_relu', ribbon = [2*sqrt.(prior_cov_relu); 2*sqrt.(prior_cov_relu)]', linestyle=:dash, label="", alpha=0.5, color=clrs[2])
                plot!(
                    get_data(xtest)',
                    pred_mean_relu',
                    ribbon = [2 * sqrt.(pred_cov_relu); 2 * sqrt.(pred_cov_relu)]',
                    label = "Relu",
                    color = clrs[2],
                )
                #plot!(get_data(xtest)', prior_mean_sig', ribbon = [2*sqrt.(prior_cov_sig); 2*sqrt.(prior_cov_sig)]', linestyle=:dash, label="", alpha=0.5, color=clrs[3])
                plot!(
                    get_data(xtest)',
                    pred_mean_sig',
                    ribbon = [2 * sqrt.(pred_cov_sig); 2 * sqrt.(pred_cov_sig)]',
                    label = "Sigmoid",
                    color = clrs[3],
                )

                scatter!(x, y, markershape = :x, label = "", color = "black", markersize = 6)
                savefig(plt, joinpath(@__DIR__, "Fit_and_predict_1D_" * string(exp_range[exp_idx]) * ".pdf"))
            end
            L2err[exp_idx, :] += [
                sqrt(sum((ytest - pred_mean) .^ 2)),
                sqrt(sum((ytest - pred_mean_relu) .^ 2)),
                sqrt(sum((ytest - pred_mean_sig) .^ 2)),
            ]
            weightedL2err[exp_idx, :] += [
                sqrt(sum(1 ./ pred_cov .* (ytest - pred_mean) .^ 2)),
                sqrt(sum(1 ./ pred_cov_relu .* (ytest - pred_mean_relu) .^ 2)),
                sqrt(sum(1 ./ pred_cov_sig .* (ytest - pred_mean_sig) .^ 2)),
            ]
        end

        #these tests can be a bit brittle
        println(L2err)
        @test all([all(L2err[i, :] .< L2err[i - 1, :]) for i in 2:size(L2err, 1)])
        #println(weightedL2err)
        #        @test all([all(weightedL2err[i,:] .< weightedL2err[i-1,:]) for i=2:size(weightedL2err,1)])





    end # testset "Fit and predict"

    @testset "Fit and predict: N-D -> 1-D" begin

        rng = StableRNG(seed + 1)
        input_dim = 10
        n_features = 1001
        ftest_nd_to_1d(x::AbstractMatrix) = mapslices(column -> cos(2 * pi * norm(column)), x, dims = 1)

        #problem formulation
        n_data = 800
        x = rand(rng, Uniform(-3, 3), (input_dim, n_data))
        noise_sd = 0.01
        lambda = noise_sd^2
        noise = rand(rng, Normal(0, noise_sd), (1, n_data))

        y = ftest_nd_to_1d(x) + noise
        io_pairs = PairedDataContainer(x, y)

        n_test = 10000
        xtestvec = rand(rng, Uniform(-3, 3), (input_dim, n_test))

        xtest = DataContainer(xtestvec)
        ytest = ftest_nd_to_1d(get_data(xtest))


        # specify features
        # note the σ_c and sigma values come from `examples/Learn_hyperparameters/nd_to_1d_regression.jl`
        μ_c = 0.0

        σ_c = 15.0
        pd = ParameterDistribution(
            Dict(
                "distribution" => VectorOfParameterized(repeat([Normal(μ_c, σ_c)], input_dim)),
                "constraint" => repeat([no_constraint()], input_dim),
                "name" => "xi",
            ),
        )

        feature_sampler = FeatureSampler(pd, rng = copy(rng))
        sigma_fixed = Dict("sigma" => 1.0)
        sff = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed)

        #second case with batching
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)

        rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = lambda)

        fitted_batched_features = fit(rfm_batch, io_pairs)
        # test prediction with different features
        pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)

        L2err = sqrt(sum((ytest - pred_mean) .^ 2))
        println("L2err: ", L2err)

        if TEST_PLOT_FLAG
            #plot slice through one dimensions, others fixed to 0
            xrange = collect(-3:0.01:3)
            xslice = zeros(input_dim, length(xrange))
            xslice[1, :] = xrange

            yslice = ftest_nd_to_1d(xslice)

            pred_mean_slice, pred_cov_slice = predict(rfm_batch, fitted_batched_features, DataContainer(xslice))
            plt = plot(
                reshape(xslice[1, :], :, 1),
                yslice',
                show = false,
                color = "black",
                linewidth = 5,
                size = (600, 600),
                legend = :topleft,
                label = "Target",
            )
            plot!(
                reshape(xslice[1, :], :, 1),
                pred_mean_slice',
                ribbon = [2 * sqrt.(pred_cov_slice); 2 * sqrt.(pred_cov_slice)]',
                label = "Fourier",
                color = "blue",
            )
            savefig(plt, joinpath(@__DIR__, "Fit_and_predict_ND.pdf"))
        end





    end # testset "Fit and predict"

    @testset "Fit and predict: N-D -> 1-D" begin

        rng = StableRNG(seed + 1)
        input_dim = 10
        n_features = 1001
        ftest_nd_to_1d(x::AbstractMatrix) = mapslices(column -> cos(2 * pi * norm(column)), x, dims = 1)

        #problem formulation
        n_data = 800
        x = rand(rng, Uniform(-3, 3), (input_dim, n_data))
        noise_sd = 0.01
        lambda = noise_sd^2
        noise = rand(rng, Normal(0, noise_sd), (1, n_data))

        y = ftest_nd_to_1d(x) + noise
        io_pairs = PairedDataContainer(x, y)

        n_test = 10000
        xtestvec = rand(rng, Uniform(-3, 3), (input_dim, n_test))

        xtest = DataContainer(xtestvec)
        ytest = ftest_nd_to_1d(get_data(xtest))


        # specify features
        # note the σ_c and sigma values come from `examples/Learn_hyperparameters/nd_to_1d_regression.jl`
        μ_c = 0.0
        σ_c = 15.0
        pd = ParameterDistribution(
            Dict(
                "distribution" => VectorOfParameterized(repeat([Normal(μ_c, σ_c)], input_dim)),
                "constraint" => repeat([no_constraint()], input_dim),
                "name" => "xi",
            ),
        )
        feature_sampler = FeatureSampler(pd, rng = copy(rng))
        sigma_fixed = Dict("sigma" => 1.0)
        sff = ScalarFourierFeature(n_features, feature_sampler, hyper_fixed = sigma_fixed)

        #second case with batching
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)

        rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = lambda)

        fitted_batched_features = fit(rfm_batch, io_pairs)
        # test prediction with different features
        pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)

        L2err = sqrt(sum((ytest - pred_mean) .^ 2))
        println("L2err: ", L2err)

        if TEST_PLOT_FLAG
            #plot slice through one dimensions, others fixed to 0
            xrange = collect(-3:0.01:3)
            xslice = zeros(input_dim, length(xrange))
            xslice[1, :] = xrange

            yslice = ftest_nd_to_1d(xslice)

            pred_mean_slice, pred_cov_slice = predict(rfm_batch, fitted_batched_features, DataContainer(xslice))
            plt = plot(
                reshape(xslice[1, :], :, 1),
                yslice',
                show = false,
                color = "black",
                linewidth = 5,
                size = (600, 600),
                legend = :topleft,
                label = "Target",
            )
            plot!(
                reshape(xslice[1, :], :, 1),
                pred_mean_slice',
                ribbon = [2 * sqrt.(pred_cov_slice); 2 * sqrt.(pred_cov_slice)]',
                label = "Fourier",
                color = "blue",
            )
            savefig(plt, joinpath(@__DIR__, "Fit_and_predict_ND.pdf"))
        end

    end


end
