using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random


using RandomFeatures.Utilities
using RandomFeatures.Samplers
using RandomFeatures.Features
using RandomFeatures.Methods
using RandomFeatures.DataContainers
using RandomFeatures.ParameterDistributions


seed = 2023
tol = 1e3 * eps()

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

        sff = ScalarFourierFeature(n_features, feature_sampler, feature_parameters = sigma_fixed)

        # configure the method, and fit 
        batch_sizes_err = Dict("train" => 100, "test" => 100, "NOT_FEATURES" => 100)
        batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)
        lambda_warn = -1
        lambda = 1e-4
        lambdamat_warn = ones(3, 3) # not pos def
        L = [0.5 0; 1.3 0.3]
        lambdamat = L * permutedims(L, (2, 1)) #pos def
        @test_throws ArgumentError RandomFeatureMethod(sff, regularization = lambda, batch_sizes = batch_sizes_err)

        rfm_warn = RandomFeatureMethod(sff, regularization = lambda_warn, batch_sizes = batch_sizes)
        @test get_regularization(rfm_warn) ≈ inv(1e12 * eps() * I) # inverted internally

        rfm_warn2 = RandomFeatureMethod(
            sff,
            regularization = lambdamat_warn,
            batch_sizes = batch_sizes,
            regularization_inverted = true,
        ) #don't invert, just make PD 
        reg_new = get_regularization(rfm_warn2)
        @test isposdef(reg_new)
        @test minimum(eigvals(reg_new)) > (1e12 - 1) * eps()

        rfm = RandomFeatureMethod(
            sff,
            regularization = lambdamat,
            batch_sizes = batch_sizes,
            regularization_inverted = true,
        )
        @test get_regularization(rfm) ≈ lambdamat

        rfm = RandomFeatureMethod(sff, regularization = lambdamat, batch_sizes = batch_sizes)
        @test get_regularization(rfm) ≈ inv(lambdamat)

        rfm = RandomFeatureMethod(sff, regularization = lambda, batch_sizes = batch_sizes)

        @test get_batch_sizes(rfm) == batch_sizes
        rf_test = get_random_feature(rfm)

        @test get_tullio_threading(rfm) == true
        rfm = RandomFeatureMethod(sff, tullio_threading = false)
        @test get_tullio_threading(rfm) == false

        #too arduous right now to check rf_test == sff will wait until "==" is overloaded for ParameterDistribution

        rfm_default = RandomFeatureMethod(sff)

        @test get_batch_sizes(rfm_default) == Dict("train" => 0, "test" => 0, "feature" => 0)
        @test get_regularization(rfm_default) ≈ inv(1e12 * eps() * I)
    end

    @testset "Fit and predict: 1-D -> 1-D" begin
        rng_base = StableRNG(seed)

        # looks like a 4th order polynomial near 0, then is damped to 0 toward +/- inf
        ftest(x::AbstractVecOrMat) = exp.(-0.5 * x .^ 2) .* (x .^ 4 - x .^ 3 - x .^ 2 + x .- 1)

        exp_range = [1, 2, 4]
        n_data_exp = 20 * exp_range


        priorL2err = zeros(length(exp_range), 3)
        priorweightedL2err = zeros(length(exp_range), 3)
        L2err = zeros(length(exp_range), 3)
        weightedL2err = zeros(length(exp_range), 3)

        # values with 1/var learning in examples/Learn_hyperparameters/1d_to_1d_regression_direct_withcov.jl

        σ_c_vec = [2.5903560156755194, 1.9826946095752571, 2.095420236641444]
        σ_c_snf_vec = [9.606414682837055, 4.406586351058134, 2.756419855446525]
        σ_c_ssf_vec = [2.2041952067873742, 3.0205667976224384, 4.307656997874708]

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
            ytest_nonoise = ftest(get_data(xtest))

            # specify feature distributions
            # NB we optimize hyperparameter values σ_c in examples/Learn_hyperparameters/1d_to_1d_regression.jl
            # Such values may change with different ftest and different noise_sd

            n_features = 400

            μ_c = 0.0
            pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
            feature_sampler = FeatureSampler(pd, rng = copy(rng))
            sff = ScalarFourierFeature(n_features, feature_sampler)

            pd_snf = constrained_gaussian("xi", μ_c, σ_c_snf, -Inf, Inf)
            feature_sampler_snf = FeatureSampler(pd_snf, rng = copy(rng))
            snf = ScalarNeuronFeature(n_features, feature_sampler_snf)

            pd_ssf = constrained_gaussian("xi", μ_c, σ_c_ssf, -Inf, Inf)
            feature_sampler_ssf = FeatureSampler(pd_ssf, rng = copy(rng))
            ssf = ScalarFeature(n_features, feature_sampler_ssf, Sigmoid())
            #first case without batches
            lambda = noise_sd^2
            rfm = RandomFeatureMethod(sff, regularization = lambda)

            fitted_features = fit(rfm, io_pairs)
            decomp = get_feature_factors(fitted_features)
            @test get_parametric_type(decomp) == Factor
            @test typeof(get_decomposition(decomp)) <: Cholesky

            coeffs = get_coeffs(fitted_features)

            #second case with batching
            batch_sizes = Dict("train" => 100, "test" => 100, "feature" => 100)

            rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = lambda)
            rfm_nothread = RandomFeatureMethod(sff, regularization = lambda, tullio_threading = false)

            fitted_batched_features = fit(rfm_batch, io_pairs)
            coeffs_batched = get_coeffs(fitted_batched_features)
            @test coeffs ≈ coeffs_batched

            fitted_features_nothread = fit(rfm_nothread, io_pairs)
            coeffs_nothread = get_coeffs(fitted_features_nothread)
            @test coeffs ≈ coeffs_nothread
            # test prediction with different features
            pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)
            if exp_idx == 1
                pmtmp = zeros(size(pred_mean)) # p x n
                pctmp = zeros(size(pred_cov)) # p x p x n
                buffer = zeros(size(pctmp, 3), size(pctmp, 1), n_features) # n x p x m
                predict!(rfm_batch, fitted_batched_features, xtest, pmtmp, pctmp, buffer)

                @test all(isapprox.(pred_mean, pmtmp, atol = tol))
                @test all(isapprox.(pred_cov, pctmp, atol = tol))
            end

            rfm_relu = RandomFeatureMethod(snf, batch_sizes = batch_sizes, regularization = lambda)
            fitted_relu_features = fit(rfm_relu, io_pairs)
            pred_mean_relu, pred_cov_relu = predict(rfm_relu, fitted_relu_features, xtest)

            rfm_sig = RandomFeatureMethod(ssf, batch_sizes = batch_sizes, regularization = lambda)
            fitted_sig_features = fit(rfm_sig, io_pairs)
            pred_mean_sig, pred_cov_sig = predict(rfm_sig, fitted_sig_features, xtest)

            prior_mean, prior_cov = predict_prior(rfm_batch, xtest) # predict inputs from unfitted features
            prior_mean_relu, prior_cov_relu = predict_prior(rfm_relu, xtest)
            prior_mean_sig, prior_cov_sig = predict_prior(rfm_sig, xtest)


            pred_mean_nothread, pred_cov_nothread =
                predict(rfm_nothread, fitted_features_nothread, xtest, tullio_threading = false)
            @test all(isapprox.(pred_mean_nothread, pred_mean, atol = tol))
            @test all(isapprox.(pred_cov_nothread, pred_cov, atol = tol))
            pmr_nothread = similar(pred_mean)
            pcr_nothread = similar(pred_cov)
            buffer = zeros(size(pred_cov, 3), size(pred_cov, 1), n_features) # n x p x m
            predict!(
                rfm_nothread,
                fitted_features_nothread,
                xtest,
                pmr_nothread,
                pcr_nothread,
                buffer,
                tullio_threading = false,
            )
            @test all(isapprox.(pmr_nothread, pred_mean, atol = tol))
            @test all(isapprox.(pcr_nothread, pred_cov, atol = tol))


            # enforce positivity
            prior_cov = max.(0, prior_cov)
            prior_cov_relu = max.(0, prior_cov_relu)
            prior_cov_sig = max.(0, prior_cov_sig)
            pred_cov = max.(0, pred_cov)
            pred_cov_relu = max.(0, pred_cov_relu)
            pred_cov_sig = max.(0, pred_cov_sig)
            # added Plots for these different predictions:
            if TEST_PLOT_FLAG

                clrs = map(x -> get(colorschemes[:hawaii], x), [0.25, 0.5, 0.75])
                plt = plot(
                    get_data(xtest)',
                    ytest_nonoise',
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )

                plot!(
                    get_data(xtest)',
                    pred_mean',
                    ribbon = [2 * sqrt.(pred_cov[1, 1, :]); 2 * sqrt.(pred_cov[1, 1, :])]',
                    label = "Fourier",
                    color = clrs[1],
                )

                plot!(
                    get_data(xtest)',
                    pred_mean_relu',
                    ribbon = [2 * sqrt.(pred_cov_relu[1, 1, :]); 2 * sqrt.(pred_cov_relu[1, 1, :])]',
                    label = "Relu",
                    color = clrs[2],
                )

                plot!(
                    get_data(xtest)',
                    pred_mean_sig',
                    ribbon = [2 * sqrt.(pred_cov_sig[1, 1, :]); 2 * sqrt.(pred_cov_sig[1, 1, :])]',
                    label = "Sigmoid",
                    color = clrs[3],
                )

                scatter!(x, y, markershape = :x, label = "", color = "black", markersize = 6)
                savefig(plt, joinpath(@__DIR__, "Fit_and_predict_1D_" * string(exp_range[exp_idx]) * ".pdf"))
            end
            priorL2err[exp_idx, :] += [
                sqrt(sum((ytest_nonoise - prior_mean) .^ 2)),
                sqrt(sum((ytest_nonoise - prior_mean_relu) .^ 2)),
                sqrt(sum((ytest_nonoise - prior_mean_sig) .^ 2)),
            ]
            priorweightedL2err[exp_idx, :] += [
                sqrt(sum(1 ./ (prior_cov .+ noise_sd^2) .* (ytest_nonoise - prior_mean) .^ 2)),
                sqrt(sum(1 ./ (prior_cov_relu .+ noise_sd^2) .* (ytest_nonoise - prior_mean_relu) .^ 2)),
                sqrt(sum(1 ./ (prior_cov_sig .+ noise_sd^2) .* (ytest_nonoise - prior_mean_sig) .^ 2)),
            ]
            L2err[exp_idx, :] += [
                sqrt(sum((ytest_nonoise - pred_mean) .^ 2)),
                sqrt(sum((ytest_nonoise - pred_mean_relu) .^ 2)),
                sqrt(sum((ytest_nonoise - pred_mean_sig) .^ 2)),
            ]
            weightedL2err[exp_idx, :] += [
                sqrt(sum(1 ./ (pred_cov .+ noise_sd^2) .* (ytest_nonoise - pred_mean) .^ 2)),
                sqrt(sum(1 ./ (pred_cov_relu .+ noise_sd^2) .* (ytest_nonoise - pred_mean_relu) .^ 2)),
                sqrt(sum(1 ./ (pred_cov_sig .+ noise_sd^2) .* (ytest_nonoise - pred_mean_sig) .^ 2)),
            ]
        end

        println("Prior for 1d->1d:")
        println("L2 errors: fourier, neuron, sigmoid")
        println(priorL2err)
        #println("weighted L2 errors: fourier, neuron, sigmoid")
        #println(priorweightedL2err)

        println("Posterior for 1d->1d, with increasing data:")
        println("L2 errors: fourier, neuron, sigmoid")
        println(L2err)
        @test all([all(L2err[i, :] .< L2err[i - 1, :]) for i in 2:size(L2err, 1)])
        ## This test is too brittle for small data
        #println("weighted L2 errors: fourier, neuron, sigmoid")
        #println(weightedL2err)
        #@test all([all(weightedL2err[i,:] .< weightedL2err[i-1,:]) for i=2:size(weightedL2err,1)])

    end # testset "Fit and predict"


    @testset "Fit and predict: d-D -> 1-D" begin

        rng = StableRNG(seed + 1)
        input_dim = 6
        n_features = 1500
        ftest_nd_to_1d(x::AbstractMatrix) =
            mapslices(column -> exp(-0.1 * norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)

        #problem formulation
        n_data = 1000
        x = rand(rng, MvNormal(zeros(input_dim), I), n_data)
        noise_sd = 1e-6
        lambda = noise_sd^2
        noise = rand(rng, Normal(0, noise_sd), (1, n_data))

        y = ftest_nd_to_1d(x) + noise
        io_pairs = PairedDataContainer(x, y)

        n_test = 500
        xtestvec = rand(rng, MvNormal(zeros(input_dim), I), n_test) # hard to get rare data far from truth

        xtest = DataContainer(xtestvec)
        ytest_nonoise = ftest_nd_to_1d(get_data(xtest))

        # specify features
        # note the σ_c and sigma values come from `examples/Learn_hyperparameters/nd_to_1d_regression_direct_withcov.jl`
        μ_c = 0.0
        σ_c = [
            0.7303238761547537,
            0.9609037430404104,
            1.281387598126217,
            2.1248671068339577,
            2.29258827903985,
            2.4442948886686304,
        ]
        pd = ParameterDistribution(
            Dict(
                "distribution" => VectorOfParameterized(map(sd -> Normal(μ_c, sd), σ_c)),
                "constraint" => repeat([no_constraint()], input_dim),
                "name" => "xi",
            ),
        )
        feature_sampler = FeatureSampler(pd, rng = copy(rng))
        sff = ScalarFourierFeature(n_features, feature_sampler)

        #second case with batching
        batch_sizes = Dict("train" => 500, "test" => 500, "feature" => 500)

        rfm_batch = RandomFeatureMethod(sff, batch_sizes = batch_sizes, regularization = lambda)

        fitted_batched_features = fit(rfm_batch, io_pairs)

        # test prediction with different features
        prior_mean, prior_cov = predict_prior(rfm_batch, xtest) # predict inputs from unfitted features

        # test other prior methods.
        prior_cov2, features_tmp = predict_prior_cov(rfm_batch, xtest)
        prior_mean2 = predict_prior_mean(rfm_batch, xtest, features_tmp)
        @test all(isapprox.(prior_cov, prior_cov2, atol = tol))
        @test all(isapprox.(prior_mean, prior_mean2, atol = tol))


        priorL2err = sqrt(sum((ytest_nonoise - prior_mean) .^ 2))
        priorweightedL2err = sqrt(sum(1 ./ (prior_cov .+ noise_sd^2) .* (ytest_nonoise - prior_mean) .^ 2))
        println("Prior for nd->1d")
        println("L2 error: ", priorL2err)
        #println("weighted L2 error: ", priorweightedL2err)

        pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)
        # test in-place calculations
        pmtmp = ones(size(pred_mean))
        pctmp = ones(size(pred_cov))
        buffer = zeros(size(pctmp, 3), size(pctmp, 1), n_features) # n x p x m
        predict!(rfm_batch, fitted_batched_features, xtest, pmtmp, pctmp, buffer)
        @test all(isapprox.(pred_mean, pmtmp, atol = tol))
        @test all(isapprox.(pred_cov, pctmp, atol = tol))

        L2err = sqrt(sum((ytest_nonoise - pred_mean) .^ 2))
        weightedL2err = sqrt(sum(1 ./ (pred_cov .+ noise_sd^2) .* (ytest_nonoise - pred_mean) .^ 2))
        println("Posterior for nd->1d")
        println("L2 error: ", L2err)
        #println("weighted L2 error: ", weightedL2err)

        @test L2err < priorL2err
        #@test weightedL2err < priorweightedL2err

        if TEST_PLOT_FLAG
            #plot slice through one dimensions, others fixed to 0
            xrange = collect(-3:0.01:3)
            xslice = zeros(input_dim, length(xrange))
            for direction in 1:input_dim
                xslicenew = copy(xslice)
                xslicenew[direction, :] = xrange

                yslice = ftest_nd_to_1d(xslicenew)

                pred_mean_slice, pred_cov_slice = predict(rfm_batch, fitted_batched_features, DataContainer(xslicenew))
                pred_cov_slice[1, 1, :] = max.(pred_cov_slice[1, 1, :], 0.0)
                plt = plot(
                    xrange,
                    yslice',
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )
                plot!(
                    xrange,
                    pred_mean_slice',
                    ribbon = [2 * sqrt.(pred_cov_slice[1, 1, :]); 2 * sqrt.(pred_cov_slice[1, 1, :])]',
                    label = "Fourier",
                    color = "blue",
                )
                savefig(
                    plt,
                    joinpath(@__DIR__, "Fit_and_predict_d-D_" * string(direction) * "of" * string(input_dim) * ".pdf"),
                )
            end
        end
    end

    @testset "Fit and predict: 1-D -> M-D" begin
        rng = StableRNG(seed + 2)
        input_dim = 1
        output_dim = 3
        n_features = 300
        function ftest_1d_to_3d(x::AbstractMatrix)
            out = zeros(3, size(x, 2))
            out[1, :] = mapslices(column -> sin(norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)
            out[2, :] = mapslices(column -> exp(-0.1 * norm([i * c for (i, c) in enumerate(column)])^2), x, dims = 1)
            out[3, :] = mapslices(
                column ->
                    norm([i * c for (i, c) in enumerate(column)]) * sin(1 / norm([i * c for (i, c) in enumerate(column)])^2) -
                    1,
                x,
                dims = 1,
            )
            return out
        end

        #utility
        function flat_to_chol(x::AbstractArray)
            choldim = Int(floor(sqrt(2 * length(x))))
            cholmat = zeros(choldim, choldim)
            for i in 1:choldim
                for j in 1:i
                    cholmat[i, j] = x[sum(0:(i - 1)) + j]
                end
            end
            return cholmat
        end
        #problem formulation
        n_data = 200
        x = rand(rng, MvNormal(zeros(input_dim), I), n_data)

        # run three sims, one with diagonal noise, one with multivariate using ID reg, one with multivariate using cov reg.
        # TODO make non-diagonal lambdamat stable for hyperparameter learning.
        exp_names = ["diagonal", "correlated-lambdaconst", "diagonal-lambdamat", "correlated-lambdamat"]
        cov_mats = [
            Diagonal((5e-2)^2 * ones(output_dim)),
            convert(
                Matrix,
                Tridiagonal((5e-3) * ones(output_dim - 1), (2e-2) * ones(output_dim), (5e-3) * ones(output_dim - 1)),
            ),
            Diagonal((5e-2)^2 * ones(output_dim)),
            convert(
                Matrix,
                Tridiagonal((5e-3) * ones(output_dim - 1), (2e-2) * ones(output_dim), (5e-3) * ones(output_dim - 1)),
            ),
        ]

        lambdas = [
            exp((1 / output_dim) * sum(log.(eigvals(cov_mats[1])))) * I, #det(C)^{1/m}*I
            exp((1 / output_dim) * sum(log.(eigvals(cov_mats[2])))) * I,
            cov_mats[3],
            cov_mats[4],
        ]

        # use learnt hyperparameters from nd_to_md_regression_direct_withcov.jl
        hps = [
            [
                0.4593643339085826,
                3.4402666569048286,
                0.7969152294959352,
                1.6851794464936298,
                3.4527613874722656,
                5.854154858415093,
                2.302845969371132,
            ],
            [
                0.5262518700390091,
                3.101856977561892,
                0.26825394655980433,
                0.7441825061473302,
                2.82685046470828,
                1.6584531227433983,
                1.7630307816260378,
            ],
            [
                1.0743245054960715,
                2.431492517819835,
                1.3719239685183025,
                2.1201669372745564,
                5.5003047613684934,
                4.331847045019546,
                2.2668423707079453,
            ],
            [
                0.3498665262324442,
                2.9456636865429298,
                0.22302714358146644,
                4.1288684250215555,
                0.43381960299341393,
                5.405698886196184,
                0.5746603230965016,
            ],
        ]
        for (cov_mat, lambda, hp, exp_name) in zip(cov_mats, lambdas, hps, exp_names)
            println(exp_name)
            U = ones(1, 1)
            cholV = flat_to_chol(hp[2:(Int(0.5 * output_dim * (output_dim + 1)) + 1)])
            V = hp[1] * (cholV * permutedims(cholV, (2, 1)) + hp[1] * I)

            noise_dist = MvNormal(zeros(output_dim), cov_mat)
            noise = rand(rng, noise_dist, n_data)

            y = ftest_1d_to_3d(x) + noise
            io_pairs = PairedDataContainer(x, y)

            n_test = 200
            #            xtestvec = rand(rng, MvNormal(zeros(input_dim), I), n_test)
            xtestvec = rand(rng, Uniform(-2.01, 2.01), (1, n_test))

            xtest = DataContainer(xtestvec)
            ytest_nonoise = ftest_1d_to_3d(get_data(xtest))

            # numbers from examples/hyperparameter_learning/nd_to_md_regression_direct_withcov.jl
            M = zeros(input_dim, output_dim)

            dist = MatrixNormal(M, U, V) #produces matrix samples
            pd = ParameterDistribution(
                Dict(
                    "distribution" => Parameterized(dist),
                    "constraint" => repeat([no_constraint()], input_dim * output_dim),
                    "name" => "xi",
                ),
            )
            feature_sampler = FeatureSampler(pd, output_dim, rng = copy(rng))
            vff = VectorFourierFeature(n_features, output_dim, feature_sampler)

            batch_sizes = Dict("train" => 500, "test" => 500, "feature" => 500)
            rfm_batch = RandomFeatureMethod(vff, batch_sizes = batch_sizes, regularization = lambda)
            fitted_batched_features = fit(rfm_batch, io_pairs)
            #quick test for the m>np case (changes the regularization)
            if exp_name == "diagonal-lambdamat"
                vff_tmp = VectorFourierFeature(n_test * output_dim + 1, output_dim, feature_sampler) #m > np
                rfm_tmp = RandomFeatureMethod(vff_tmp, regularization = lambda)
                #                @test_logs (:info,) fit(rfm_tmp, io_pairs) loop vec throws some warning - messing this test up
                fit_tmp = fit(rfm_tmp, io_pairs)
                @test fit_tmp.regularization ≈ inv(lambda) # exp(1.0 / output_dim * log(det(lambda))) * I
            end

            # test prediction L^2 error of mean
            prior_mean, prior_cov = predict_prior(rfm_batch, xtest) # predict inputs from unfitted features
            #  2 x n,  2 x 2 x n
            priorL2err = sqrt.(sum((ytest_nonoise - prior_mean) .^ 2))
            priorweightedL2err = [0.0]
            for i in 1:n_test
                diff = reshape(ytest_nonoise[:, i] - prior_mean[:, i], :, 1)
                priorweightedL2err .+= sum(permutedims(diff, (2, 1)) * inv(prior_cov[:, :, i] + cov_mat) * diff)
            end
            priorweightedL2err = sqrt.(priorweightedL2err)[:]
            println("Prior for 1d->3d")
            println("L2 error: ", priorL2err)
            #        println("weighted L2 error: ", priorweightedL2err)

            pred_mean, pred_cov = predict(rfm_batch, fitted_batched_features, xtest)
            # and other internal methods not called by predict

            pred_cov_tmp, features_tmp = predictive_cov(rfm_batch, fitted_batched_features, xtest)
            pred_mean_tmp = predictive_mean(rfm_batch, fitted_batched_features, xtest, features_tmp)

            @test all(isapprox.(pred_mean, pred_mean_tmp, atol = tol))
            @test all(isapprox.(pred_cov, pred_cov_tmp, atol = tol))

            if exp_name ∈ ["correlated-lambdaconst", "diagonal-lambdamat"]
                pmtmp = similar(pred_mean)
                pctmp = similar(pred_cov)
                buffer = zeros(size(pctmp, 3), size(pctmp, 1), n_features) # n x p x m
                predict!(rfm_batch, fitted_batched_features, xtest, pmtmp, pctmp, buffer)
                @test all(isapprox.(pred_mean, pmtmp, atol = tol))
                @test all(isapprox.(pred_cov, pctmp, atol = tol))
                # now create features in cov and pass to mean (check these methods)
                pmtmp2 = similar(pred_mean)
                pctmp2 = similar(pred_cov)
                features_tmp = predictive_cov!(rfm_batch, fitted_batched_features, xtest, pctmp2, buffer)
                predictive_mean!(rfm_batch, fitted_batched_features, xtest, pmtmp2, features_tmp)
                @test all(isapprox.(pred_mean, pmtmp2, atol = tol))
                @test all(isapprox.(pred_cov, pctmp2, atol = tol))
            end

            # no threading
            if exp_name ∈ ["correlated-lambdamat", "diagonal-lambdamat"]
                rfm_nothread = RandomFeatureMethod(vff, regularization = lambda, tullio_threading = false)
                fitted_features_nothread = fit(rfm_nothread, io_pairs)
                coeffs = get_coeffs(fitted_batched_features)
                coeffs_nothread = get_coeffs(fitted_features_nothread)
                @test coeffs ≈ coeffs_nothread
                # it appears there is a significant difference in the linear algebra with and without threads here.
                tol_tmp = 1e-10

                prior_mean_nothread, prior_cov_nothread = predict_prior(rfm_nothread, xtest, tullio_threading = false) # predict inputs from unfitted features
                @test all(isapprox.(prior_mean_nothread, prior_mean, atol = tol_tmp))
                @test all(isapprox.(prior_cov_nothread, prior_cov, atol = tol_tmp))

                pred_mean_nothread, pred_cov_nothread =
                    predict(rfm_nothread, fitted_features_nothread, xtest, tullio_threading = false)
                @test all(isapprox.(pred_mean_nothread, pred_mean, atol = tol_tmp))
                @test all(isapprox.(pred_cov_nothread, pred_cov, atol = tol_tmp))
                pmr_nothread = similar(pred_mean)
                pcr_nothread = similar(pred_cov)
                buffer = zeros(size(pred_cov, 3), size(pred_cov, 1), n_features) # n x p x m
                predict!(
                    rfm_nothread,
                    fitted_features_nothread,
                    xtest,
                    pmr_nothread,
                    pcr_nothread,
                    buffer,
                    tullio_threading = false,
                )
                @test all(isapprox.(pmr_nothread, pred_mean, atol = tol_tmp))
                @test all(isapprox.(pcr_nothread, pred_cov, atol = tol_tmp))
            end

            #println(pred_mean)
            L2err = sqrt.(sum((ytest_nonoise - pred_mean) .^ 2))
            weightedL2err = [0.0]
            #for i in 1:n_test
            #    diff = reshape(ytest_nonoise[:, i] - pred_mean[:, i], :, 1)
            #    weightedL2err .+= sum(permutedims(diff, (2, 1)) * inv(pred_cov[:, :, i] + cov_mat) * diff)
            #end
            #weightedL2err = sqrt.(weightedL2err)[:]
            println("Posterior for 1d->3d")
            println("L2 error: ", L2err)
            #println("weighted L2 error: ", weightedL2err)

            @test L2err < priorL2err
            #@test weightedL2err < priorweightedL2err


            if TEST_PLOT_FLAG
                # learning on Normal(0,1) dist, forecast on (-2.01,2.01)
                xrange = reshape(collect(-2.01:0.02:2.01), 1, :)

                yrange = ftest_1d_to_3d(xrange)

                pred_mean_slice, pred_cov_slice = predict(rfm_batch, fitted_batched_features, DataContainer(xrange))

                for i in 1:output_dim
                    pred_cov_slice[i, i, :] = max.(pred_cov_slice[i, i, :], 0.0)
                end
                #plot diagonal
                xplot = xrange[:]
                plt = plot(
                    xplot,
                    yrange[1, :],
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )
                plot!(
                    xplot,
                    yrange[2, :],
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )
                plot!(
                    xplot,
                    yrange[3, :],
                    show = false,
                    color = "black",
                    linewidth = 5,
                    size = (600, 600),
                    legend = :topleft,
                    label = "Target",
                )
                scatter!(x[:], y[1, :], color = "blue", label = "", marker = :x)

                plot!(
                    xplot,
                    pred_mean_slice[1, :],
                    ribbon = [2 * sqrt.(pred_cov_slice[1, 1, :]); 2 * sqrt.(pred_cov_slice[1, 1, :])],
                    label = "Fourier",
                    color = "blue",
                )
                scatter!(x[:], y[2, :], color = "red", label = "", marker = :x)
                plot!(
                    xplot,
                    pred_mean_slice[2, :],
                    ribbon = [2 * sqrt.(pred_cov_slice[2, 2, :]); 2 * sqrt.(pred_cov_slice[2, 2, :])],
                    label = "Fourier",
                    color = "red",
                )
                scatter!(x[:], y[3, :], color = "green", label = "", marker = :x)
                plot!(
                    xplot,
                    pred_mean_slice[3, :],
                    ribbon = [2 * sqrt.(pred_cov_slice[3, 3, :]); 2 * sqrt.(pred_cov_slice[3, 3, :])],
                    label = "Fourier",
                    color = "green",
                )

                savefig(plt, joinpath(@__DIR__, "Fit_and_predict_1D_to_3D_" * exp_name * ".pdf"))
            end
        end
    end

end
