using Test
using StableRNGs
using StatsBase
using LinearAlgebra
using Random
using Distributions
using Tullio

using RandomFeatures.ParameterDistributions
using RandomFeatures.Samplers
using RandomFeatures.Features

seed = 2202

@testset "Features" begin
    @testset "ScalarFunctions" begin

        af_list = [
            Relu(),
            Lrelu(),
            Gelu(),
            Elu(),
            Selu(),
            Heaviside(),
            SmoothHeaviside(),
            Sawtooth(),
            Softplus(),
            Tansig(),
            Sigmoid(),
        ]
        # very rough tests that these are activation functions
        for af in af_list
            @test isa(af, ScalarActivation)

            x_test_neg = collect(-1:0.1:-0.1)
            x_test_pos = collect(0:0.1:1)
            println("Testing ", af)
            @test all(apply_scalar_function(af, x_test_neg) .<= log(2)) # small for negative x

            if !isa(af, Sawtooth)
                @test all(
                    apply_scalar_function(af, x_test_pos[2:end]) -
                    apply_scalar_function(af, x_test_pos[1:(end - 1)]) .>= 0,
                ) # monotone increasing for positive x
            else
                x_test_0_0pt5 = collect(0:0.1:0.5)
                x_test_0pt5_1 = collect(0.5:0.1:1)
                @test all(
                    apply_scalar_function(af, x_test_0_0pt5[2:end]) -
                    apply_scalar_function(af, x_test_0_0pt5[1:(end - 1)]) .>= 0,
                )
                @test all(
                    apply_scalar_function(af, x_test_0pt5_1[2:end]) -
                    apply_scalar_function(af, x_test_0pt5_1[1:(end - 1)]) .<= 0,
                )
            end
        end

        # others
        sf = Features.Cosine() # as Distributions also has a Cosine()
        println("Testing ", sf)
        @test isa(sf, ScalarFunction)

        x_test = collect(-1:0.1:1)
        @test all(abs.(apply_scalar_function(sf, x_test) - cos.(x_test)) .< 2 * eps())

    end

    @testset "Scalar: Constructors" begin

        n_features = 20
        relu = Relu()
        rng = StableRNG(seed)

        #setup sampler xi distributions
        μ_c = 0.0
        σ_c = 2.0
        pd_err = constrained_gaussian("test", μ_c, σ_c, -Inf, Inf)
        feature_sampler_err = FeatureSampler(pd_err, rng = copy(rng))
        pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
        feature_sampler = FeatureSampler(pd, rng = copy(rng))

        # postive constraints for sigma
        sigma_fixed_err = Dict("not sigma" => 10.0)

        # Error checks
        @test_throws ArgumentError ScalarFeature(
            n_features,
            feature_sampler_err, # causes error
            relu,
        )
        @test_logs (
            :info,
            " Required feature parameter key \"sigma\" not defined, continuing with default value \"sigma\" = 1 ",
        ) ScalarFeature(n_features, feature_sampler, relu, feature_parameters = sigma_fixed_err)

        # ScalarFeature and getters
        feature_sampler = FeatureSampler(pd, rng = copy(rng)) # to reset the rng
        sf_test = ScalarFeature(n_features, feature_sampler, relu)
        @test get_n_features(sf_test) == n_features
        @test get_feature_parameters(sf_test) == Dict("sigma" => 1.0)
        @test get_output_dim(sf_test) == 1
        @test get_feature_parameters(sf_test)["sigma"] == sqrt(1)

        test_sample = sample(copy(rng), feature_sampler, n_features)
        sf_test_sample = get_feature_sample(sf_test)
        @test get_distribution(sf_test_sample)["xi"] == get_distribution(test_sample)["xi"]
        @test get_distribution(sf_test_sample)["bias"] == get_distribution(test_sample)["bias"]
        @test get_all_constraints(sf_test_sample) == get_all_constraints(test_sample)
        @test get_name(sf_test_sample) == get_name(test_sample)
        sf_test_sampler = get_feature_sampler(sf_test)


        sff_test = ScalarFourierFeature(n_features, feature_sampler)

        snf_test = ScalarNeuronFeature(n_features, feature_sampler)

        @test isa(get_scalar_function(sff_test), Features.Cosine)
        @test get_feature_parameters(sff_test)["sigma"] == sqrt(2.0)
        @test isa(get_scalar_function(snf_test), Relu)

    end

    @testset "Scalar: build features" begin

        n_features = 20
        rng = StableRNG(seed)

        μ_c = 0.0
        σ_c = 2.0
        pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
        feature_sampler_1d = FeatureSampler(pd, rng = copy(rng))

        sigma_value = 10.0
        sigma_fixed = Dict("sigma" => sigma_value)

        sff_1d_test = ScalarFourierFeature(n_features, feature_sampler_1d, feature_parameters = sigma_fixed)

        # 1D input space -> 1D output space
        inputs_1d = reshape(collect(-1:0.01:1), (1, length(collect(-1:0.01:1))))
        n_samples_1d = length(inputs_1d)
        features_1d = build_features(sff_1d_test, inputs_1d)

        rng1 = copy(rng)
        samp_xi = reshape(sample(rng1, pd, n_features), (1, n_features))
        samp_unif = reshape(rand(rng1, Uniform(0, 2 * pi), n_features), (1, n_features))
        inputs_1d_T = permutedims(inputs_1d, (2, 1))
        rf_test = sigma_value * cos.(inputs_1d_T * samp_xi .+ samp_unif)
        @test size(features_1d) == (n_samples_1d, 1, n_features) # we store internally with output_dim = 1
        @test all(abs.(rf_test - features_1d[:, 1, :]) .< 10 * eps()) # sufficiently big to deal with inaccuracy of cosine

        # 10D input space -> 1D output space
        # generate a bunch of random samples as data points
        n_samples = 200
        inputs_10d = rand(MvNormal(zeros(10), convert(Matrix, SymTridiagonal(2 * ones(10), 0.5 * ones(9)))), n_samples) # 10 x n_samples
        # 10D indep gaussians on input space as feature distribution
        pd_10d = ParameterDistribution(
            Dict(
                "distribution" => VectorOfParameterized(repeat([Normal(μ_c, σ_c)], 10)),
                "constraint" => repeat([no_constraint()], 10),
                "name" => "xi",
            ),
        )
        feature_sampler_10d = FeatureSampler(pd_10d, rng = copy(rng))

        sff_10d_test = ScalarNeuronFeature(n_features, feature_sampler_10d, feature_parameters = sigma_fixed)

        features_10d = build_features(sff_10d_test, inputs_10d)

        rng2 = copy(rng)
        samp_xi = reshape(sample(rng2, pd_10d, n_features), (10, n_features))
        samp_unif = reshape(rand(rng2, Uniform(0, 2 * pi), n_features), (1, n_features))
        inputs_10d_T = permutedims(inputs_10d, (2, 1))
        rf_test2 = sigma_value * max.(inputs_10d_T * samp_xi .+ samp_unif, 0)
        @test size(features_10d) == (n_samples, 1, n_features) # we store internall with output_dim = 1
        @test all(abs.(rf_test2 - features_10d[:, 1, :]) .< 1e3 * eps()) # sufficiently big to deal with inaccuracy of relu

    end


    @testset "Vector: Constructors" begin

        n_features = 20
        input_dim = 5
        output_dim = 2
        relu = Relu()
        rng = StableRNG(seed)


        #just to test error flag
        μ_c = 0.0
        σ_c = 2.0
        pd_err = constrained_gaussian("test", μ_c, σ_c, -Inf, Inf, repeats = output_dim)
        feature_sampler_err = FeatureSampler(pd_err, output_dim, rng = copy(rng))

        #setup sampler xi distributions:
        dist = MatrixNormal(zeros(input_dim, output_dim), Diagonal(ones(input_dim)), Diagonal(ones(output_dim))) #produces 5 x 2 matrix samples
        pd = ParameterDistribution(
            Dict(
                "distribution" => Parameterized(dist),
                "constraint" => repeat([no_constraint()], input_dim * output_dim), #flattened
                "name" => "xi",
            ),
        )
        feature_sampler = FeatureSampler(pd, output_dim, rng = copy(rng))

        # postive constraints for sigma
        sigma_fixed_err = Dict("not sigma" => 10.0)

        # Error checks
        @test_throws ArgumentError VectorFeature(
            n_features,
            output_dim,
            feature_sampler_err, # causes error
            relu,
        )
        @test_logs (
            :info,
            " Required feature parameter key \"sigma\" not defined, continuing with default value \"sigma\" = 1 ",
        ) VectorFeature(n_features, output_dim, feature_sampler, relu, feature_parameters = sigma_fixed_err)

        # VectorFeature and getters
        feature_sampler = FeatureSampler(pd, output_dim, rng = copy(rng)) # to reset the rng
        vf_test = VectorFeature(n_features, output_dim, feature_sampler, relu)
        @test get_n_features(vf_test) == n_features
        @test get_feature_parameters(vf_test) == Dict("sigma" => 1.0)
        @test get_output_dim(vf_test) == output_dim
        @test get_feature_parameters(vf_test)["sigma"] == sqrt(1)

        test_sample = sample(copy(rng), feature_sampler, n_features)
        vf_test_sample = get_feature_sample(vf_test)
        @test get_distribution(vf_test_sample)["xi"] == get_distribution(test_sample)["xi"]
        @test get_distribution(vf_test_sample)["bias"] == get_distribution(test_sample)["bias"]
        @test get_all_constraints(vf_test_sample) == get_all_constraints(test_sample)
        @test get_name(vf_test_sample) == get_name(test_sample)
        vf_test_sampler = get_feature_sampler(vf_test)

        vff_test = VectorFourierFeature(n_features, output_dim, feature_sampler)
        vnf_test = VectorNeuronFeature(n_features, output_dim, feature_sampler)

        @test isa(get_scalar_function(vff_test), Features.Cosine)
        @test get_feature_parameters(vff_test)["sigma"] == sqrt(2.0)
        @test isa(get_scalar_function(vnf_test), Relu)

    end

    @testset "Vector: build features" begin

        n_features = 20
        input_dim = 5
        output_dim = 2
        rng = StableRNG(seed)

        #setup sampler xi distributions:
        dist = MatrixNormal(zeros(input_dim, output_dim), Diagonal(ones(input_dim)), Diagonal(ones(output_dim))) #produces 5 x 2 matrix samples
        pd = ParameterDistribution(
            Dict(
                "distribution" => Parameterized(dist),
                "constraint" => repeat([no_constraint()], input_dim * output_dim),
                "name" => "xi",
            ),
        )
        feature_sampler_5d = FeatureSampler(pd, output_dim, rng = copy(rng))

        sigma_value = 10.0
        sigma_fixed = Dict("sigma" => sigma_value)

        vff_5d_2d_test =
            VectorFourierFeature(n_features, output_dim, feature_sampler_5d, feature_parameters = sigma_fixed)

        # and a flat one
        dist = MvNormal(zeros(input_dim * output_dim), Diagonal(ones(input_dim * output_dim))) #produces 10-length vector samples
        pd = ParameterDistribution(
            Dict(
                "distribution" => Parameterized(dist),
                "constraint" => repeat([no_constraint()], input_dim * output_dim),
                "name" => "xi",
            ),
        )
        feature_sampler_10dflat = FeatureSampler(pd, output_dim, rng = copy(rng))

        vff_10dflat_test =
            VectorFourierFeature(n_features, output_dim, feature_sampler_10dflat, feature_parameters = sigma_fixed)


        # 5D input space -> 2D output space
        n_samples = 200
        inputs_5d_2d = rand(Uniform(-1, 1), (input_dim, n_samples))
        features_5d_2d = build_features(vff_5d_2d_test, inputs_5d_2d)

        rng1 = copy(rng)
        samp_flat = sample(rng1, feature_sampler_5d, n_features)
        samp_xi_flat = get_distribution(samp_flat)["xi"]
        # as we flatten the samples currently in the sampler.sample. reshape with dist.
        samp_xi = reshape(samp_xi_flat, (input_dim, output_dim, size(samp_xi_flat, 2))) # in x out x n_feature_batch

        @tullio features[n, p, b] := inputs_5d_2d[d, n] * samp_xi[d, p, b]
        samp_bias = get_distribution(samp_flat)["bias"]
        @tullio features[n, p, b] += samp_bias[p, b]

        rf_test = sigma_value * cos.(features)
        @test size(features_5d_2d) == (n_samples, output_dim, n_features) # we store internally with output_dim = 1
        @test all(abs.(rf_test - features_5d_2d) .< 1e3 * eps()) # sufficiently big to deal with inaccuracy of cosine

        features_10dflat = build_features(vff_10dflat_test, inputs_5d_2d)

        rng1 = copy(rng)
        samp_flat = sample(rng1, feature_sampler_10dflat, n_features)
        samp_xi_flat = get_distribution(samp_flat)["xi"]
        # as we flatten the samples currently in the sampler.sample. reshape with dist.
        samp_xi = reshape(samp_xi_flat, (input_dim, output_dim, size(samp_xi_flat, 2))) # in x out x n_feature_batch

        @tullio features[n, p, b] := inputs_5d_2d[d, n] * samp_xi[d, p, b]
        samp_bias = get_distribution(samp_flat)["bias"]
        @tullio features[n, p, b] += samp_bias[p, b]

        rf_test = sigma_value * cos.(features)
        #        rf_test = sigma_value * cos.(inputs_5d_2d_T * samp_xi .+ samp_unif)
        @test size(features_10dflat) == (n_samples, output_dim, n_features) # we store internally with output_dim = 1
        @test all(abs.(rf_test - features_10dflat) .< 1e3 * eps()) # sufficiently big to deal with inaccuracy of cosine



    end


end
