using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random

using RandomFeatures.ParameterDistributions
using RandomFeatures.Samplers

seed = 2022


@testset "Samplers" begin

    # create a Gaussian(0,4) distribution with EKP's ParameterDistribution constructors
    μ_c = -4.0
    σ_c = 1.0

    pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, 0.0)

    #1d output space
    fsampler = FeatureSampler(pd) # takes a uniform with shift as the bias
    unif_pd = ParameterDistribution(
        Dict("distribution" => Parameterized(Uniform(0, 2 * π)), "constraint" => no_constraint(), "name" => "bias"),
    )
    fsamplerbias = FeatureSampler(pd, unif_pd)#provide bias distribution explicitly

    full_pd = combine_distributions([pd, unif_pd])

    @test get_parameter_distribution(fsampler) == full_pd
    @test get_parameter_distribution(fsamplerbias) == full_pd

    #5d input - 3d output-space
    output_dim = 3
    pd_3d = constrained_gaussian("xi", μ_c, σ_c, -Inf, 0.0, repeats = 5)
    fsampler_3d = FeatureSampler(pd_3d, output_dim) # 3d output space
    unif_pd_3d = ParameterDistribution(
        Dict(
            "distribution" => VectorOfParameterized(repeat([Uniform(0, 2 * π)], output_dim)),
            "constraint" => repeat([no_constraint()], output_dim),
            "name" => "bias",
        ),
    )
    fsamplerbias_3d = FeatureSampler(pd_3d, unif_pd_3d)#provide bias distribution explicitly
    full_pd_3d = combine_distributions([pd_3d, unif_pd_3d])

    @test get_parameter_distribution(fsampler_3d) == full_pd_3d
    @test get_parameter_distribution(fsamplerbias_3d) == full_pd_3d


    # and other option for settings
    no_bias = nothing
    sampler_no_bias = FeatureSampler(pd, no_bias)
    @test pd == get_parameter_distribution(sampler_no_bias)



    # test method: sample
    function sample_to_Sample(pd::ParameterDistribution, samp::AbstractMatrix)
        constrained_samp = transform_unconstrained_to_constrained(pd, samp)
        #now create a Samples-type distribution from the samples
        s_names = get_name(pd)
        s_slices = batch(pd) # e.g., [1, 2:3, 4:9]
        s_samples = [Samples(constrained_samp[slice, :]) for slice in s_slices]
        s_constraints = [repeat([no_constraint()], size(slice, 1)) for slice in s_slices]

        return combine_distributions([
            ParameterDistribution(ss, sc, sn) for (ss, sc, sn) in zip(s_samples, s_constraints, s_names)
        ])
    end

    # first with global rng
    Random.seed!(seed)
    sample1 = sample(fsampler) # produces a Samples ParameterDistribution
    Random.seed!(seed)
    @test sample1 == sample_to_Sample(full_pd, sample(full_pd))

    n_samples = 40
    Random.seed!(seed)
    sample2 = sample(fsampler, n_samples)
    Random.seed!(seed)
    @test sample2 == sample_to_Sample(full_pd, sample(full_pd, n_samples))
    # now with two explicit rng's
    rng1 = Random.MersenneTwister(seed)
    sampler_rng1 = FeatureSampler(pd, rng = copy(rng1))
    @test get_rng(sampler_rng1) == copy(rng1)
    sample3 = sample(sampler_rng1)
    @test !(get_rng(sampler_rng1) == copy(rng1))
    @test sample3 == sample_to_Sample(full_pd, sample(copy(rng1), full_pd, 1))

    sampler_rng1 = FeatureSampler(pd, rng = copy(rng1))
    sample4 = sample(sampler_rng1, n_samples)
    @test sample4 == sample_to_Sample(full_pd, sample(copy(rng1), full_pd, n_samples))

    #this time override use rng2 to override the default Random.GLOBAL_RNG at the point of sampling
    rng2 = StableRNG(seed)
    sample5 = sample(copy(rng2), fsampler)
    sample5 == sample_to_Sample(full_pd, sample(copy(rng2), full_pd, 1))

    sample6 = sample(copy(rng2), fsampler, n_samples)
    @test sample6 == sample_to_Sample(full_pd, sample(copy(rng2), full_pd, n_samples))


end
