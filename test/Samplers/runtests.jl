using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random
using EnsembleKalmanProcesses.ParameterDistributions

using RandomFeatures.Samplers

seed = 2022


@testset "Samplers" begin

    # create a Gaussian(0,4) distribution with EKP's ParameterDistribution constructors
    μ_c = 0.0
    σ_c = 4.0
    pd_err = constrained_gaussian("test", μ_c, σ_c, -Inf, Inf)

    # test internals
    @test_throws ArgumentError Sampler(pd_err)

    pd = constrained_gaussian("xi", μ_c, σ_c, -Inf, Inf)
    sampler = Sampler(pd)
    @test get_optimizable_parameters(sampler) == nothing
    @test get_uniform_shift_bounds(sampler) == [0,2*pi] 
    unif_pd = ParameterDistribution(
        Dict(
            "distribution" => Parameterized(Uniform(0,2*π)),
            "constraint" => no_constraint(),
            "name" => "uniform",
        )
    )
    full_pd = combine_distributions([pd,unif_pd])

    test_pd = get_parameter_distribution(sampler)
    @test get_distribution(test_pd) == get_distribution(full_pd)
    @test get_all_constraints(test_pd) == get_all_constraints(full_pd)
    @test get_name(test_pd) == get_name(full_pd)
    
    # and other option for settings
    usb = nothing
    oh = [:μ] 
    sampler2 = Sampler(
        pd,
        optimizable_parameters=oh,
        uniform_shift_bounds=usb
    )

    @test get_uniform_shift_bounds(sampler2) == usb
    @test get_optimizable_parameters(sampler2) == oh

    # test method: sample
    function sample_to_Sample(pd::ParameterDistribution, samp::AbstractMatrix)
        #now create a Samples-type distribution from the samples
        s_names = get_name(pd)
        s_slices = batch(pd) # e.g., [1, 2:3, 4:9]
        s_constraints = get_all_constraints(pd)
        s_samples = [Samples(samp[slice,:]) for slice in s_slices]
        
        return combine_distributions([
            ParameterDistribution(ss, sc, sn) for (ss, sc, sn) in zip(s_samples, s_constraints, s_names)
        ])
    end

    # first with global rng
    Random.seed!(seed)
    sample1 = sample(sampler) # produces a Samples ParameterDistribution
    Random.seed!(seed)
    test1 = sample_to_Sample(full_pd, sample(full_pd)) 
    @test get_distribution(sample1) == get_distribution(test1)
    @test get_all_constraints(sample1) == get_all_constraints(test1)
    @test get_name(sample1) == get_name(test1)

    n_samples = 40
    Random.seed!(seed)
    sample2 = sample(sampler, n_samples)
    Random.seed!(seed)
    test2 = sample_to_Sample(full_pd, sample(full_pd, n_samples))
    @test get_distribution(sample2) == get_distribution(test2)
    @test get_all_constraints(sample2) == get_all_constraints(test2)
    @test get_name(sample2) == get_name(test2)

    # now with two explicit rng's
    rng1 = Random.MersenneTwister(seed)
    sampler_rng1 = Sampler(pd, rng=copy(rng1))
    sample3 = sample(sampler_rng1)
    test3 = sample_to_Sample(full_pd, sample(copy(rng1), full_pd, 1))
    @test get_distribution(sample3) == get_distribution(test3)
    @test get_all_constraints(sample3) == get_all_constraints(test3)
    @test get_name(sample3) == get_name(test3)

    
    sampler_rng1 = Sampler(pd, rng=copy(rng1))
    sample4 = sample(sampler_rng1, n_samples)
    test4 = sample_to_Sample(full_pd, sample(copy(rng1), full_pd, n_samples))
    @test get_distribution(sample4) == get_distribution(test4)
    @test get_all_constraints(sample4) == get_all_constraints(test4)
    @test get_name(sample4) == get_name(test4)                             

    #this time override use rng2 to override the default Random.GLOBAL_RNG at the point of sampling
    rng2 = StableRNG(seed)
    sample5 = sample(copy(rng2), sampler)
    test5 = sample_to_Sample(full_pd, sample(copy(rng2), full_pd, 1))
    @test get_distribution(sample5) == get_distribution(test5)
    @test get_all_constraints(sample5) == get_all_constraints(test5)
    @test get_name(sample5) == get_name(test5)

    
    sample6 = sample(copy(rng2), sampler, n_samples)
    test6 = sample_to_Sample(full_pd, sample(copy(rng2), full_pd, n_samples))
    @test get_distribution(sample6) == get_distribution(test6)
    @test get_all_constraints(sample6) == get_all_constraints(test6)
    @test get_name(sample6) == get_name(test6)                             
    
    
end
