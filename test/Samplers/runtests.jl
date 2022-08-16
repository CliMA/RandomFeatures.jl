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
    pd = constrained_gaussian("test", μ_c, σ_c, -Inf, Inf)

    # test internals
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
    # first with global rng
    Random.seed!(seed)
    sample1 = sample(sampler)
    Random.seed!(seed)
    @test sample1 == sample(full_pd)

    n_samples = 40
    Random.seed!(seed)
    sample2 = sample(sampler, n_samples)
    Random.seed!(seed)
    @test sample2 == sample(full_pd, n_samples)


    # now with two explicit rng's
    rng1 = Random.MersenneTwister(seed)
   
    @test sample(copy(rng1),sampler) == sample(copy(rng1),full_pd,1)
    @test sample(copy(rng1),sampler,n_samples) == sample(copy(rng1),full_pd,n_samples)

    rng2 = StableRNG(seed)
    @test sample(copy(rng2),sampler) == sample(copy(rng2),full_pd,1)
    @test sample(copy(rng2),sampler,n_samples) == sample(copy(rng2),full_pd,n_samples)
    
    
end
