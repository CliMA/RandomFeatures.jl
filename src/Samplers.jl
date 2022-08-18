module Samplers

import
    StatsBase: sample

using
    Random,
    Distributions,
    EnsembleKalmanProcesses.ParameterDistributions

export
    Sampler,
    FeatureSampler,
    HyperSampler,
    get_parameter_distribution,
    get_optimizable_parameters,
    get_uniform_shift_bounds,
    get_rng,
    sample

struct Sampler
    parameter_distribution::ParameterDistribution
    optimizable_parameters::Union{AbstractVector,Nothing}
    uniform_shift_bounds::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

function FeatureSampler(
    parameter_distribution::ParameterDistribution;
    optimizable_parameters::Union{AbstractVector,Nothing}=nothing,
    uniform_shift_bounds::Union{AbstractVector,Nothing}=[0,2*pi],
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    
    # adds a uniform distribution to the parameter distribution
    if isa(uniform_shift_bounds,AbstractVector)
        if length(uniform_shift_bounds) == 2
            unif_pd = ParameterDistribution(
                Dict(
                    "distribution" => Parameterized(Uniform(uniform_shift_bounds[1],uniform_shift_bounds[2])),
                    "constraint" => no_constraint(),
                    "name" => "uniform",
                )
            )
            pd = combine_distributions([parameter_distribution, unif_pd])
        end
    else
        pd = parameter_distribution
        uniform_shift_bounds=nothing
    end
    
    return Sampler(
        pd,
        optimizable_parameters,
        uniform_shift_bounds,
        rng,
    )
    
end

function HyperSampler(
    parameter_distribution::ParameterDistribution;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    return Sampler(
        parameter_distribution,
        nothing,
        nothing,
        rng,
    )
    
end

get_parameter_distribution(s::Sampler) = s.parameter_distribution
get_optimizable_parameters(s::Sampler) = s.optimizable_parameters
get_uniform_shift_bounds(s::Sampler) = s.uniform_shift_bounds
get_rng(s::Sampler) = s.rng

# methods - calls to ParameterDistribution methods
#=
sample(rng::AbstractRNG, s::Sampler, n_draws::Int) = sample(rng, s.parameter_distribution, n_draws)
=#

function sample(rng::AbstractRNG, s::Sampler, n_draws::Int)
    pd = get_parameter_distribution(s)
    samp = sample(rng, pd, n_draws)

    #now create a Samples-type distribution from the samples
    s_names = get_name(pd)
    s_slices = batch(pd) # e.g., [1, 2:3, 4:9]
    s_constraints = get_all_constraints(pd)
    s_samples = [Samples(samp[slice,:]) for slice in s_slices]

    return combine_distributions([
        ParameterDistribution(ss, sc, sn) for (ss, sc, sn) in zip(s_samples, s_constraints, s_names)
    ])
end


sample(s::Sampler, n_draws::Int) = sample(s.rng, s, n_draws)
sample(rng::AbstractRNG, s::Sampler) = sample(rng, s, 1)
sample(s::Sampler) = sample(s.rng, s, 1)

# function set_optimized_parameters(s::Sampler, optimized_parameters::NamedTuple) 
#     
#     #setting replacing distribution parameters
# end


end # module
