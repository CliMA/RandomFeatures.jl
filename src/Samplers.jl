module Samplers

import
    StatsBase: sample

using
    Random,
    Distributions,
    EnsembleKalmanProcesses.ParameterDistributions

export
    Sampler,
    get_parameter_distribution,
    get_optimizable_parameters,
    get_uniform_shift_bounds,
    sample

struct Sampler
    parameter_distribution::ParameterDistribution
    optimizable_parameters::Union{AbstractVector,Nothing}
    uniform_shift_bounds::Union{AbstractVector,Nothing}
end

function Sampler(
    parameter_distribution::ParameterDistribution;
    optimizable_parameters::Union{AbstractVector,Nothing}=nothing,
    uniform_shift_bounds::Union{AbstractVector,Nothing}=[0,2*pi],
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
            pd = combine_distributions([parameter_distribution,unif_pd])
        end
    else
        pd = parameter_distribution
        uniform_shift_bounds=nothing
    end
    
    return Sampler(pd,optimizable_parameters,uniform_shift_bounds)
    
end

get_parameter_distribution(s::Sampler) = s.parameter_distribution
get_optimizable_parameters(s::Sampler) = s.optimizable_parameters
get_uniform_shift_bounds(s::Sampler) = s.uniform_shift_bounds

# methods - calls to ParameterDistribution methods
sample(rng::AbstractRNG, s::Sampler, n_draws::Int) = sample(rng, s.parameter_distribution, n_draws)
sample(s::Sampler, n_draws::Int) = sample(Random.GLOBAL_RNG, s, n_draws)
sample(rng::AbstractRNG, s::Sampler) = sample(rng, s, 1)
sample(s::Sampler) = sample(Random.GLOBAL_RNG, s, 1)

# function set_optimized_parameters(s::Sampler, optimized_parameters::NamedTuple) 
#     
#     #setting replacing distribution parameters
# end


end # module
