module Samplers

using EnsembleKalmanProcesses.ParameterDistributions

struct DistributionSampler
    parameter_distribution::ParameterDistribution
    optimizable_hyperparameters::AbstractVector
end

sample(ds::DistributionSampler) = sample(ds.parameter_distribution)

get_optimizable_hyperparameters(ds::DistributionSampler) = ds.optimizable_hyperparameters
    
function set_optimized_hyperparameters(ds::DistributionSampler, optimized_hyperparameters::NamedTuple) 
    #setting replacing distriution hyperparameters
end


end # module
