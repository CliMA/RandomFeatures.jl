module Features

include("ActivationFunctions.jl")

abstract type RandomFeature end


function sample_feature_distribution(rf::RandomFeature)

end

function get_optimizable_hyperparameters(rf::RandomFeature)

end

function set_optimized_hyperparameters(rf::RandomFeature, optimized_hyperparameters)

end


struct FourierFeature <: RandomFeature
    sampler::DistributionSampler
    current_sample::Dict
end

function build_features(rf::FourierFeature)

end

struct NeuronFeature <: RandomFeature
    sampler::DistributionSampler
    activation::ScalarActivation
    current_sample::Dict
end

function build_features(rf::NeuronFeature)

end
 
end #module
