module RandomFeatureMethod

# container for SVD/QR/Chol etc
using LinearAlgebra

struct RandomFeatureMethod
    rf::RandomFeature
    feature_factors::Union{AbstractMatrix,LinearAlgebra.Factorization}
    feature_coeffs::AbstractVector
    batch_sizes::Dict
end 


sample_feature_distribution(rfm::RandomFeatureMethod) = sample_feature_distribution(rfm.rf)

function fit!(rfm::RandomFeatureMethod, input_data, output_data)

end

function predict(rfm::RandomFeatureMethod, new_input_data)
    pred_mean = predict_mean(rfm, new_input_data)
    pred_cov = predict_cov(rfm, new_input_data)
    return pred_mean, pred_cov
end

function predict_mean(rfm::RandomFeatureMethod, new_input_data)

end

function predict_cov(rfm::RandomFeatureMethod, new_input_data)

end

function posterior_cov(rfm::RandomFeatureMethod, u_input, v_input)

end

function get_optimizable_hyperparameters(rfm::RandomFeatureMethod)

end

function set_optimized_hyperparameters(rfm::RandomFeatureMethod, optimized_hyperparameters)

end

function evaluate_hyperparameter_cost(rfm::RandomFeatureMethod, input_data, output_data)

end



end # module
