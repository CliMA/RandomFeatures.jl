# Features

```@meta
CurrentModule = RandomFeatures.Features
```

```@docs
ScalarFeature
ScalarFourierFeature
ScalarNeuronFeature
get_n_features
get_scalar_function
get_feature_sampler
get_feature_sample
get_feature_parameters
sample(rf::RandomFeature)
```
# [Scalar Functions](@id scalar-functions)

```@docs
ScalarFunction
ScalarActivation
apply_scalar_function
```

```@docs
    Cosine
    Relu
    Lrelu
    Gelu
    Elu
    Selu
    Heaviside
    SmoothHeaviside
    Sawtooth
    Softplus
    Tansig
    Sigmoid
```