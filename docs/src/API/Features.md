# Features

```@meta
CurrentModule = RandomFeatures.Features
```

## Abstract types

```@docs
RandomFeature
```

## Getter functions

```@docs
get_scalar_function
get_feature_sampler
get_feature_sample
get_n_features
get_feature_parameters
get_output_dim
sample(rf::RandomFeature)
```

## [Scalar Features](@id scalar-features)

```@docs
ScalarFeature
ScalarFourierFeature
ScalarNeuronFeature
build_features(rf::ScalarFeature, inputs::AbstractMatrix, batch_feature_idx::AbstractVector)
```

## [Vector Features](@id vector-features)

```@docs
VectorFeature
VectorFourierFeature
VectorNeuronFeature
build_features(rf::VectorFeature, inputs::AbstractMatrix, batch_feature_idx::AbstractVector)
```

## [Scalar Functions](@id scalar-functions)

```@docs
ScalarFunction
ScalarActivation
apply_scalar_function
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