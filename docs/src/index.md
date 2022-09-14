# RandomFeatures

A julia package to construct and apply random feature methods for regression.
RandomFeatures can be viewed as an approximation of kernel methods. They can be used both as a substitution in Kernel ridge regression and Gaussian Process regresison. 

Module         | Purpose
---------------|-----------------------------------------------------------------------------------------
RandomFeatures | Container of all tools
Samplers       | Samplers for constrained probability distributions 
Features       | Builds feature functions from input data
Methods        | Fits features to output data, and prediction on new inputs  
Utilities      | Utilities to aid batching, and matrix decompositions 


## Highlights
- A flexible probability distribution backend with which to sample features, with a comprehensive API
- A library of modular scalar functions to choose from
- Methods for solving ridge regression or Gaussian Process regression problem, with functions for producing predictive means and (co)variances using fitted features. 
- Examples that demonstrate using the package `EnsembleKalmanProcesses.jl` to optimize hyperparameters of the probability distribution.


## Authors

`RandomFeatures.jl` is being developed by the [Climate Modeling
Alliance](https://clima.caltech.edu). The main developers are Oliver R. A. Dunbar and Thomas Jackson, with acknowledgement that the code was based on a python repository developed by Oliver R. A. Dunbar, Maya Mutic, and Nicholas H. Nelsen.
