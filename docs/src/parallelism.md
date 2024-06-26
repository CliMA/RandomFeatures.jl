# Explicit bottlenecks

### Explicit bottlenecks
- By far the highest computational demand for high-dimensional and/or large data systems is the building of the system matrix, particularly the multiplication ``\Phi^T\Phi`` (for scalar) and ``\Phi_{n,i,m}\Phi_{n,j,m}``. These can be accelerated by multithreading (see below)
- For large number of features, the inversion `inv(factorization(system_matrix))` is noticeable, though typically still small when in the regime of `n_features << n_samples * output_dim`.
- For the vector case, the square output-dimensional regularization matrix ``\Lambda`` must be inverted. For high-dimensional spaces, diagonal approximation will avoid this.
- Prediction bottlenecks are largely due to allocations and matrix multiplications. Please see our `predict!()` methods which allow for users to pass in preallocated of arrays. This is very beneficial for repeated predictions.
- For systems without enough regularization, positive definiteness may need to be enforced. If done too often, it has non-negligible cost, as it involves calculating eigenvalues of the non p.d. matrix (particularly `abs(min(eigval)`) that is then added to the diagonal. It is better to add more regularization into ``\Lambda``

### Implicit bottlenecks
- The optimization of hyperparameters is a costly operation that may require construction and evaluation of thousands of `RandomFeatureMethod`s. The dimensionality (i.e. complexity) of this task will depend on how many free parameters are taken to be within a distribution though. ``\mathcal{O}(1000s)`` parameters may take even hours to optimize (on multiple threads).

# Parallelism/memory
- We make use of [`Tullio.jl`](https://github.com/mcabbott/Tullio.jl) which comes with in-built memory management. We are phasing out our own batches in favour of using this for now.
- [`Tullio.jl`](https://github.com/mcabbott/Tullio.jl) comes with multithreading routines, Simply call the code with `julia --project -t n_threads` to take advantage of this. Depending on problem size you may wish to use your own external threading, Tullio will greedily steal threads in this case. To prevent this interference we provide a keyword argument: 
```julia
RandomFeatureMethod(... ; tullio_threading=false) # serial threading during the build and fit! methods
predict(...; tullio_threading=false) # serial threading for prediction
predict!(...; tullio_threading=false) # serial threading for in-place prediction
```
An example where `tullio_threading=false` is useful is when optimizing hyperparameters with ensemble methods (see our examples), here one could use threading/multiprocessing approaches across ensemble members to make better use of the embarassingly parallel framework (e.g. see this page for [EnsembleKalmanProcessess: Parallelism and HPC](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/parallel_hpc/). 

!!! note
    We do not yet have GPU functionality

