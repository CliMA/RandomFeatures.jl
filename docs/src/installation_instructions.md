# Installation

RandomFeatures.jl is a not a registered Julia package. To install perform the following in the `julia` command prompt

```julia
julia> ]
(v1.8) pkg> add https://github.com/CliMA/RandomFeatures.jl
(v1.8) pkg> instantiate
```

This will install the latest version of the package Git repository
    
You can run the tests via the package manager by:

```julia
julia> ]
(v1.8) pkg> test RandomFeatures
```

### Cloning the repository

If you are interested in getting your hands dirty and modifying the code then, you can also
clone the repository and then instantiate, e.g.,

```
> cd RandomFeatures.jl
> julia --project -e 'using Pkg; Pkg.instantiate()'
```

!!! info "Do I need to clone the repository?"
    Most times, cloning the repository in not necessary. If you only want to use the package's
    functionality, adding the packages as a dependency on your project is enough.

### Running the test suite

You can run the package's tests:

```
> julia --project -e 'using Pkg; Pkg.test()'
```
Alternatively, you can do this from within the repository:
```
> julia --project
julia> ]
(RandomFeatures) pkg> test
```

### Building the documentation locally

Once the project is built, you can build the project documentation under the `docs/` sub-project:

```
> julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
> julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`

### Running repository examples

We have a selection of examples, found within the `examples/` directory to demonstrate different use of our toolbox.
Each example directory contains a `Project.toml`

To build with the latest `RandomFeatures.jl` release:
```
> cd examples/example-name/
> julia --project -e 'using Pkg; Pkg.instantiate()'
> julia --project example-file-name.jl
```
If you wish to run a local modified version of `RandomFeatures.jl` then try the following (starting from the `RandomFeatures.jl` package root)
```
> cd examples/example-name/
> julia --project 
> julia> ]
> (example-name)> rm RandomFeatures.jl
> (example-name)> dev ../..
> (example-name)> instantiate
```
followed by
```
> julia --project example-file-name.jl
```
