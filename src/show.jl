# ── Utilities / Decomposition ─────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Utilities.Decomposition)
    if get(io, :compact, false)
        show(io, x)
    else
        m, n = size(x.full_matrix)
        println(io, "Decomposition")
        println(io, "  size  : ", m, " × ", n)
        println(io, "  method: ", nameof(typeof(x.decomposition)))
    end
end

function Base.show(io::IO, x::Utilities.Decomposition)
    m, n = size(x.full_matrix)
    print(io, "Decomposition (", m, "×", n, ", ", nameof(typeof(x.decomposition)), ")")
end

function Base.summary(io::IO, x::Utilities.Decomposition)
    m, n = size(x.full_matrix)
    print(io, "Decomposition (", m, "×", n, ", ", nameof(typeof(x.decomposition)), ")")
end

# ── Samplers ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Samplers.Sampler)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "Sampler")
        println(io, "  rng : ", nameof(typeof(x.rng)))
        println(io, "  dist: ", nameof(typeof(x.parameter_distribution)))
    end
end

function Base.show(io::IO, x::Samplers.Sampler)
    print(io, "Sampler (", nameof(typeof(x.rng)), ")")
end

function Base.summary(io::IO, x::Samplers.Sampler)
    print(io, "Sampler (", nameof(typeof(x.rng)), ")")
end

# ── ScalarFeatures ────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Features.ScalarFeature)
    if get(io, :compact, false)
        show(io, x)
    else
        n = x.n_features
        println(io, "ScalarFeature")
        println(io, "  n_features     : ", n)
        println(io, "  scalar_function: ", nameof(typeof(x.scalar_function)))
        if !isnothing(x.feature_parameters)
            np = length(x.feature_parameters)
            println(io, "  feature_params : Dict with ", np, " entr", np == 1 ? "y" : "ies")
        end
    end
end

function Base.show(io::IO, x::Features.ScalarFeature)
    print(
        io,
        "ScalarFeature (",
        x.n_features,
        " feature",
        x.n_features == 1 ? "" : "s",
        ", ",
        nameof(typeof(x.scalar_function)),
        ")",
    )
end

function Base.summary(io::IO, x::Features.ScalarFeature)
    print(
        io,
        "ScalarFeature (",
        x.n_features,
        " feature",
        x.n_features == 1 ? "" : "s",
        ", ",
        nameof(typeof(x.scalar_function)),
        ")",
    )
end

# ── VectorFeatures ────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Features.VectorFeature)
    if get(io, :compact, false)
        show(io, x)
    else
        n = x.n_features
        println(io, "VectorFeature")
        println(io, "  n_features     : ", n)
        println(io, "  output_dim     : ", x.output_dim)
        println(io, "  scalar_function: ", nameof(typeof(x.scalar_function)))
        if !isnothing(x.feature_parameters)
            np = length(x.feature_parameters)
            println(io, "  feature_params : Dict with ", np, " entr", np == 1 ? "y" : "ies")
        end
    end
end

function Base.show(io::IO, x::Features.VectorFeature)
    print(
        io,
        "VectorFeature (",
        x.n_features,
        " features → ",
        x.output_dim,
        "D, ",
        nameof(typeof(x.scalar_function)),
        ")",
    )
end

function Base.summary(io::IO, x::Features.VectorFeature)
    print(
        io,
        "VectorFeature (",
        x.n_features,
        " features → ",
        x.output_dim,
        "D, ",
        nameof(typeof(x.scalar_function)),
        ")",
    )
end

# ── RandomFeatureMethod / Fit ─────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Methods.RandomFeatureMethod)
    if get(io, :compact, false)
        show(io, x)
    else
        bs = x.batch_sizes
        println(io, "RandomFeatureMethod")
        println(io, "  feature type  : ", nameof(typeof(x.random_feature)))
        println(io, "  n_features    : ", x.random_feature.n_features)
        println(io, "  regularization: ", nameof(typeof(x.regularization)))
        println(io, "  batch_sizes   : train=", bs["train"], ", test=", bs["test"], ", feature=", bs["feature"])
        println(io, "  threading     : ", x.tullio_threading)
    end
end

function Base.show(io::IO, x::Methods.RandomFeatureMethod)
    print(io, "RandomFeatureMethod (", nameof(typeof(x.random_feature)), ", n=", x.random_feature.n_features, ")")
end

function Base.summary(io::IO, x::Methods.RandomFeatureMethod)
    print(io, "RandomFeatureMethod (", nameof(typeof(x.random_feature)), ", n=", x.random_feature.n_features, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::Methods.Fit)
    if get(io, :compact, false)
        show(io, x)
    else
        n = length(x.coeffs)
        m, k = size(x.feature_factors.full_matrix)
        println(io, "Fit")
        println(io, "  coeffs         : ", n, " element", n == 1 ? "" : "s")
        println(io, "  feature_factors: Decomposition (", m, "×", k, ")")
        println(io, "  regularization : ", nameof(typeof(x.regularization)))
    end
end

function Base.show(io::IO, x::Methods.Fit)
    n = length(x.coeffs)
    print(io, "Fit (", n, " coeff", n == 1 ? "" : "s", ")")
end

function Base.summary(io::IO, x::Methods.Fit)
    n = length(x.coeffs)
    print(io, "Fit (", n, " coeff", n == 1 ? "" : "s", ")")
end
