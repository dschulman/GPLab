mutable struct RepGPRegressor{Tk<:Kernel}
    kernel::Tk
end

RepGPRegressor() = RepGPRegressor(Matern52Kernel())

struct RepGPRegression{Tk<:Kernel}
    lml
    params
    kernel::Tk
    Cchol::Cholesky
    x
    α
end

function init_params(::RepGPRegressor, x, y, w; assume_noise=0.5)
    u = mean(y.u, w)
    v = mean(y.v, w)
    return (
        mean = u,
        noise_var = positive(v * assume_noise),
        k = _init_kernel_params(x; init_var = v * (1 - assume_noise))
    )
end

function fit(
    gpr::RepGPRegressor, x::AbstractMatrix, y::AbstractVector;
    w=fweights(length.(y))
)
    ystats = _sufficient_statistics(y)
    initθ, unflatten = ParameterHandling.value_flatten(init_params(gpr, x, ystats, w))
    res = Optim.optimize(
        _zygote_fg!(θ -> _rep_nlml(gpr.kernel, unflatten(θ), x, ystats, w)),
        initθ,
        Optim.LBFGS(alphaguess=LineSearches.InitialStatic(scaled=true))
    )
    lml = - minimum(res)
    params = unflatten(Optim.minimizer(res))
    kernel = _kernel(gpr.kernel, params.k)
    Cchol = _Cchol(kernel, x, ystats, w, params.noise_var)
    α = Cchol \ (ystats.u .- params.mean)
    return RepGPRegression(lml, params, kernel, Cchol, x, α)
end

function _sufficient_statistics(y::AbstractVector)
    u = mean.(y)
	v = varm.(y, u; corrected=false)
	return (u=u, v=v)
end

function _rep_nlml(base_kernel::Kernel, params, x, ystats, w)
    kernel = _kernel(base_kernel, params.k)
    n = size(x, 1)
    Cchol = _Cchol(kernel, x, ystats, w, params.noise_var)
    z = ystats.u .- params.mean
	return - _logV(ystats, w, params.noise_var) + (1/2) * (
		(n * log2π) +
		logdet(Cchol) +
		(z' * (Cchol \ z))
	)
end

function _Cchol(kernel::Kernel, x, ystats, w, noise_var)
    C = kernelmatrix(kernel, RowVecs(x)) + Diagonal(noise_var ./ w)
    return cholesky(C)
end

function _logV(ystats, w, noise_var)
	n = length(w)
	sum_w = sum(w)
    W = Diagonal(w)
	return -(1/2) * (
		((sum_w - n) * (log2π + log(noise_var))) +
		logdet(W) +
		((w ⋅ ystats.v) / noise_var)
	)
end

function _predict(gpfit::RepGPRegression, xtest::AbstractMatrix, noise_var)
    Kstar = kernelmatrix(gpfit.kernel, RowVecs(gpfit.x), RowVecs(xtest))
    Kstarstar = kernelmatrix_diag(gpfit.kernel, RowVecs(xtest))
    mean = gpfit.params.mean .+ (Kstar' * gpfit.α)
    var = Kstarstar .- AbstractGPs.diag_Xt_invA_X(gpfit.Cchol, Kstar) .+ noise_var
    return mean, var
end

function predict_latent(gpfit::RepGPRegression, xtest::AbstractMatrix)
    return _predict(gpfit, xtest, 0.0)
end

function predict(gpfit::RepGPRegression, xtest::AbstractMatrix)
    return _predict(gpfit, xtest, gpfit.params.noise_var)
end
