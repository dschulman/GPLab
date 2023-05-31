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

function init_params(::RepGPRegressor, x::AbstractMatrix)
    return (
        mean = 0.0,
        noise_var = positive(1.0),
        k = _init_kernel_params(x)
    )
end

function fit(gpr::RepGPRegressor, x::AbstractMatrix, y::AbstractVector)
    initθ, unflatten = ParameterHandling.value_flatten(init_params(gpr, x))
    ystats = _sufficient_statistics(y)
    res = Optim.optimize(
        _zygote_fg!(θ -> _rep_nlml(gpr.kernel, unflatten(θ), x, ystats)),
        initθ,
        Optim.LBFGS(alphaguess=LineSearches.InitialStatic(scaled=true))
    )
    lml = - minimum(res)
    params = unflatten(Optim.minimizer(res))
    kernel = _kernel(gpr.kernel, params.k)
    Cchol = _Cchol(kernel, x, ystats, params.noise_var)
    α = Cchol \ (ystats.u .- params.mean)
    return RepGPRegression(lml, params, kernel, Cchol, x, α)
end

function _sufficient_statistics(y::AbstractVector)
    u = mean.(y)
	v = varm.(y, u; corrected=false)
	return (u=u, v=v, m=length.(y))
end

function _rep_nlml(base_kernel::Kernel, params, x, ystats)
    kernel = _kernel(base_kernel, params.k)
    n = size(x, 1)
    Cchol = _Cchol(kernel, x, ystats, params.noise_var)
    z = ystats.u .- params.mean
	return - _logV(ystats, params.noise_var) + (1/2) * (
		(n * log2π) +
		logdet(Cchol) +
		(z' * (Cchol \ z))
	)
end

function _Cchol(kernel::Kernel, x, ystats, noise_var)
    C = kernelmatrix(kernel, RowVecs(x)) + Diagonal(noise_var ./ ystats.m)
    return cholesky(C)
end

function _logV(ystats, noise_var)
	n = length(ystats.m)
	sum_m = sum(ystats.m)
	M = Diagonal(ystats.m)
	return -(1/2) * (
		((sum_m - n) * (log2π + log(noise_var))) +
		logdet(M) +
		((ystats.m ⋅ ystats.v) / noise_var)
	)
end

function predict_latent(gpfit::RepGPRegression, xtest::AbstractMatrix)
    Kstar = kernelmatrix(gpfit.kernel, RowVecs(gpfit.x), RowVecs(xtest))
    fstar = gpfit.params.mean .+ (Kstar' * gpfit.α)
    return reshape(fstar, :, 1)
end

function predict(gpfit::RepGPRegression, xtest::AbstractMatrix)
    Kstar = kernelmatrix(gpfit.kernel, RowVecs(gpfit.x), RowVecs(xtest))
    Kstarstar = kernelmatrix_diag(gpfit.kernel, RowVecs(xtest))
    mean = gpfit.params.mean .+ (Kstar' * gpfit.α)
    var = Kstarstar .- AbstractGPs.diag_Xt_invA_X(gpfit.Cchol, Kstar) .+ gpfit.params.noise_var
    return Normal.(mean, sqrt.(var))
end
