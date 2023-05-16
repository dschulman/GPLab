mutable struct RepGPRegressor
    kernel::Kernel
end

RepGPRegressor() = RepGPRegressor(Matern52Kernel())

struct RepGPRegression
    lml
    params
    kernel::Kernel
    Cchol::Cholesky
    α
end

function fit(gpr::RepGPRegressor, x::AbstractMatrix, y::AbstractVector)
    initθ, unflatten = ParameterHandling.value_flatten(_init_params(x))
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
    return RepGPRegression(lml, params, kernel, Cchol, α)
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
