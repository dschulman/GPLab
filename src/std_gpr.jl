mutable struct GPRegressor{Tk<:Kernel}
    kernel::Tk
end

GPRegressor() = GPRegressor(Matern52Kernel())

struct GPRegression
    lml
    params
    posterior
end

function fit(gpr::GPRegressor, x::AbstractMatrix, y::AbstractVector)
    initθ, unflatten = ParameterHandling.value_flatten(_init_params(x))
    res = Optim.optimize(
        _zygote_fg!(θ -> _std_nlml(gpr.kernel, unflatten(θ), x, y)),
        initθ,
        Optim.LBFGS(alphaguess=LineSearches.InitialStatic(scaled=true))
    )
    lml = - minimum(res)
    params = unflatten(Optim.minimizer(res))
    f = _gp(gpr.kernel, params)
    post = posterior(f(RowVecs(x), params.noise_var), y)
    return GPRegression(lml, params, post)
end

function _std_nlml(base_kernel::Kernel, params, x, y)
    f = _gp(base_kernel, params)
    return -logpdf(f(RowVecs(x), params.noise_var), y)
end

function _gp(base_kernel::Kernel, params)
    return GP(params.mean, _kernel(base_kernel, params.k))
end
