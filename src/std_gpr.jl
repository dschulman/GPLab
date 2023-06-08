mutable struct GPRegressor{Tk<:Kernel}
    kernel::Tk
end

GPRegressor() = GPRegressor(Matern52Kernel())

struct GPRegression
    lml
    params
    posterior
end

function init_params(::GPRegressor, x, y; assume_noise=0.5)
    yvar = var(y)
    return (
        mean = mean(y),
        noise_var = positive(yvar * assume_noise),
        k = _init_kernel_params(x; init_var=yvar * (1 - assume_noise))
    )
end

function fit(gpr::GPRegressor, x::AbstractMatrix, y::AbstractVector)
    initθ, unflatten = ParameterHandling.value_flatten(init_params(gpr, x, y))
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

function predict_latent(gpfit::GPRegression, xtest::AbstractMatrix)
    return marginals(gpfit.posterior(RowVecs(xtest)))
end

function predict(gpfit::GPRegression, xtest::AbstractMatrix)
    return marginals(gpfit.posterior(RowVecs(xtest), gpfit.params.noise_var))
end
