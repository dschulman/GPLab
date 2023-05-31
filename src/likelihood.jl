abstract type Likelihood end

_each_param(lik::Likelihood, θ) = eachcol(reshape(θ, :, nparam(lik)))

struct Gaussian <: Likelihood end

nparam(lik::Gaussian) = 2

compute_statistics(::Gaussian, y) = y

function init_latent(::Gaussian, y)
    n = length(y)
    m, v = mean_and_var(y; corrected=false)
    return [fill(m, n) ; fill(log(v), n)]
end

function loglik(lik::Gaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    return -0.5 * (
        (length(y) * log2π) +
        sum(logvar) +
        sum((y .- mean).^2 .* exp.(-logvar))
    )
end

function grad_loglik(lik::Gaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    dmean = z .* prec
    dlogvar = @. -0.5 + (0.5 * z * z * prec)
    return [dmean ; dlogvar]
end

function hess_loglik(lik::Gaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    ddm = Diagonal(-prec)
    dm_dlv = Diagonal(-z .* prec)
    ddlv = Diagonal(-0.5 .* z .* z .* prec)
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info(lik::Gaussian, θ)
    logvar = _each_param(lik, θ)[2]
    return Diagonal([exp.(-logvar) ; fill(0.5, size(logvar))])
end
