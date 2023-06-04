abstract type Likelihood end

_each_param(lik::Likelihood, θ) = eachcol(reshape(θ, :, nparam(lik)))

struct GaussianLikelihood <: Likelihood end

nparam(lik::GaussianLikelihood) = 2

compute_statistics(::GaussianLikelihood, y) = y

function init_latent(::GaussianLikelihood, y)
    n = length(y)
    m, v = mean_and_var(y; corrected=false)
    return [fill(m, n) ; fill(log(v), n)]
end

function loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    return -0.5 * (
        (length(y) * log2π) +
        sum(logvar) +
        sum((y .- mean).^2 .* exp.(-logvar))
    )
end

function grad_loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    dmean = z .* prec
    dlogvar = @. -0.5 + (0.5 * z * z * prec)
    return [dmean ; dlogvar]
end

function hess_loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    ddm = Diagonal(-prec)
    dm_dlv = Diagonal(-z .* prec)
    ddlv = Diagonal(-0.5 .* z .* z .* prec)
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info(lik::GaussianLikelihood, θ, y)
    logvar = _each_param(lik, θ)[2]
    return Diagonal([exp.(-logvar) ; fill(0.5, size(logvar))])
end

function postpred(::GaussianLikelihood, θ)
    emean, elogvar = mean(θ)
    vmean, vlogvar = var(θ)
    yvar = vmean + exp(elogvar + (vlogvar / 2))
    return Normal(emean, sqrt(yvar))
end
